#!/usr/bin/env python3
"""
Slideshow Builder — Cinematic photo slideshow with Ken Burns effects,
crossfade transitions, and a music soundtrack.

Usage:
    python slideshow_builder.py                        # standard run
    python slideshow_builder.py --dry-run              # preview without rendering
    python slideshow_builder.py --setlist custom.txt   # use a custom setlist
    python slideshow_builder.py --no-cache             # ignore cached clips
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from moviepy import (
    afx,
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_audioclips,
    vfx,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("slideshow_builder")
logger.setLevel(logging.DEBUG)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s",
                                         datefmt="%H:%M:%S"))
logger.addHandler(_console)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SETLIST = PROJECT_ROOT / "final-setlist.txt"
PHOTO_DIR = PROJECT_ROOT / "movie-content" / "pic-playlist"
MUSIC_DIR = PROJECT_ROOT / "movie-content" / "music-playlist"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / ".slideshow_cache"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".wma"}

KEN_BURNS_STYLES = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]
KEN_BURNS_INTENSITIES = {
    "subtle":   0.05,
    "medium":   0.12,
    "dramatic": 0.22,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PhotoEntry:
    path: Path
    duration: Optional[float] = None      # None → use default
    effect: Optional[str] = None          # None → use global setting

@dataclass
class MusicEntry:
    path: Path
    fade_in: Optional[float] = None
    fade_out: Optional[float] = None

@dataclass
class SlideshowSettings:
    output_filename: str = "brothers_slideshow.mp4"
    output_width: int = 1920
    output_height: int = 1080
    default_photo_duration: float = 5.0
    photo_duration_variance: float = 0.5
    crossfade_duration: float = 1.0
    ken_burns_intensity: str = "medium"
    ken_burns_style: str = "random"
    video_fps: int = 24
    fade_in_duration: float = 2.0
    fade_out_duration: float = 3.0

@dataclass
class SlideshowConfig:
    settings: SlideshowSettings = field(default_factory=SlideshowSettings)
    photos: List[PhotoEntry] = field(default_factory=list)
    music: List[MusicEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Setlist Parser
# ---------------------------------------------------------------------------
def parse_setlist(setlist_path: Path) -> SlideshowConfig:
    """Parse the final-setlist.txt into a SlideshowConfig."""
    config = SlideshowConfig()

    if not setlist_path.exists():
        logger.warning("Setlist file not found: %s — using all defaults", setlist_path)
        config.photos = _auto_discover_photos()
        config.music = _auto_discover_music()
        return config

    text = setlist_path.read_text(encoding="utf-8")
    current_section = None
    photo_entries: List[PhotoEntry] = []
    music_entries: List[MusicEntry] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Detect section headers
        if line.upper() == "[SETTINGS]":
            current_section = "settings"
            continue
        elif line.upper() == "[PHOTOS]":
            current_section = "photos"
            continue
        elif line.upper() == "[MUSIC]":
            current_section = "music"
            continue

        if current_section == "settings":
            _parse_setting(config.settings, line)
        elif current_section == "photos":
            entry = _parse_photo_line(line)
            if entry:
                photo_entries.append(entry)
        elif current_section == "music":
            entry = _parse_music_line(line)
            if entry:
                music_entries.append(entry)

    # Auto-discover if sections are empty
    config.photos = photo_entries if photo_entries else _auto_discover_photos()
    config.music = music_entries if music_entries else _auto_discover_music()

    return config


def _parse_setting(settings: SlideshowSettings, line: str):
    """Parse a single key = value setting line."""
    if "=" not in line:
        return
    key, _, value = line.partition("=")
    key = key.strip().lower()
    value = value.strip()

    # Strip inline comments
    if "#" in value:
        value = value[:value.index("#")].strip()

    if key == "output_filename":
        settings.output_filename = value
    elif key == "output_resolution":
        m = re.match(r"(\d+)\s*x\s*(\d+)", value, re.IGNORECASE)
        if m:
            settings.output_width = int(m.group(1))
            settings.output_height = int(m.group(2))
        else:
            logger.warning("Invalid resolution format '%s', using default 1920x1080", value)
    elif key == "default_photo_duration":
        settings.default_photo_duration = float(value)
    elif key == "photo_duration_variance":
        settings.photo_duration_variance = float(value)
    elif key == "crossfade_duration":
        settings.crossfade_duration = float(value)
    elif key == "ken_burns_intensity":
        if value.lower() in KEN_BURNS_INTENSITIES:
            settings.ken_burns_intensity = value.lower()
        else:
            logger.warning("Unknown Ken Burns intensity '%s', using 'medium'", value)
    elif key == "ken_burns_style":
        if value.lower() in KEN_BURNS_STYLES + ["random"]:
            settings.ken_burns_style = value.lower()
        else:
            logger.warning("Unknown Ken Burns style '%s', using 'random'", value)
    elif key == "video_fps":
        settings.video_fps = int(value)
    elif key == "fade_in_duration":
        settings.fade_in_duration = float(value)
    elif key == "fade_out_duration":
        settings.fade_out_duration = float(value)


def _parse_photo_line(line: str) -> Optional[PhotoEntry]:
    """Parse a photo line like  filename.jpg | duration=8 | effect=zoom_in"""
    parts = [p.strip() for p in line.split("|")]
    filename = parts[0]
    path = PHOTO_DIR / filename

    if not path.exists():
        logger.warning("Photo not found, skipping: %s", path)
        return None

    entry = PhotoEntry(path=path)
    for part in parts[1:]:
        kv = part.split("=", 1)
        if len(kv) != 2:
            continue
        k, v = kv[0].strip().lower(), kv[1].strip()
        if k == "duration":
            entry.duration = float(v)
        elif k == "effect":
            if v.lower() in KEN_BURNS_STYLES + ["random"]:
                entry.effect = v.lower()
            else:
                logger.warning("Unknown effect '%s' for %s, using global default", v, filename)
    return entry


def _parse_music_line(line: str) -> Optional[MusicEntry]:
    """Parse a music line like  song.mp3 | fade_in=2 | fade_out=3"""
    parts = [p.strip() for p in line.split("|")]
    filename = parts[0]
    path = MUSIC_DIR / filename

    if not path.exists():
        logger.warning("Music file not found, skipping: %s", path)
        return None

    entry = MusicEntry(path=path)
    for part in parts[1:]:
        kv = part.split("=", 1)
        if len(kv) != 2:
            continue
        k, v = kv[0].strip().lower(), kv[1].strip()
        if k == "fade_in":
            entry.fade_in = float(v)
        elif k == "fade_out":
            entry.fade_out = float(v)
    return entry


def _auto_discover_photos() -> List[PhotoEntry]:
    """Auto-discover all image files from PHOTO_DIR, sorted alphabetically."""
    if not PHOTO_DIR.exists():
        logger.error("Photo directory does not exist: %s", PHOTO_DIR)
        return []
    photos = sorted(
        [f for f in PHOTO_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower()
    )
    logger.info("Auto-discovered %d photos from %s", len(photos), PHOTO_DIR)
    return [PhotoEntry(path=p) for p in photos]


def _auto_discover_music() -> List[MusicEntry]:
    """Auto-discover all audio files from MUSIC_DIR, sorted alphabetically."""
    if not MUSIC_DIR.exists():
        logger.error("Music directory does not exist: %s", MUSIC_DIR)
        return []
    tracks = sorted(
        [f for f in MUSIC_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS],
        key=lambda p: p.name.lower()
    )
    logger.info("Auto-discovered %d music tracks from %s", len(tracks), MUSIC_DIR)
    return [MusicEntry(path=p) for p in tracks]


# ---------------------------------------------------------------------------
# Ken Burns Effect Engine
# ---------------------------------------------------------------------------
def _pick_style(style_setting: str) -> str:
    """Resolve 'random' to a concrete style, or pass through."""
    if style_setting == "random":
        return random.choice(KEN_BURNS_STYLES)
    return style_setting


def ken_burns_at_t(image_array: np.ndarray, t_norm: float,
                   target_w: int, target_h: int,
                   style: str, intensity_name: str) -> np.ndarray:
    """
    Compute a single Ken Burns frame at normalised time t_norm ∈ [0.0, 1.0].
    Called on-demand by MoviePy during export — no frames are pre-allocated.
    The source image must already be oversized relative to the target.
    """
    travel = KEN_BURNS_INTENSITIES.get(intensity_name, 0.12)
    src_h, src_w = image_array.shape[:2]
    t = max(0.0, min(t_norm, 1.0))

    if style == "zoom_in":
        # Start wide, end narrow
        scale = 1.0 - travel * t
        crop_w = int(src_w * scale)
        crop_h = int(src_h * scale)
        x0 = (src_w - crop_w) // 2
        y0 = (src_h - crop_h) // 2

    elif style == "zoom_out":
        # Start narrow, end wide
        scale = 1.0 - travel * (1 - t)
        crop_w = int(src_w * scale)
        crop_h = int(src_h * scale)
        x0 = (src_w - crop_w) // 2
        y0 = (src_h - crop_h) // 2

    elif style == "pan_left":
        crop_w = int(src_w * (1.0 - travel))
        crop_h = src_h
        max_offset = src_w - crop_w
        x0 = int(max_offset * (1 - t))
        y0 = 0

    elif style == "pan_right":
        crop_w = int(src_w * (1.0 - travel))
        crop_h = src_h
        max_offset = src_w - crop_w
        x0 = int(max_offset * t)
        y0 = 0

    elif style == "pan_up":
        crop_w = src_w
        crop_h = int(src_h * (1.0 - travel))
        max_offset = src_h - crop_h
        x0 = 0
        y0 = int(max_offset * (1 - t))

    elif style == "pan_down":
        crop_w = src_w
        crop_h = int(src_h * (1.0 - travel))
        max_offset = src_h - crop_h
        x0 = 0
        y0 = int(max_offset * t)

    else:
        # Fallback — no motion
        crop_w, crop_h = src_w, src_h
        x0, y0 = 0, 0

    # Clamp bounds
    crop_w = max(crop_w, 1)
    crop_h = max(crop_h, 1)
    x0 = max(0, min(x0, src_w - crop_w))
    y0 = max(0, min(y0, src_h - crop_h))

    cropped = image_array[y0:y0 + crop_h, x0:x0 + crop_w]

    # Resize to target via Pillow (high quality)
    pil_img = Image.fromarray(cropped)
    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil_img)


# ---------------------------------------------------------------------------
# Image preprocessing — cover-mode resize with oversizing for Ken Burns
# ---------------------------------------------------------------------------
def load_and_prepare_image(photo_path: Path, target_w: int, target_h: int,
                           intensity_name: str) -> np.ndarray:
    """
    Load an image and scale it to fit the target resolution. Fill the
    remaining letterbox/pillarbox space with a blurred, darkened version
    of the same image, with extra margin for Ken Burns motion.
    """
    travel = KEN_BURNS_INTENSITIES.get(intensity_name, 0.12)
    # Oversize factor: add enough margin so the full travel range fits
    oversize = 1.0 + travel + 0.02  # small extra safety margin

    desired_w = int(target_w * oversize)
    desired_h = int(target_h * oversize)

    img = Image.open(photo_path).convert("RGB")

    # EXIF orientation fix
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    from PIL import ImageFilter, ImageEnhance

    iw, ih = img.size

    # 1. Create the blurred background (cover mode)
    bg_scale = max(desired_w / iw, desired_h / ih)
    bg_w = int(iw * bg_scale)
    bg_h = int(ih * bg_scale)
    bg_img = img.resize((bg_w, bg_h), Image.LANCZOS)

    # Centre crop the background
    left = (bg_w - desired_w) // 2
    top = (bg_h - desired_h) // 2
    bg_img = bg_img.crop((left, top, left + desired_w, top + desired_h))

    # Apply heavy blur and darken slightly
    bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=40))
    enhancer = ImageEnhance.Brightness(bg_img)
    bg_img = enhancer.enhance(0.5)  # Darken to 50% for better contrast

    # 2. Create the foreground (fit mode)
    fg_scale = min(desired_w / iw, desired_h / ih)
    fg_w = int(iw * fg_scale)
    fg_h = int(ih * fg_scale)
    fg_img = img.resize((fg_w, fg_h), Image.LANCZOS)

    # 3. Composite foreground over blurred background
    paste_x = (desired_w - fg_w) // 2
    paste_y = (desired_h - fg_h) // 2
    bg_img.paste(fg_img, (paste_x, paste_y))

    return np.array(bg_img)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_key(photo_path: Path, intensity: str, target_w: int, target_h: int) -> str:
    """Deterministic cache key for a prepared (pre-processed) photo image."""
    raw = f"{photo_path}|{intensity}|{target_w}x{target_h}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cached_image(cache_key: str) -> Optional[np.ndarray]:
    """Try to load a cached prepared image from disk."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_cached_image(cache_key: str, image: np.ndarray):
    """Save a prepared image to disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(image, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.debug("Could not write cache for %s: %s", cache_key, exc)


# ---------------------------------------------------------------------------
# Photo Clip Builder
# ---------------------------------------------------------------------------
def build_photo_clips(config: SlideshowConfig, use_cache: bool = True
                      ) -> List[ImageClip]:
    """
    Build a list of MoviePy ImageClips with Ken Burns effect applied on-demand.
    Images are loaded lazily during export rather than pre-loaded into memory,
    so arbitrarily large photo sets can be handled without hitting RAM limits.
    At most 2 prepared images live in RAM at once (during crossfade overlaps).
    """
    settings = config.settings
    tw, th = settings.output_width, settings.output_height
    clips: List[ImageClip] = []

    total = len(config.photos)
    for idx, photo in enumerate(config.photos, start=1):
        # Resolve duration
        base_dur = photo.duration if photo.duration is not None else settings.default_photo_duration
        if settings.photo_duration_variance > 0 and photo.duration is None:
            base_dur += random.uniform(-settings.photo_duration_variance,
                                        settings.photo_duration_variance)
        base_dur = max(base_dur, 0.5)  # safety floor

        # Resolve Ken Burns style
        style_setting = photo.effect if photo.effect else settings.ken_burns_style
        style = _pick_style(style_setting)

        intensity = settings.ken_burns_intensity
        fps = settings.video_fps

        logger.info("Processing photo %d/%d: %s — %s, %.1fs",
                     idx, total, photo.path.name, style, base_dur)

        ck = _cache_key(photo.path, intensity, tw, th)

        # Build a lazy frame function — the prepared image is loaded on first
        # frame request and held only for the life of this clip's rendering.
        def make_frame_func(path, dur, sty, intens, _tw, _th, cache_key, _use_cache):
            _img = [None]  # mutable cell; populated on first access
            def get_frame(t):
                if _img[0] is None:
                    cached = _load_cached_image(cache_key) if _use_cache else None
                    if cached is not None:
                        logger.debug("  (cache hit: %s)", path.name)
                        _img[0] = cached
                    else:
                        _img[0] = load_and_prepare_image(path, _tw, _th, intens)
                        if _use_cache:
                            _save_cached_image(cache_key, _img[0])
                t_norm = min(t / max(dur, 1e-6), 1.0)
                return ken_burns_at_t(_img[0], t_norm, _tw, _th, sty, intens)
            return get_frame

        frame_func = make_frame_func(
            photo.path, base_dur, style, intensity, tw, th, ck, use_cache
        )

        # Black placeholder to set clip dimensions without loading the image
        placeholder = np.zeros((th, tw, 3), dtype=np.uint8)
        clip = ImageClip(placeholder, duration=base_dur)
        clip = clip.with_fps(fps)
        clip = clip.with_updated_frame_function(frame_func)

        clips.append(clip)

    return clips


# ---------------------------------------------------------------------------
# Crossfade Assembly
# ---------------------------------------------------------------------------
def assemble_with_crossfades(clips: List[ImageClip],
                              crossfade_dur: float) -> CompositeVideoClip:
    """
    Layer clips with crossfade overlaps using CompositeVideoClip.
    Each successive clip starts `crossfade_dur` seconds before the
    previous clip ends, and both clips are cross-faded in the overlap.
    """
    if not clips:
        raise ValueError("No clips to assemble")

    if len(clips) == 1:
        return CompositeVideoClip([clips[0]])

    # Calculate start times
    start_times = [0.0]
    for i in range(1, len(clips)):
        prev_end = start_times[i - 1] + clips[i - 1].duration
        start_times.append(prev_end - crossfade_dur)

    total_duration = start_times[-1] + clips[-1].duration

    # Apply fade effects and set start times
    layered = []
    for i, (clip, start) in enumerate(zip(clips, start_times)):
        c = clip.with_start(start)
        effects = []
        # Fade in on all clips except the first
        if i > 0:
            effects.append(vfx.CrossFadeIn(crossfade_dur))
        # Fade out on all clips except the last
        if i < len(clips) - 1:
            effects.append(vfx.CrossFadeOut(crossfade_dur))
        if effects:
            c = c.with_effects(effects)
        layered.append(c)

    return CompositeVideoClip(layered, size=(clips[0].w, clips[0].h)).with_duration(total_duration)


# ---------------------------------------------------------------------------
# Music Engine
# ---------------------------------------------------------------------------
def build_audio_track(config: SlideshowConfig, video_duration: float) -> Optional[AudioFileClip]:
    """
    Load, concatenate, and fit the music tracks to the video duration.
    Returns a single AudioFileClip or None if no music.
    """
    if not config.music:
        logger.warning("No music tracks available")
        return None

    settings = config.settings
    audio_clips = []

    for entry in config.music:
        try:
            aclip = AudioFileClip(str(entry.path))
            # Per-track fades
            fade_effects = []
            if entry.fade_in and entry.fade_in > 0:
                fade_effects.append(afx.AudioFadeIn(entry.fade_in))
            if entry.fade_out and entry.fade_out > 0:
                fade_effects.append(afx.AudioFadeOut(entry.fade_out))
            if fade_effects:
                aclip = aclip.with_effects(fade_effects)
            audio_clips.append(aclip)
            logger.info("  Loaded music: %s (%.1fs)", entry.path.name, aclip.duration)
        except Exception as exc:
            logger.warning("  Could not load %s: %s — skipping", entry.path.name, exc)

    if not audio_clips:
        logger.warning("No music tracks could be loaded")
        return None

    # Concatenate
    combined = concatenate_audioclips(audio_clips)
    total_music = combined.duration
    logger.info("Total music duration: %.1fs | Video duration: %.1fs", total_music, video_duration)

    if total_music >= video_duration:
        # Trim and fade out
        combined = combined.subclipped(0, video_duration)
        fade_dur = min(settings.fade_out_duration, video_duration)
        combined = combined.with_effects([afx.AudioFadeOut(fade_dur)])
    else:
        # Loop to fill video duration
        loops_needed = int(video_duration / total_music) + 1
        loop_clips = []
        accumulated = 0.0
        for _ in range(loops_needed):
            for aclip in audio_clips:
                if accumulated >= video_duration:
                    break
                remaining = video_duration - accumulated
                if aclip.duration <= remaining:
                    loop_clips.append(aclip)
                    accumulated += aclip.duration
                else:
                    trimmed = aclip.subclipped(0, remaining)
                    loop_clips.append(trimmed)
                    accumulated += remaining
                    break
            if accumulated >= video_duration:
                break
        combined = concatenate_audioclips(loop_clips)
        combined = combined.subclipped(0, video_duration)
        fade_dur = min(settings.fade_out_duration, video_duration)
        combined = combined.with_effects([afx.AudioFadeOut(fade_dur)])

    return combined


# ---------------------------------------------------------------------------
# Final Assembly
# ---------------------------------------------------------------------------
def build_slideshow(config: SlideshowConfig, use_cache: bool = True,
                     use_gpu: bool = False):
    """Full pipeline: photos → crossfade → music → export."""
    settings = config.settings
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / settings.output_filename

    # Build photo clips
    logger.info("=" * 60)
    logger.info("BUILDING PHOTO CLIPS")
    logger.info("=" * 60)
    t0 = time.time()
    clips = build_photo_clips(config, use_cache=use_cache)
    if not clips:
        logger.error("No photo clips were generated. Exiting.")
        sys.exit(1)
    logger.info("Photo clips built in %.1fs", time.time() - t0)

    # Assemble with crossfades
    logger.info("=" * 60)
    logger.info("ASSEMBLING VIDEO WITH CROSSFADES")
    logger.info("=" * 60)
    video = assemble_with_crossfades(clips, settings.crossfade_duration)

    # Global fade-in from black and fade-out to black
    global_effects = []
    if settings.fade_in_duration > 0:
        global_effects.append(vfx.CrossFadeIn(settings.fade_in_duration))
    if settings.fade_out_duration > 0:
        global_effects.append(vfx.CrossFadeOut(settings.fade_out_duration))
    if global_effects:
        video = video.with_effects(global_effects)

    video_duration = video.duration
    logger.info("Video duration: %.1fs (%.1f minutes)", video_duration, video_duration / 60)

    # Build audio
    logger.info("=" * 60)
    logger.info("BUILDING AUDIO TRACK")
    logger.info("=" * 60)
    audio = build_audio_track(config, video_duration)
    if audio is not None:
        video = video.with_audio(audio)
    else:
        logger.warning("Proceeding with silent video (no audio)")

    # Export
    logger.info("=" * 60)
    logger.info("EXPORTING VIDEO")
    logger.info("=" * 60)
    logger.info("Output: %s", output_path)
    t0 = time.time()

    encoder_used = "libx264 (CPU)"
    if use_gpu:
        logger.info("GPU encoding enabled — trying NVIDIA NVENC H.264")
        try:
            video.write_videofile(
                str(output_path),
                fps=settings.video_fps,
                codec="h264_nvenc",
                audio_codec="aac",
                ffmpeg_params=["-preset", "p4", "-rc", "vbr",
                              "-cq", "23", "-b:v", "0"],
                threads=os.cpu_count() or 4,
                logger="bar",
            )
            encoder_used = "h264_nvenc (GPU)"
        except (IOError, OSError) as exc:
            logger.warning("GPU encoding failed: %s", exc)
            logger.warning("Falling back to CPU encoding (libx264). "
                           "Update your NVIDIA driver to 551.76+ to enable GPU.")
            video.write_videofile(
                str(output_path),
                fps=settings.video_fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                threads=os.cpu_count() or 4,
                logger="bar",
            )
    else:
        video.write_videofile(
            str(output_path),
            fps=settings.video_fps,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=os.cpu_count() or 4,
            logger="bar",
        )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info("Output file : %s", output_path)
    logger.info("Duration    : %.1fs (%.1f minutes)", video_duration, video_duration / 60)
    logger.info("Photos used : %d", len(config.photos))
    logger.info("Music tracks: %d", len(config.music))
    logger.info("Render time : %.1fs (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("Encoder     : %s", encoder_used)


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------
def dry_run(config: SlideshowConfig):
    """Print a full summary without rendering."""
    settings = config.settings

    print("\n" + "=" * 60)
    print("  DRY RUN — Slideshow Preview")
    print("=" * 60)

    print(f"\n  Output file      : {settings.output_filename}")
    print(f"  Resolution       : {settings.output_width}x{settings.output_height}")
    print(f"  FPS              : {settings.video_fps}")
    print(f"  Photo duration   : {settings.default_photo_duration}s "
          f"(±{settings.photo_duration_variance}s)")
    print(f"  Crossfade        : {settings.crossfade_duration}s")
    print(f"  Ken Burns        : {settings.ken_burns_style} / {settings.ken_burns_intensity}")
    print(f"  Fade in          : {settings.fade_in_duration}s")
    print(f"  Fade out         : {settings.fade_out_duration}s")

    # Photo summary
    print(f"\n  PHOTOS ({len(config.photos)} total):")
    total_photo_dur = 0.0
    for i, p in enumerate(config.photos, 1):
        dur = p.duration if p.duration is not None else settings.default_photo_duration
        eff = p.effect if p.effect else settings.ken_burns_style
        print(f"    {i:4d}. {p.path.name:<40s}  {dur:5.1f}s  {eff}")
        total_photo_dur += dur

    # Account for crossfades reducing total duration
    crossfade_overlap = settings.crossfade_duration * max(len(config.photos) - 1, 0)
    estimated_video_dur = total_photo_dur - crossfade_overlap
    estimated_video_dur = max(estimated_video_dur, 0)

    print(f"\n  Estimated video duration: {estimated_video_dur:.1f}s "
          f"({estimated_video_dur / 60:.1f} minutes)")

    # Music summary
    print(f"\n  MUSIC ({len(config.music)} tracks):")
    for i, m in enumerate(config.music, 1):
        extras = ""
        if m.fade_in:
            extras += f"  fade_in={m.fade_in}s"
        if m.fade_out:
            extras += f"  fade_out={m.fade_out}s"
        print(f"    {i:4d}. {m.path.name}{extras}")

    # Try to get actual music durations
    total_music_dur = 0.0
    try:
        for m in config.music:
            aclip = AudioFileClip(str(m.path))
            total_music_dur += aclip.duration
            aclip.close()
    except Exception:
        total_music_dur = -1

    if total_music_dur > 0:
        print(f"\n  Total music duration: {total_music_dur:.1f}s "
              f"({total_music_dur / 60:.1f} minutes)")
        if total_music_dur >= estimated_video_dur:
            print("  → Music will be trimmed to fit video length")
        else:
            print("  → Music will loop to fill video length")
    else:
        print("\n  (Could not determine music durations)")

    print("\n" + "=" * 60)
    print("  Dry run complete — no video rendered.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build a cinematic photo slideshow with Ken Burns effects and music."
    )
    parser.add_argument(
        "--setlist",
        type=str,
        default=str(DEFAULT_SETLIST),
        help="Path to the setlist configuration file (default: final-setlist.txt)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the setlist and timing without rendering video"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force full rebuild, ignoring cached photo clips"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use NVIDIA NVENC GPU acceleration for H.264 encoding"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Only use the first N photos (for quick test builds)"
    )
    args = parser.parse_args()

    setlist_path = Path(args.setlist)
    if not setlist_path.is_absolute():
        setlist_path = PROJECT_ROOT / setlist_path

    logger.info("Parsing setlist: %s", setlist_path)
    config = parse_setlist(setlist_path)

    # Apply --limit
    if args.limit > 0 and config.photos:
        original_count = len(config.photos)
        config.photos = config.photos[:args.limit]
        logger.info("--limit %d: using %d of %d photos",
                    args.limit, len(config.photos), original_count)

    # Validate
    if not config.photos:
        logger.error("No photos found. Cannot build slideshow. Exiting.")
        sys.exit(1)
    if not config.music:
        logger.warning("No music found. Will generate a silent video.")

    # Print summary
    logger.info("Loaded %d photos, %d music tracks", len(config.photos), len(config.music))
    logger.info("Settings: %dx%d @ %dfps, %.1fs/photo, crossfade=%.1fs, "
                "Ken Burns=%s/%s",
                config.settings.output_width, config.settings.output_height,
                config.settings.video_fps,
                config.settings.default_photo_duration,
                config.settings.crossfade_duration,
                config.settings.ken_burns_style,
                config.settings.ken_burns_intensity)

    if args.dry_run:
        dry_run(config)
    else:
        build_slideshow(config, use_cache=not args.no_cache,
                        use_gpu=args.gpu)


if __name__ == "__main__":
    main()
