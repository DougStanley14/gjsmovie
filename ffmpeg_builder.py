#!/usr/bin/env python3
"""
FFmpeg-native slideshow builder — Option 3: Pure GPU Ken Burns.

Replaces the Python/Pillow/NumPy frame-by-frame Ken Burns pipeline with
FFmpeg filter graphs (zoompan, xfade, overlay).  Zero Python overhead
during render; the entire video is produced by FFmpeg subprocesses.

Key optimisations over the MoviePy pipeline:
  - Ken Burns via FFmpeg zoompan filter (native C, not Python per-frame)
  - Parallel clip generation (N ffmpeg processes at once)
  - Crossfades via FFmpeg xfade (filter_complex_script for Windows compat)
  - Blurred/darkened background for non-matching aspect ratios
  - Batched xfade assembly for 100s of photos without hitting cmd limits
  - Global fades folded into the final encode pass (one fewer transcode)

Usage (standalone):
    python ffmpeg_builder.py                        # standard run
    python ffmpeg_builder.py --dry-run              # preview without rendering
    python ffmpeg_builder.py --gpu                  # NVENC encoding
    python ffmpeg_builder.py --limit 5              # quick test with 5 photos

Usage (from slideshow_builder.py):
    python slideshow_builder.py --ffmpeg            # use this pipeline
    python slideshow_builder.py --ffmpeg --gpu      # GPU encode
"""

import argparse
import concurrent.futures
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# FFmpeg binary detection
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str:
    """
    Locate the ffmpeg binary.  Checks, in order:
      1. System PATH
      2. imageio_ffmpeg bundled binary (installed with MoviePy)
    Returns the full path or raises RuntimeError.
    """
    # Try system PATH first
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    # Fall back to imageio_ffmpeg bundle
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except ImportError:
        pass

    raise RuntimeError(
        "ffmpeg not found.  Either install FFmpeg and add it to PATH, "
        "or install the imageio-ffmpeg package (pip install imageio-ffmpeg)."
    )


FFMPEG_BIN: str = _find_ffmpeg()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("ffmpeg_builder")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _console = logging.StreamHandler(sys.stdout)
    _console.setLevel(logging.INFO)
    _console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s",
                                             datefmt="%H:%M:%S"))
    logger.addHandler(_console)

# ---------------------------------------------------------------------------
# Constants  (shared with slideshow_builder.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SETLIST = PROJECT_ROOT / "final-setlist.txt"
PHOTO_DIR = PROJECT_ROOT / "movie-content" / "pic-playlist"
MUSIC_DIR = PROJECT_ROOT / "movie-content" / "music-playlist"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / ".ffmpeg_temp"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".wma"}

KEN_BURNS_STYLES = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]
KEN_BURNS_INTENSITIES = {
    "subtle":   0.05,
    "medium":   0.12,
    "dramatic": 0.22,
}

# How many ffmpeg zoompan processes to run in parallel (Stage 1).
# Each process is CPU-bound during zoompan, so roughly 1 per core is ideal.
PARALLEL_CLIP_WORKERS = max(1, (os.cpu_count() or 4) // 2)

# Maximum clips to chain in a single xfade filtergraph before batching.
# Keeps the filter script well under Windows 32 kB command-line limit and
# avoids excessive FFmpeg memory use.
XFADE_BATCH_SIZE = 40


# ---------------------------------------------------------------------------
# Data classes  (mirrors slideshow_builder.py)
# ---------------------------------------------------------------------------
@dataclass
class PhotoEntry:
    path: Path
    duration: Optional[float] = None
    effect: Optional[str] = None

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
# Setlist Parser  (reused from slideshow_builder.py)
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

    config.photos = photo_entries if photo_entries else _auto_discover_photos()
    config.music = music_entries if music_entries else _auto_discover_music()
    return config


def _parse_setting(settings: SlideshowSettings, line: str):
    if "=" not in line:
        return
    key, _, value = line.partition("=")
    key = key.strip().lower()
    value = value.strip()
    if "#" in value:
        value = value[:value.index("#")].strip()

    if key == "output_filename":
        settings.output_filename = value
    elif key == "output_resolution":
        m = re.match(r"(\d+)\s*x\s*(\d+)", value, re.IGNORECASE)
        if m:
            settings.output_width = int(m.group(1))
            settings.output_height = int(m.group(2))
    elif key == "default_photo_duration":
        if value.lower() == "auto":
            settings.default_photo_duration = -1.0
        else:
            settings.default_photo_duration = float(value)
    elif key == "photo_duration_variance":
        settings.photo_duration_variance = float(value)
    elif key == "crossfade_duration":
        settings.crossfade_duration = float(value)
    elif key == "ken_burns_intensity":
        if value.lower() in KEN_BURNS_INTENSITIES:
            settings.ken_burns_intensity = value.lower()
    elif key == "ken_burns_style":
        if value.lower() in KEN_BURNS_STYLES + ["random"]:
            settings.ken_burns_style = value.lower()
    elif key == "video_fps":
        settings.video_fps = int(value)
    elif key == "fade_in_duration":
        settings.fade_in_duration = float(value)
    elif key == "fade_out_duration":
        settings.fade_out_duration = float(value)


def _parse_photo_line(line: str) -> Optional[PhotoEntry]:
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
    return entry


def _parse_music_line(line: str) -> Optional[MusicEntry]:
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
# Helper: pick style
# ---------------------------------------------------------------------------
def _pick_style(style_setting: str) -> str:
    if style_setting == "random":
        return random.choice(KEN_BURNS_STYLES)
    return style_setting


# ---------------------------------------------------------------------------
# Helper: run ffmpeg / ffprobe
# ---------------------------------------------------------------------------
def _run_ffmpeg(args: List[str], desc: str = "ffmpeg"):
    """Run an ffmpeg command, raising on failure."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-y"] + args
    logger.debug("  %s: %s", desc, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("  %s FAILED (rc=%d):\n%s", desc, result.returncode, result.stderr[-2000:])
        raise RuntimeError(f"{desc} failed: {result.stderr[-500:]}")
    return result


def _get_duration(path: Path) -> float:
    """
    Get the duration of a media file in seconds.
    Uses ffmpeg (not ffprobe) since imageio_ffmpeg doesn't bundle ffprobe.
    Parses the 'Duration: HH:MM:SS.ss' line from ffmpeg stderr.
    """
    cmd = [FFMPEG_BIN, "-hide_banner", "-i", str(path), "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # ffmpeg prints duration info to stderr even when it "fails" with no output
    for line in result.stderr.splitlines():
        line = line.strip()
        if "Duration:" in line:
            # e.g.  "Duration: 00:03:45.12, start: 0.000000, bitrate: 320 kb/s"
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", line)
            if m:
                h, mins, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
                return h * 3600 + mins * 60 + s
    raise RuntimeError(f"Could not determine duration of {path}")


def _make_even(n: int) -> int:
    """Round up to the nearest even integer (required for yuv420p)."""
    return n + (n % 2)


# ---------------------------------------------------------------------------
# Stage 1: Generate individual photo clips with zoompan Ken Burns
# ---------------------------------------------------------------------------
def _zoompan_filter(style: str, travel: float, duration: float,
                    fps: int, target_w: int, target_h: int) -> str:
    """
    Build an FFmpeg zoompan filter expression for a Ken Burns style.

    The zoompan filter works on a single image input and produces a video
    stream by panning/zooming over the image frame by frame.

    Key zoompan params:
      z  = zoom factor expression (1.0 = no zoom)
      x  = pan x expression (top-left corner of crop in the zoomed image)
      y  = pan y expression
      d  = total number of output frames
      s  = output size
      fps = output fps

    In zoompan, the source image is conceptually scaled by z, then a
    viewport of size s is extracted at position (x, y) from that scaled
    image.  z > 1 means the image appears bigger (zoom in).
    """
    total_frames = int(duration * fps)

    if style == "zoom_in":
        z_expr = f"1+{travel}*(on/{total_frames})"
        x_expr = "(iw-iw/zoom)/2"
        y_expr = "(ih-ih/zoom)/2"

    elif style == "zoom_out":
        z_expr = f"1+{travel}*(1-on/{total_frames})"
        x_expr = "(iw-iw/zoom)/2"
        y_expr = "(ih-ih/zoom)/2"

    elif style == "pan_right":
        z_expr = f"1+{travel}"
        x_expr = f"(iw-iw/zoom)*(on/{total_frames})"
        y_expr = "(ih-ih/zoom)/2"

    elif style == "pan_left":
        z_expr = f"1+{travel}"
        x_expr = f"(iw-iw/zoom)*(1-on/{total_frames})"
        y_expr = "(ih-ih/zoom)/2"

    elif style == "pan_down":
        z_expr = f"1+{travel}"
        x_expr = "(iw-iw/zoom)/2"
        y_expr = f"(ih-ih/zoom)*(on/{total_frames})"

    elif style == "pan_up":
        z_expr = f"1+{travel}"
        x_expr = "(iw-iw/zoom)/2"
        y_expr = f"(ih-ih/zoom)*(1-on/{total_frames})"

    else:
        z_expr = "1"
        x_expr = "0"
        y_expr = "0"

    return (
        f"zoompan=z='{z_expr}'"
        f":x='{x_expr}'"
        f":y='{y_expr}'"
        f":d={total_frames}"
        f":s={target_w}x{target_h}"
        f":fps={fps}"
    )


def _build_single_photo_clip(
    photo_path: Path, duration: float, style: str,
    intensity_name: str, fps: int, target_w: int, target_h: int,
    output_path: Path
):
    """
    Produce a short .mp4 clip from a single photo with Ken Burns
    (zoompan) applied entirely in native FFmpeg code.

    The filter chain replicates the original pipeline's visual output:
      1.  Two parallel branches from the same input image:
          - Background: scale to cover → crop to target → heavy blur → darken
          - Foreground: scale to fit inside target (no crop)
      2.  Overlay foreground centred on background
      3.  Apply zoompan to the composited frame
      4.  Output yuv420p at target resolution
    """
    travel = KEN_BURNS_INTENSITIES.get(intensity_name, 0.12)
    total_frames = int(duration * fps)
    overscan = 1 + travel + 0.05

    # Zoompan input must be larger than output by the travel factor.
    # We prescale the composite to overscan_w × overscan_h, then
    # zoompan crops/zooms within that to produce target_w × target_h.
    overscan_w = _make_even(int(target_w * overscan))
    overscan_h = _make_even(int(target_h * overscan))

    zp_filter = _zoompan_filter(style, travel, duration, fps, target_w, target_h)

    # Build a complex filtergraph with two branches from the same input.
    # [0:v] is the input image (looped).
    #
    # Branch A (background): scale-to-cover → centre crop → blur → darken
    # Branch B (foreground): scale-to-fit (no crop, preserves aspect)
    # Then overlay B centred on A → zoompan → format
    filter_graph = (
        # --- Background branch ---
        f"[0:v]scale={overscan_w}:{overscan_h}"
        f":force_original_aspect_ratio=increase,"
        f"crop={overscan_w}:{overscan_h},"
        f"gblur=sigma=40,"
        f"colorbalance=bs=0:bm=0:bh=0,"
        f"eq=brightness=-0.3"
        f"[bg];"

        # --- Foreground branch ---
        f"[0:v]scale={overscan_w}:{overscan_h}"
        f":force_original_aspect_ratio=decrease"
        f"[fg];"

        # --- Composite: overlay foreground centred on background ---
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[comp];"

        # --- Ken Burns via zoompan ---
        f"[comp]{zp_filter},"
        f"format=yuv420p[out]"
    )

    args = [
        "-loop", "1",
        "-i", str(photo_path),
        "-t", f"{duration:.4f}",
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "17",
        "-an",
        str(output_path),
    ]

    _run_ffmpeg(args, desc=f"zoompan {photo_path.name}")


def _build_clip_task(task: dict):
    """
    Worker function for parallel clip generation.
    Wraps _build_single_photo_clip and returns (index, clip_path, duration)
    or raises on failure.
    """
    _build_single_photo_clip(
        task["photo_path"], task["duration"], task["style"],
        task["intensity"], task["fps"],
        task["target_w"], task["target_h"],
        task["output_path"],
    )
    return task["index"], task["output_path"], task["duration"]


def build_photo_clips_ffmpeg(config: SlideshowConfig) -> List[Tuple[Path, float]]:
    """
    Generate individual Ken Burns video clips for each photo via ffmpeg,
    running up to PARALLEL_CLIP_WORKERS processes concurrently.

    Returns:
        List of (clip_path, duration) tuples in order.
    """
    settings = config.settings
    tw, th = settings.output_width, settings.output_height
    fps = settings.video_fps
    intensity = settings.ken_burns_intensity

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    total = len(config.photos)

    # Build task list
    tasks = []
    for idx, photo in enumerate(config.photos):
        base_dur = photo.duration if photo.duration is not None else settings.default_photo_duration
        if settings.photo_duration_variance > 0 and photo.duration is None:
            base_dur += random.uniform(-settings.photo_duration_variance,
                                        settings.photo_duration_variance)
        base_dur = max(base_dur, 0.5)

        style_setting = photo.effect if photo.effect else settings.ken_burns_style
        style = _pick_style(style_setting)

        clip_path = TEMP_DIR / f"clip_{idx:05d}.mp4"

        tasks.append({
            "index": idx,
            "photo_path": photo.path,
            "duration": base_dur,
            "style": style,
            "intensity": intensity,
            "fps": fps,
            "target_w": tw,
            "target_h": th,
            "output_path": clip_path,
        })
        logger.info("Queued %d/%d: %s — %s, %.1fs",
                     idx + 1, total, photo.path.name, style, base_dur)

    # Execute in parallel
    workers = min(PARALLEL_CLIP_WORKERS, total)
    logger.info("Generating %d clips with %d parallel workers", total, workers)

    results: List[Optional[Tuple[int, Path, float]]] = [None] * total

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_build_clip_task, t): t for t in tasks}
        done_count = 0
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            try:
                idx, clip_path, dur = future.result()
                results[idx] = (clip_path, dur)
                done_count += 1
                logger.info("  Completed %d/%d: %s", done_count, total,
                            task["photo_path"].name)
            except Exception as exc:
                logger.error("  FAILED %s: %s", task["photo_path"].name, exc)

    # Filter out failures and preserve order
    clips = [r for r in results if r is not None]
    if len(clips) < total:
        logger.warning("%d of %d clips failed", total - len(clips), total)
    return clips


# ---------------------------------------------------------------------------
# Stage 2: Assemble clips with xfade crossfade transitions (batched)
# ---------------------------------------------------------------------------
def _xfade_batch(clips: List[Tuple[Path, float]],
                  crossfade_dur: float, fps: int,
                  output_path: Path):
    """
    Chain clips with FFmpeg xfade in a single filtergraph written to a
    temporary script file (avoids Windows 32 kB command-line limit).
    """
    if len(clips) == 0:
        raise ValueError("No clips to assemble")

    if len(clips) == 1:
        shutil.copy2(clips[0][0], output_path)
        return

    n = len(clips)

    # Calculate xfade offsets.  The i-th xfade begins at:
    #   offset_i = sum(dur[0..i]) - crossfade_dur * i
    offsets = []
    cum_dur = 0.0
    for i in range(n - 1):
        cum_dur += clips[i][1]
        offset = cum_dur - crossfade_dur * (i + 1)
        offsets.append(max(offset, 0.0))

    # Build the filtergraph as a string
    filter_lines = []
    prev_label = "[0:v]"
    for i in range(n - 1):
        next_input = f"[{i + 1}:v]"
        out_label = f"[v{i}]"
        filter_lines.append(
            f"{prev_label}{next_input}xfade=transition=fade"
            f":duration={crossfade_dur:.4f}"
            f":offset={offsets[i]:.4f}"
            f"{out_label}"
        )
        prev_label = out_label

    final_label = f"[v{n - 2}]"
    filter_graph = ";\n".join(filter_lines)

    # Write filtergraph to a script file (avoids cmd-line length issues)
    script_path = TEMP_DIR / "xfade_filter.txt"
    script_path.write_text(filter_graph, encoding="utf-8")

    input_args = []
    for clip_path, _ in clips:
        input_args += ["-i", str(clip_path)]

    args = input_args + [
        "-filter_complex_script", str(script_path),
        "-map", final_label,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "17",
        "-an",
        "-fps_mode", "cfr",
        "-r", str(fps),
        str(output_path),
    ]

    _run_ffmpeg(args, desc=f"xfade batch ({n} clips)")


def _assemble_with_xfade(clips: List[Tuple[Path, float]],
                          crossfade_dur: float, fps: int,
                          output_path: Path):
    """
    Assemble all clips with crossfades.  For large photo sets the clips
    are processed in batches of XFADE_BATCH_SIZE, producing intermediate
    files, then the intermediates are concat-demuxed (no re-encode on the
    final join, because each batch's last crossfade_dur of content
    overlaps with the next batch's first clip).

    For simplicity the batch boundaries are chosen so that the last clip
    in each batch is duplicated as the first clip in the next batch,
    giving a clean xfade at the seam.
    """
    n = len(clips)

    if n <= XFADE_BATCH_SIZE:
        # Small enough — single pass
        _xfade_batch(clips, crossfade_dur, fps, output_path)
        return

    # --- Batched assembly ---
    logger.info("Batching %d clips into groups of ~%d for xfade assembly",
                n, XFADE_BATCH_SIZE)
    batch_outputs: List[Tuple[Path, float]] = []
    batch_idx = 0
    start = 0

    while start < n:
        end = min(start + XFADE_BATCH_SIZE, n)
        batch_clips = clips[start:end]
        batch_path = TEMP_DIR / f"batch_{batch_idx:03d}.mp4"

        logger.info("  Batch %d: clips %d–%d (%d clips)",
                     batch_idx, start + 1, end, len(batch_clips))
        _xfade_batch(batch_clips, crossfade_dur, fps, batch_path)

        # Compute the duration of this batch's output
        batch_dur = (sum(d for _, d in batch_clips)
                     - crossfade_dur * max(len(batch_clips) - 1, 0))
        batch_outputs.append((batch_path, max(batch_dur, 0.1)))

        batch_idx += 1
        # Overlap: start next batch from the last clip of the current one
        # so that we can xfade across the batch boundary.
        if end < n:
            start = end - 1
        else:
            break

    # Now join the batches.  If we only have 1 batch, we're done.
    if len(batch_outputs) == 1:
        shutil.copy2(batch_outputs[0][0], output_path)
    else:
        # Use xfade to join the batches (there will be few of them)
        _xfade_batch(batch_outputs, crossfade_dur, fps, output_path)


# ---------------------------------------------------------------------------
# Stage 3+5 combined: Global fades + final mux + audio in one pass
# ---------------------------------------------------------------------------
def _final_encode(video_path: Path, audio_path: Optional[Path],
                   output_path: Path, total_duration: float,
                   fade_in_dur: float, fade_out_dur: float,
                   use_gpu: bool, fps: int) -> str:
    """
    Final encode: apply global video fades, mux audio, and produce the
    output file with production-quality encoding — all in a single
    ffmpeg pass to avoid an extra transcode.
    """
    # --- Video filters: global fades ---
    vfilters = []
    if fade_in_dur > 0:
        vfilters.append(f"fade=t=in:st=0:d={fade_in_dur:.4f}")
    if fade_out_dur > 0:
        fade_out_start = total_duration - fade_out_dur
        vfilters.append(f"fade=t=out:st={fade_out_start:.4f}:d={fade_out_dur:.4f}")

    # --- Encoder selection ---
    if use_gpu:
        v_codec_args = [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "23",
            "-b:v", "0",
        ]
        encoder_name = "h264_nvenc (GPU)"
    else:
        v_codec_args = [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
        ]
        encoder_name = "libx264 (CPU)"

    # --- Build command ---
    args = ["-i", str(video_path)]
    if audio_path and audio_path.exists():
        args += ["-i", str(audio_path)]

    if vfilters:
        args += ["-vf", ",".join(vfilters)]

    args += v_codec_args
    args += ["-pix_fmt", "yuv420p"]  # Ensure broad compatibility with players
    args += ["-c:a", "aac", "-b:a", "256k"]
    args += ["-r", str(fps)]
    args += ["-movflags", "+faststart"]
    args += ["-shortest"]

    if audio_path and audio_path.exists():
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        args += ["-an"]

    args.append(str(output_path))

    _run_ffmpeg(args, desc=f"final encode ({encoder_name})")
    return encoder_name


# ---------------------------------------------------------------------------
# Stage 4: Build audio track
# ---------------------------------------------------------------------------
def _build_audio_track(config: SlideshowConfig, video_duration: float,
                        output_path: Path) -> Optional[Path]:
    """
    Concatenate, loop/trim, and fade music tracks to match video duration.
    Produces a single audio file.
    """
    if not config.music:
        logger.warning("No music tracks available")
        return None

    settings = config.settings

    # Get durations and build concat list
    track_info = []
    for entry in config.music:
        try:
            dur = _get_duration(entry.path)
            track_info.append((entry, dur))
            logger.info("  Music: %s (%.1fs)", entry.path.name, dur)
        except Exception as exc:
            logger.warning("  Could not probe %s: %s — skipping", entry.path.name, exc)

    if not track_info:
        return None

    total_music = sum(d for _, d in track_info)
    logger.info("Total music: %.1fs | Video: %.1fs", total_music, video_duration)

    # Build the playlist (loop if shorter than video)
    playlist: List[MusicEntry] = []
    accumulated = 0.0
    loops = max(1, int(math.ceil(video_duration / total_music))) if total_music > 0 else 1

    for _ in range(loops):
        for entry, dur in track_info:
            if accumulated >= video_duration:
                break
            playlist.append(entry)
            accumulated += dur
        if accumulated >= video_duration:
            break

    # Instead of using the concat demuxer (which fails on mixed formats/sample rates),
    # we'll build a filter_complex graph to safely concatenate the audio tracks.
    # We must explicitly resample and convert to stereo so all inputs match.
    
    inputs = []
    filter_parts = []
    
    for i, entry in enumerate(playlist):
        inputs.extend(["-i", str(entry.path)])
        # aformat ensures consistent sample rate, channel layout, and sample format
        filter_parts.append(f"[{i}:a:0]aformat=sample_rates=44100:channel_layouts=stereo[a{i}];")
    
    # Concatenate the standardized streams
    concat_inputs = "".join(f"[a{i}]" for i in range(len(playlist)))
    concat_filter = "".join(filter_parts) + f"{concat_inputs}concat=n={len(playlist)}:v=0:a=1[outa]"
    
    concat_out = TEMP_DIR / "audio_concat.m4a"
    
    # Run the complex concat command
    args = inputs + [
        "-filter_complex", concat_filter,
        "-map", "[outa]",
        "-c:a", "aac", "-b:a", "256k",
        str(concat_out)
    ]
    
    _run_ffmpeg(args, desc="audio concat")

    # Trim + fade-out
    fade_dur = min(settings.fade_out_duration, video_duration)
    fade_start = max(video_duration - fade_dur, 0)

    _run_ffmpeg([
        "-i", str(concat_out),
        "-t", f"{video_duration:.4f}",
        "-af", f"afade=t=out:st={fade_start:.4f}:d={fade_dur:.4f}",
        "-c:a", "aac", "-b:a", "256k",
        str(output_path),
    ], desc="audio trim+fade")

    return output_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def build_slideshow_ffmpeg(config: SlideshowConfig, use_gpu: bool = False):
    """Full FFmpeg-native pipeline: photos → zoompan → xfade → audio → export."""
    settings = config.settings
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / settings.output_filename

    # ---- Stage 1: Generate individual photo clips (parallel) ----
    logger.info("=" * 60)
    logger.info("STAGE 1: GENERATING KEN BURNS CLIPS (FFmpeg zoompan)")
    logger.info("         %d parallel workers", PARALLEL_CLIP_WORKERS)
    logger.info("=" * 60)
    t0 = time.time()
    clips = build_photo_clips_ffmpeg(config)
    if not clips:
        logger.error("No photo clips generated. Exiting.")
        sys.exit(1)
    logger.info("Photo clips generated in %.1fs", time.time() - t0)

    # ---- Stage 2: Crossfade assembly (batched xfade) ----
    logger.info("=" * 60)
    logger.info("STAGE 2: ASSEMBLING WITH XFADE CROSSFADES")
    logger.info("=" * 60)
    t1 = time.time()
    assembled_path = TEMP_DIR / "assembled.mp4"
    _assemble_with_xfade(clips, settings.crossfade_duration,
                          settings.video_fps, assembled_path)
    logger.info("Assembly complete in %.1fs", time.time() - t1)

    # Calculate total video duration
    total_dur = (sum(d for _, d in clips)
                 - settings.crossfade_duration * max(len(clips) - 1, 0))
    total_dur = max(total_dur, 0.1)
    logger.info("Video duration: %.1fs (%.1f minutes)", total_dur, total_dur / 60)

    # ---- Stage 3: Audio ----
    logger.info("=" * 60)
    logger.info("STAGE 3: BUILDING AUDIO TRACK")
    logger.info("=" * 60)
    audio_path = TEMP_DIR / "audio_final.m4a"
    audio_result = _build_audio_track(config, total_dur, audio_path)

    # ---- Stage 4: Final encode (global fades + audio + production codec) ----
    logger.info("=" * 60)
    logger.info("STAGE 4: FINAL ENCODE (fades + audio + mux)")
    logger.info("=" * 60)
    t2 = time.time()
    encoder_name = _final_encode(
        assembled_path, audio_result, output_path, total_dur,
        settings.fade_in_duration, settings.fade_out_duration,
        use_gpu, settings.video_fps,
    )
    logger.info("Final encode in %.1fs", time.time() - t2)

    # ---- Summary ----
    total_elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info("Output file : %s", output_path)
    logger.info("Duration    : %.1fs (%.1f minutes)", total_dur, total_dur / 60)
    logger.info("Photos used : %d", len(clips))
    logger.info("Music tracks: %d", len(config.music))
    logger.info("Render time : %.1fs (%.1f minutes)", total_elapsed, total_elapsed / 60)
    logger.info("Encoder     : %s", encoder_name)

    # Clean up temp files
    logger.info("Cleaning up temp directory: %s", TEMP_DIR)
    try:
        shutil.rmtree(TEMP_DIR)
    except Exception as exc:
        logger.warning("Could not clean temp dir: %s", exc)


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------
def dry_run(config: SlideshowConfig):
    """Print a full summary without rendering."""
    settings = config.settings

    print("\n" + "=" * 60)
    print("  DRY RUN — FFmpeg Pipeline Preview")
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
    print(f"  Pipeline         : FFmpeg-native (zoompan + xfade)")
    print(f"  Parallel workers : {PARALLEL_CLIP_WORKERS}")
    print(f"  xfade batch size : {XFADE_BATCH_SIZE}")

    print(f"\n  PHOTOS ({len(config.photos)} total):")
    total_photo_dur = 0.0
    for i, p in enumerate(config.photos, 1):
        dur = p.duration if p.duration is not None else settings.default_photo_duration
        eff = p.effect if p.effect else settings.ken_burns_style
        print(f"    {i:4d}. {p.path.name:<40s}  {dur:5.1f}s  {eff}")
        total_photo_dur += dur

    crossfade_overlap = settings.crossfade_duration * max(len(config.photos) - 1, 0)
    estimated_video_dur = total_photo_dur - crossfade_overlap
    estimated_video_dur = max(estimated_video_dur, 0)

    print(f"\n  Estimated video duration: {estimated_video_dur:.1f}s "
          f"({estimated_video_dur / 60:.1f} minutes)")

    print(f"\n  MUSIC ({len(config.music)} tracks):")
    total_music_dur = 0.0
    for i, m in enumerate(config.music, 1):
        extras = ""
        if m.fade_in:
            extras += f"  fade_in={m.fade_in}s"
        if m.fade_out:
            extras += f"  fade_out={m.fade_out}s"
        try:
            dur = _get_duration(m.path)
            total_music_dur += dur
            extras += f"  ({dur:.1f}s)"
        except Exception:
            extras += "  (duration unknown)"
        print(f"    {i:4d}. {m.path.name}{extras}")

    if total_music_dur > 0:
        print(f"\n  Total music duration: {total_music_dur:.1f}s "
              f"({total_music_dur / 60:.1f} minutes)")
        if total_music_dur >= estimated_video_dur:
            print("  → Music will be trimmed to fit video length")
            final_duration = estimated_video_dur
        else:
            print("  → Music will loop to fill video length")
            final_duration = estimated_video_dur
    else:
        final_duration = estimated_video_dur
        
    print(f"\n  ============================================================")
    print(f"  FINAL ESTIMATES:")
    print(f"  Total Photos      : {len(config.photos)}")
    print(f"  Total Audio Tracks: {len(config.music)}")
    print(f"  Final Video Length: {final_duration:.1f}s ({final_duration / 60:.1f} minutes)")
    print(f"  ============================================================")

    print("\n" + "=" * 60)
    print("  Dry run complete — no video rendered.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FFmpeg-native slideshow builder with Ken Burns zoompan effects."
    )
    parser.add_argument(
        "--setlist", type=str, default=str(DEFAULT_SETLIST),
        help="Path to the setlist configuration file (default: final-setlist.txt)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview the setlist and timing without rendering video"
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use NVIDIA NVENC GPU acceleration for final H.264 encoding"
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Only use the first N photos (for quick test builds)"
    )
    args = parser.parse_args()

    setlist_path = Path(args.setlist)
    if not setlist_path.is_absolute():
        setlist_path = PROJECT_ROOT / setlist_path

    logger.info("Parsing setlist: %s", setlist_path)
    config = parse_setlist(setlist_path)

    if args.limit > 0 and config.photos:
        original_count = len(config.photos)
        config.photos = config.photos[:args.limit]
        logger.info("--limit %d: using %d of %d photos",
                    args.limit, len(config.photos), original_count)

    if not config.photos:
        logger.error("No photos found. Cannot build slideshow. Exiting.")
        sys.exit(1)
    if not config.music:
        logger.warning("No music found. Will generate a silent video.")

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
        build_slideshow_ffmpeg(config, use_gpu=args.gpu)


if __name__ == "__main__":
    main()
