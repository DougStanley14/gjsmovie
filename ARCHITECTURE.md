# Slideshow Builder Architecture

This document outlines the technical architecture and design decisions behind `slideshow_builder.py`. Use this to understand the codebase when tweaking parameters or extending functionality.

## Overview

The application follows a **pipeline architecture** with clear separation of concerns:

```
Setlist Parser → Photo Engine → Crossfade Engine → Music Engine → Export
```

Each stage produces data structures that feed into the next, with extensive caching and error handling throughout.

## Core Data Structures

### SlideshowConfig
The central configuration object that holds everything parsed from the setlist file:

```python
@dataclass
class SlideshowConfig:
    settings: SlideshowSettings      # Global output parameters
    photos: List[PhotoEntry]         # Ordered photo list with overrides
    music: List[MusicEntry]          # Ordered music list with fades
```

### PhotoEntry & MusicEntry
Lightweight containers that include per-item overrides:

```python
@dataclass
class PhotoEntry:
    path: Path
    duration: Optional[float] = None      # Override default_photo_duration
    effect: Optional[str] = None          # Override ken_burns_style

@dataclass
class MusicEntry:
    path: Path
    fade_in: Optional[float] = None
    fade_out: Optional[float] = None
```

## Pipeline Stages

### 1. Setlist Parser (`parse_setlist()`)

**Purpose**: Convert human-readable `final-setlist.txt` into `SlideshowConfig`.

**Key Design**:
- Section-based parsing (`[SETTINGS]`, `[PHOTOS]`, `[MUSIC]`)
- Graceful fallback: if `[PHOTOS]`/`[MUSIC]` sections are empty, auto-discover files
- Per-item override parsing with `|` delimiter
- Inline comment stripping (`#` comments anywhere on a line)

**Auto-discovery Logic**:
- Photos: All files with extensions in `IMAGE_EXTENSIONS` from `PHOTO_DIR`, sorted alphabetically
- Music: All files with extensions in `AUDIO_EXTENSIONS` from `MUSIC_DIR`, sorted alphabetically

**Error Handling**:
- Missing setlist file → use all defaults (auto-discover everything)
- Missing individual files → log warning and skip (don't crash)

### 2. Ken Burns Effect Engine (`apply_ken_burns()`)

**Purpose**: Generate smooth per-frame pan/zoom transforms for a single image.

**Core Algorithm**:
1. **Oversizing**: Source image is oversized by `travel + 2%` to prevent black borders during motion
2. **Per-frame Calculation**: For each frame `t ∈ [0,1]`, calculate crop window based on style
3. **High-quality Resize**: Use Pillow `LANCZOS` for final resize to target resolution

**Motion Styles**:
- `zoom_in`: Crop size `scale = 1.0 - travel * t` (start wide, end tight)
- `zoom_out`: Crop size `scale = 1.0 - travel * (1-t)` (start tight, end wide)
- `pan_left/right`: Fixed crop width, slide `x0 = max_offset * t` or `max_offset * (1-t)`
- `pan_up/down`: Fixed crop height, slide `y0` similarly
- `random`: Picks a different style for each photo

**Intensity Levels**:
- `subtle`: 5% travel (`0.05`)
- `medium`: 12% travel (`0.12`)
- `dramatic`: 22% travel (`0.22`)

**Implementation Details**:
- Uses NumPy arrays for per-frame manipulation
- Returns `List[np.ndarray]` (one array per frame)
- All math is integer-safe for pixel coordinates

### 3. Photo Clip Builder (`build_photo_clips()`)

**Purpose**: Convert images into MoviePy `ImageClip` objects with Ken Burns effects applied.

**Workflow**:
1. **Resolve Duration**: Use per-photo override or global default + random variance
2. **Cache Check**: Look for existing cached frames based on deterministic key
3. **Image Loading**: `load_and_prepare_image()` handles EXIF orientation and cover-mode resize
4. **Ken Burns**: Call `apply_ken_burns()` to generate frame sequence
5. **Cache Write**: Save frames to pickle cache for future runs
6. **Clip Construction**: Build `ImageClip` with custom frame function

**Cover-mode Resize Logic**:
- Calculate scale to fully cover target dimensions (`max(target_w/src_w, target_h/src_h)`)
- Resize, then center-crop to exact target size
- Ensures no black bars regardless of aspect ratio

**Caching Strategy**:
- **Cache Key**: MD5 hash of `path|duration|style|intensity|resolution|fps`
- **Storage**: `.slideshow_cache/{hash}.pkl` using pickle protocol `HIGHEST_PROTOCOL`
- **Benefits**: Resume after crash, faster re-runs with unchanged parameters

### 4. Crossfade Assembly (`assemble_with_crossfades()`)

**Purpose**: Layer photo clips with smooth crossfade transitions.

**Key Insight**: MoviePy's `CompositeVideoClip` allows overlapping clips with transparency. We exploit this for crossfades.

**Algorithm**:
1. **Calculate Start Times**: Each clip starts `crossfade_duration` before previous clip ends
   ```
   start_times[0] = 0
   start_times[i] = start_times[i-1] + duration[i-1] - crossfade_duration
   ```
2. **Apply Effects**: Build effects list in one pass to avoid `with_effects` overwrite bug
   - Clip 1: Only `CrossFadeOut`
   - Middle clips: Both `CrossFadeIn` and `CrossFadeOut`
   - Last clip: Only `CrossFadeIn`
3. **Composite**: Layer all clips with calculated start times

**Why Not `concatenate_videoclips()`?**:
- `concatenate_videoclips()` doesn't support true crossfade overlap
- `CompositeVideoClip` gives precise control over timing and transparency

### 5. Music Engine (`build_audio_track()`)

**Purpose**: Create a single audio track that fits the video duration exactly.

**Workflow**:
1. **Load Clips**: Load each `MusicEntry` with per-track fades
2. **Concatenate**: Create base timeline with `concatenate_audioclips()`
3. **Duration Comparison**:
   - If music ≥ video: Trim to video length, apply global fade-out
   - If music < video: Loop playlist to fill video, then fade out

**Looping Logic**:
- Calculate `loops_needed = ceil(video_duration / total_music_duration)`
- Iterate through playlist, adding full or partial clips until video duration filled
- Apply global fade-out to final trimmed clip

**Audio Effects**:
- Per-track: `afx.AudioFadeIn`/`AudioFadeOut` (from setlist overrides)
- Global: `afx.AudioFadeOut` on final clip (always applied)

### 6. Final Assembly (`build_slideshow()`)

**Purpose**: Combine video and audio, apply global fades, export.

**Steps**:
1. **Build Photo Clips**: Call `build_photo_clips()`
2. **Crossfade Assembly**: Call `assemble_with_crossfades()`
3. **Global Fades**: Apply `CrossFadeIn` and `CrossFadeOut` in single `with_effects()` call
4. **Audio Track**: Call `build_audio_track()` and attach if available
5. **Export**: `write_videofile()` with H.264 video + AAC audio

**Export Parameters**:
- `codec="libx264"`: Standard H.264 video
- `audio_codec="aac"`: Universal audio codec
- `preset="medium"`: Balance of speed and quality
- `threads=os.cpu_count()`: Parallel encoding

## Configuration System

### Settings Resolution Order
1. **Setlist `[SETTINGS]` section** (highest priority)
2. **Hardcoded defaults** in `SlideshowSettings.__init__()`

### Parameter Validation
- Resolution format: `(\d+)\s*x\s*(\d+)` regex, fallback to 1920×1080
- Ken Burns intensity: Must be in `KEN_BURNS_INTENSITIES` keys
- Ken Burns style: Must be in `KEN_BURNS_STYLES + ["random"]`

### Duration Calculations
- **Photo Duration**: `base_duration + uniform(-variance, +variance)`
- **Crossfade Reduction**: Total video duration = sum(photo_durations) - crossfade_duration × (N-1)
- **Music Fitting**: Trim or loop to match video duration exactly

## Error Handling Strategy

### Philosophy
- **Never crash** on missing files or bad data
- **Log warnings** for recoverable issues
- **Provide clear error messages** for unrecoverable conditions

### Specific Cases
- **Missing Photo**: Log warning, skip (continue with other photos)
- **Missing Music**: Log warning, skip (may result in silent video)
- **No Photos Found**: Exit with clear error message
- **No Music Found**: Continue with silent video (log warning)
- **Invalid Settings**: Use defaults, log warning
- **Cache Corruption**: Ignore cache, regenerate

## Performance Optimizations

### Caching
- **Photo Processing**: Expensive Ken Burns calculations cached per unique parameter set
- **Cache Invalidation**: Automatic when any parameter changes (duration, effect, intensity, resolution)
- **Cache Size**: Each cached clip ≈ `duration × fps × width × height × 3` bytes

### Memory Management
- **Frame Streaming**: Generate frames on-demand via custom frame function
- **Pickle Protocol**: Use `HIGHEST_PROTOCOL` for efficient serialization
- **Garbage Collection**: Close audio clips after duration calculation in dry-run

### Parallel Processing
- **Export**: MoviePy uses multiple threads via `threads=os.cpu_count()`
- **Photo Processing**: Sequential (to avoid memory explosion with large photo sets)

## Extending the System

### Adding New Ken Burns Styles
1. Add style name to `KEN_BURNS_STYLES`
2. Implement motion logic in `apply_ken_burns()` function
3. Update documentation in README.md

### Adding New Settings
1. Add field to `SlideshowSettings`
2. Add parsing logic in `_parse_setting()`
3. Use setting in appropriate pipeline stage
4. Update `final-setlist.txt` template and README

### Supporting New File Formats
1. Add extension to `IMAGE_EXTENSIONS` or `AUDIO_EXTENSIONS`
2. Test with MoviePy's loading capabilities
3. Update README documentation

## Debugging Tips

### Dry Run Mode
Use `--dry-run` to verify:
- Setlist parsing results
- Photo/music file discovery
- Duration calculations
- Parameter overrides

### Cache Inspection
- Cache files stored in `.slideshow_cache/`
- Each file: `{hash}.pkl` containing `List[np.ndarray]`
- Delete cache directory to force full rebuild

### Logging Levels
- **INFO**: Progress updates, file counts, duration summaries
- **DEBUG**: Detailed frame calculations, cache operations
- **WARNING**: Missing files, invalid settings, recoverable errors

### Common Issues
- **FFmpeg not found**: Install FFmpeg and add to PATH
- **Memory errors**: Reduce photo count, lower resolution, or use `--no-cache`
- **Audio sync issues**: Check audio file formats, ensure consistent sample rates
- **Black borders**: Verify cover-mode resize logic, check source image aspect ratios

## Dependencies and Versions

### Core Libraries
- **MoviePy 2.1.1**: Video/audio processing
- **NumPy**: Per-frame calculations and array operations
- **Pillow**: Image loading, EXIF handling, high-quality resizing

### External Dependencies
- **FFmpeg**: Video encoding (must be on system PATH)
- **Python 3.10+**: Required for MoviePy 2.x compatibility

### Version Compatibility
- MoviePy 1.x: Different API (`with_effects` vs older methods)
- MoviePy 2.x: Current target, uses `with_effects`, `with_updated_frame_function`
- Future versions: Monitor API changes in clip construction and effect application
