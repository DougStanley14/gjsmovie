# gjsmovie

A cinematic photo slideshow generator with Ken Burns effects, crossfade transitions, and a music soundtrack — built for tribute and memorial videos.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place photos in movie-content/pic-playlist/
# 3. Place music in movie-content/music-playlist/
# 4. Edit final-setlist.txt to customize (or leave defaults)

# 5. Preview what will be built (no rendering)
python slideshow_builder.py --dry-run

# 6. Build the video
python slideshow_builder.py
```

The output video is saved to `output/brothers_slideshow.mp4` (or whatever filename is set in the setlist).

---

## Slideshow Builder

`slideshow_builder.py` generates a polished slideshow video from a folder of photos and music tracks. It applies Ken Burns (pan/zoom) effects to each photo, crossfade transitions between photos, and layers a music soundtrack underneath.

### Prerequisites

- **Python 3.10+**
- **FFmpeg** must be installed and accessible on PATH (MoviePy uses it for encoding)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### CLI Usage

```bash
# Standard run — uses final-setlist.txt, renders full video
python slideshow_builder.py

# Dry run — parse setlist, show timing summary, skip rendering
python slideshow_builder.py --dry-run

# Use a custom setlist file
python slideshow_builder.py --setlist custom_setlist.txt

# Force full rebuild, ignoring cached photo clips
python slideshow_builder.py --no-cache

# Use GPU (NVIDIA NVENC) for faster encoding
python slideshow_builder.py --gpu

# Quick test build with only the first 10 photos
python slideshow_builder.py --limit 10

# Combine flags — fast GPU test build
python slideshow_builder.py --gpu --limit 10

# Combine flags
python slideshow_builder.py --dry-run --setlist custom_setlist.txt
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview photo/music lineup and estimated duration without rendering |
| `--setlist PATH` | Use a custom setlist file instead of `final-setlist.txt` |
| `--no-cache` | Ignore cached processed photo clips and rebuild from scratch |
| `--gpu` | Use NVIDIA NVENC GPU acceleration for H.264 encoding (much faster) |
| `--limit N` | Only use the first N photos — ideal for quick test builds |

### The Setlist File (`final-setlist.txt`)

This is the central configuration file. It has three sections:

**`[SETTINGS]`** — Global output parameters:
- `output_filename` — name of the output video file
- `output_resolution` — e.g. `1920x1080`
- `default_photo_duration` — seconds each photo is displayed
- `photo_duration_variance` — randomly vary duration by ± this many seconds
- `crossfade_duration` — seconds of crossfade overlap between photos
- `ken_burns_intensity` — `subtle` (~5%), `medium` (~12%), or `dramatic` (~22%)
- `ken_burns_style` — `zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `pan_up`, `pan_down`, or `random`
- `video_fps` — frames per second (default 24)
- `fade_in_duration` — fade-in from black at start (seconds)
- `fade_out_duration` — fade-to-black at end (seconds)

**`[PHOTOS]`** — Ordered list of photo filenames from `movie-content/pic-playlist/`:
```
# Simple entry
00000023.jpg

# With per-photo overrides
00000027.jpg | duration=8 | effect=zoom_in
```
Leave empty (all lines commented) to auto-load all photos alphabetically.

**`[MUSIC]`** — Ordered list of music filenames from `movie-content/music-playlist/`:
```
# Simple entry
07 Here Comes The Sun.m4a

# With per-track fade overrides
01 Give A Little Bit.m4a | fade_in=2 | fade_out=3
```
Leave empty to auto-load all music alphabetically. If music is longer than the video, it's trimmed with a fade-out. If shorter, it loops.

### Ken Burns Effects

Each photo gets a smooth animated pan or zoom. The `random` style picks a different effect for each photo so the video feels dynamic.

| Style | Motion |
|-------|--------|
| `zoom_in` | Slowly zooms into the center |
| `zoom_out` | Starts zoomed in, pulls back |
| `pan_left` | Slowly drifts left |
| `pan_right` | Slowly drifts right |
| `pan_up` | Slowly drifts upward |
| `pan_down` | Slowly drifts downward |

### Caching

Processed photo clips are cached in `.slideshow_cache/` so if a render is interrupted, restarting skips already-processed photos. Use `--no-cache` to force a full rebuild.

### Error Handling

- Missing photo/music files are logged as warnings and skipped (no crash)
- If zero photos are found, the script exits with a clear error
- If zero music tracks are found, it generates a silent video
- Invalid resolution formats fall back to 1920×1080

### Project Structure

```
project-root/
├── movie-content/
│   ├── pic-playlist/        # Source photos (JPG, PNG, HEIC, etc.)
│   └── music-playlist/      # Source music files (M4A, MP3, WAV, etc.)
├── final-setlist.txt        # Editable config — controls everything
├── output/                  # Generated video saved here
├── .slideshow_cache/        # Cached processed clips (gitignored)
├── slideshow_builder.py     # Main application
└── requirements.txt         # Python dependencies
```

---

## Music List Generator

The `generate_music_list.py` script scans a folder for music files and creates a `music-list.txt` file with relative paths to all found music files.

### Usage

```bash
# Scan folder and create music-list.txt in that folder
python generate_music_list.py "C:\My\Music"

# Specify custom output filename
python generate_music_list.py "C:\My\Music" -o my-music-list.txt

# Scan current folder
python generate_music_list.py .
```

### Supported File Types

- Audio: `.mp3`, `.flac`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.wma`
- Video: `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.mpg`, `.mpeg`

## Music Table Generator

The `generate_music_table.py` script creates a formatted music table from a music list file using metadata extracted from audio files.

### Prerequisites

Install the required dependency:
```bash
pip install mutagen
```

### Usage

```bash
# Default usage (uses movie-content folder and music-list-table.txt output)
python generate_music_table.py

# Specify custom folder
python generate_music_table.py "C:\My\Music"

# Specify custom output filename
python generate_music_table.py -o my-music-table.txt

# Combine custom folder and output filename
python generate_music_table.py "C:\My\Music" -o custom.txt
```

### Requirements

- The specified folder must contain a `music-list.txt` file with paths to audio files
- Audio files should have embedded metadata for best results
- Output will be saved as a pipe-delimited markdown table in the specified folder