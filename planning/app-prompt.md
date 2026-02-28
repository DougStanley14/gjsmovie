# Coding Agent Prompt: Photo Slideshow Video Generator with Ken Burns Effect

## Project Overview
Build a Python application that generates a cinematic photo slideshow video with Ken Burns effects, crossfade transitions, and a music soundtrack. The app must be flexible enough to handle changing photo and music file sets through an editable setlist configuration file.

---

## Directory Structure
The project will use the following structure:
```
project-root/
├── movie-content/
│   ├── pic-playlist/        # Source photos (JPG, PNG, etc.)
│   └── music-playlist/      # Source music files (MP3, WAV, etc.)
├── final-setlist.txt        # Editable config file (see format below)
├── output/                  # Generated video saved here
├── slideshow_builder.py     # Main application script
└── requirements.txt
```

---

## The `final-setlist.txt` File
This is the central control file for the app. It must be human-editable and clearly commented. Design it with the following sections:

```
# ============================================================
# SLIDESHOW FINAL SETLIST - Edit this file to customize output
# ============================================================
# Lines starting with # are comments and are ignored
# Leave a section blank to use defaults

[SETTINGS]
output_filename = brothers_slideshow.mp4
output_resolution = 1920x1080
default_photo_duration = 5          # seconds per photo (can be overridden per photo)
photo_duration_variance = 0.5       # randomly vary duration by +/- this many seconds
crossfade_duration = 1.0            # seconds of crossfade between photos
ken_burns_intensity = medium        # options: subtle, medium, dramatic
ken_burns_style = random            # options: zoom_in, zoom_out, pan_left, pan_right, random
video_fps = 24
fade_in_duration = 2                # fade in from black at start (seconds)
fade_out_duration = 3               # fade to black at end (seconds)

[PHOTOS]
# List photos in desired order, one per line
# Format: filename.jpg
# OR: filename.jpg | duration_override=8 | effect=zoom_in
# Leave this section empty to auto-load ALL photos from movie-content/pic-playlist/ alphabetically
# Example overrides:
# IMG_001.jpg | duration=8 | effect=zoom_in
# IMG_002.jpg
# IMG_050.jpg | duration=10 | effect=pan_right

[MUSIC]
# List music files in playback order, one per line
# Format: filename.mp3
# OR: filename.mp3 | fade_in=2 | fade_out=3
# The app will loop/sequence all listed tracks to fill the video duration
# Leave this section empty to auto-load ALL music from movie-content/music-playlist/ alphabetically
# Example:
# song1.mp3 | fade_in=2 | fade_out=3
# song2.mp3
```

---

## Core Features to Build

### 1. Setlist Parser
- Parse `final-setlist.txt` into a configuration object
- Handle missing sections gracefully with smart defaults
- If `[PHOTOS]` section is empty, auto-discover all image files from `movie-content/pic-playlist/` sorted alphabetically
- If `[MUSIC]` section is empty, auto-discover all music files from `movie-content/music-playlist/` sorted alphabetically
- Support per-photo duration overrides and per-photo Ken Burns effect overrides
- Support per-track music fade in/out overrides
- Print a clear summary of what was loaded before building begins

### 2. Ken Burns Effect Engine
Build a reusable `apply_ken_burns(clip, style, intensity)` function that:
- Accepts a still image clip and applies a smooth, animated zoom or pan over its duration
- Supports these styles: `zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `pan_up`, `pan_down`, `random`
- Supports three intensity levels:
  - `subtle`: ~5% zoom or pan travel
  - `medium`: ~10–15% zoom or pan travel
  - `dramatic`: ~20–25% zoom or pan travel
- Uses NumPy/resize to generate per-frame transformations so motion is perfectly smooth
- Always maintains the target output resolution (no black bars, no stretching) by slightly oversizing the source image before applying the effect
- When `random` is selected, picks a different random style for each photo so the video feels dynamic

### 3. Photo Clip Builder
- Load each image and resize/crop it to fill the output resolution (cover mode, no letterboxing)
- Apply the Ken Burns effect to each clip
- Apply the per-photo duration (with optional variance using `random.uniform`)
- Apply crossfade transitions (`moviepy.video.fx.fadein` / `fadeout` or `CompositeVideoClip` crossfade)
- Log progress as each clip is processed (e.g., `Processing photo 47/500: IMG_047.jpg — zoom_in, 5.3s`)

### 4. Music Engine
- Load all music tracks listed in `[MUSIC]` in order
- Concatenate them into a single audio timeline
- If the total music duration is longer than the video: trim the audio at the end with a smooth fade out
- If the total music duration is shorter than the video: loop the playlist from the beginning seamlessly (with crossfade between loop points)
- Apply individual track fade-in and fade-out as specified
- Apply a global fade-out over the last `fade_out_duration` seconds of the final video

### 5. Final Assembly
- Concatenate all photo clips with crossfades into a single `CompositeVideoClip` or `concatenate_videoclips`
- Apply a global fade-in from black at the start
- Apply a global fade-to-black at the end
- Mix the music audio onto the video
- Export using `write_videofile()` with H.264 video codec and AAC audio codec
- Save to `output/[output_filename]`
- Print final stats on completion: total duration, number of photos, number of music tracks used

### 6. Dry Run Mode
- Add a `--dry-run` command line flag
- In dry run mode, parse the setlist, calculate total estimated video duration, list all photos and music that will be used, and print a full summary — but do NOT render the video
- This lets the user verify their setlist and tweak durations before committing to a full render

### 7. Resume/Checkpoint Support (nice to have)
- Cache processed photo clips as temporary files so if the render crashes at photo 400, it doesn't have to redo photos 1–399
- Add a `--no-cache` flag to force a full rebuild

---

## Technical Requirements
- Use **MoviePy** (v1.0.3 or v2.x — handle both if possible) as the primary video library
- Use **NumPy** for Ken Burns frame calculations
- Use **Pillow (PIL)** for image loading and preprocessing
- Use **argparse** for CLI flags (`--dry-run`, `--no-cache`, `--setlist path/to/file`)
- Use **logging** module for clean, leveled output (INFO for progress, DEBUG for verbose frame details)
- All file paths should be relative to the project root, cross-platform safe (use `pathlib.Path`)
- `requirements.txt` should pin all dependencies

---

## Error Handling
- If a photo file listed in `final-setlist.txt` doesn't exist in the directory, log a warning and skip it (don't crash)
- If a music file doesn't exist, log a warning and skip it
- If no photos are found at all, exit with a clear error message
- If no music is found, offer to generate a silent video or exit with a message
- Validate output resolution format in settings

---

## Deliverables
1. `slideshow_builder.py` — fully commented, production-quality Python script
2. `final-setlist.txt` — pre-populated template with all options, fully commented
3. `requirements.txt` — all dependencies with pinned versions
4. A short `README.md` explaining how to install dependencies, populate the setlist, and run the script including all CLI flags and example commands

---

## Example CLI Usage
```bash
# Standard run
python slideshow_builder.py

# Dry run to preview timing before rendering
python slideshow_builder.py --dry-run

# Use a custom setlist file
python slideshow_builder.py --setlist custom_setlist.txt

# Force full rebuild ignoring cache
python slideshow_builder.py --no-cache

# Dry run with custom setlist
python slideshow_builder.py --dry-run --setlist custom_setlist.txt
```

---

*The goal is a polished, cinematic slideshow that feels like a professional tribute video. Prioritize smooth motion, clean transitions, and resilient code that handles a real-world messy photo library without crashing.*

