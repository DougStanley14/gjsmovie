# gjsmovie

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