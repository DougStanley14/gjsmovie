#!/usr/bin/env python3
"""
Generate music-list-table.txt from music-list.txt using mutagen for metadata.
Outputs a pipe-delimited markdown-style table.
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    from mutagen import File as MutagenFile
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False
    print("ERROR: mutagen not installed. Run: pip install mutagen")
    sys.exit(1)


def format_duration(seconds):
    if seconds is None:
        return ""
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def clean_title_from_filename(stem):
    """Strip leading track-number patterns from a filename stem."""
    cleaned = re.sub(r'^[\d]+[-.][\d]+\s*[-]?\s*', '', stem)  # 1-01 or 1-01 -
    cleaned = re.sub(r'^[\d]+\s*[-]?\s*', '', cleaned)         # 01 or 01 -
    return cleaned.strip() or stem.strip()


def parse_path(rel_path):
    """Parse artist, album, title from path like music\\Artist\\Album\\TrackNum Title.ext"""
    parts = rel_path.replace('\\', '/').split('/')
    # parts[0] == 'music'
    artist = album = track_file = None

    if len(parts) >= 4:
        artist = parts[1]
        album = parts[2]
        track_file = parts[3]
    elif len(parts) == 3:
        album = parts[1]
        track_file = parts[2]
    elif len(parts) == 2:
        track_file = parts[1]
    else:
        track_file = parts[-1] if parts else rel_path

    stem = Path(track_file).stem if track_file else ""
    title = clean_title_from_filename(stem)
    return artist, album, title


def get_tag_str(tags, *keys):
    """Try multiple tag keys, return first found string value."""
    for key in keys:
        val = tags.get(key)
        if val:
            v = val[0] if isinstance(val, (list, tuple)) else val
            s = str(v).strip()
            if s:
                return s
    return ""


def get_mutagen_info(filepath):
    try:
        f = MutagenFile(filepath, easy=True)
        if f is None:
            return {}
        info = {}
        if hasattr(f, 'info') and hasattr(f.info, 'length'):
            info['duration_seconds'] = f.info.length

        tags = f.tags or {}

        title = get_tag_str(tags, 'title')
        if title:
            info['title'] = title

        artist = get_tag_str(tags, 'artist', 'albumartist')
        if artist:
            info['artist'] = artist

        album = get_tag_str(tags, 'album')
        if album:
            info['album'] = album

        date = get_tag_str(tags, 'date', 'year', 'originaldate', 'originalyear')
        if date:
            m = re.match(r'(\d{4})', date)
            if m:
                info['year'] = m.group(1)

        genre = get_tag_str(tags, 'genre')
        if genre:
            info['genre'] = genre

        bpm = get_tag_str(tags, 'bpm', 'tbpm')
        if bpm:
            info['bpm'] = bpm

        mood = get_tag_str(tags, 'mood')
        if mood:
            info['mood'] = mood

        return info
    except Exception:
        return {}


def sanitize(val):
    """Remove pipe chars and strip whitespace from a cell value."""
    return str(val).replace('|', '/').replace('\n', ' ').replace('\r', '').strip()


def main():
    parser = argparse.ArgumentParser(description='Generate music-list-table.txt from music-list.txt using mutagen for metadata.')
    parser.add_argument('folder', nargs='?', 
                       default='d:\\greg-memorial\\gjsmovie\\movie-content',
                       help='Folder containing music-list.txt and where music-list-table.txt will be written')
    parser.add_argument('-o', '--output', 
                       default='music-list-table.txt',
                       help='Output filename (default: music-list-table.txt)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.folder)
    list_file = base_dir / 'music-list.txt'
    output_file = base_dir / args.output

    if not list_file.exists():
        print(f"ERROR: music-list.txt not found in {base_dir}")
        sys.exit(1)

    with open(list_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    columns = ['title', 'artist', 'album', 'year', 'genre', 'duration',
               'bpm', 'mood', 'popularity', 'lyrics_snippet', 'tags', 'file']

    col_widths = [len(c) for c in columns]
    rows = []
    total = len(lines)

    for i, rel_path in enumerate(lines):
        if i % 200 == 0:
            print(f"  Processing {i}/{total}...", flush=True)

        full_path = base_dir / rel_path
        meta = get_mutagen_info(str(full_path))

        path_artist, path_album, path_title = parse_path(rel_path)

        title     = sanitize(meta.get('title',  path_title  or ''))
        artist    = sanitize(meta.get('artist', path_artist or ''))
        album     = sanitize(meta.get('album',  path_album  or ''))
        year      = sanitize(meta.get('year',   ''))
        genre     = sanitize(meta.get('genre',  ''))
        duration  = sanitize(format_duration(meta.get('duration_seconds')))
        bpm       = sanitize(meta.get('bpm',    ''))
        mood      = sanitize(meta.get('mood',   ''))
        popularity    = ''
        lyrics_snippet = ''
        tags_val      = ''

        row = [title, artist, album, year, genre, duration,
               bpm, mood, popularity, lyrics_snippet, tags_val, rel_path]

        rows.append(row)
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    print(f"  Writing {len(rows)} rows...", flush=True)

    with open(output_file, 'w', encoding='utf-8') as out:
        def pad(val, width):
            return str(val).ljust(width)

        header = '| ' + ' | '.join(pad(c, col_widths[i]) for i, c in enumerate(columns)) + ' |'
        sep    = '|-' + '-|-'.join('-' * col_widths[i] for i in range(len(columns))) + '-|'
        out.write(header + '\n')
        out.write(sep    + '\n')

        for row in rows:
            line = '| ' + ' | '.join(pad(row[j], col_widths[j]) for j in range(len(columns))) + ' |'
            out.write(line + '\n')

    print(f"\nDone! {len(rows)} rows → {output_file}")


if __name__ == '__main__':
    main()
