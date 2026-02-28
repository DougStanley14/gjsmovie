#!/usr/bin/env python3
"""
Copy all target songs from setlist files to ./movie-content/music-playlist/.
Uses fullmusictable.txt as the lookup table to find source file paths.
"""

import argparse
import re
import shutil
from pathlib import Path
from difflib import SequenceMatcher


# All unique songs from setlist1.txt, setlist1.1.txt, setlist-a.txt, setlist-b.txt, setlist-c.txt
TARGET_SONGS = [
    # --- setlist1 / all setlists ---
    ("The Beatles",                 "Here Comes The Sun"),
    ("Supertramp",                  "Give A Little Bit"),
    ("Neil Young",                  "Heart of Gold"),
    ("Van Morrison",                "Into The Mystic"),
    ("Cream",                       "Sunshine Of Your Love"),
    ("Creedence Clearwater Revival","Have You Ever Seen The Rain?"),
    ("Tom Petty",                   "Free Fallin'"),
    ("The Who",                     "Baba O'Riley"),
    ("David Bowie",                 "Heroes"),
    ("The Rolling Stones",          "Jumpin' Jack Flash"),
    ("The Allman Brothers Band",    "Ramblin' Man"),
    ("The Band",                    "The Weight"),
    ("Elvis Costello",              "(What's So Funny 'Bout) Peace, Love & Understanding"),
    ("Neil Young",                  "Long May You Run"),
    ("The Rolling Stones",          "Wild Horses"),
    # --- setlist1.1 swaps ---
    ("CSNY",                        "Carry On"),
    ("Creedence Clearwater Revival","Long As I Can See The Light"),
    ("Pink Floyd",                  "Wish You Were Here"),
    ("The Who",                     "Getting in Tune"),
    ("Yes",                         "I've Seen All Good People"),
    ("The Rolling Stones",          "Sweet Virginia"),
    ("The Beach Boys",              "God Only Knows"),
    ("Eagles",                      "Hotel California"),
    # --- setlist-a ---
    ("Genesis",                     "I Know What I Like (In Your Wardrobe)"),
    # --- setlist-b ---
    ("Supertramp",                  "The Logical Song"),
    # --- manual additions ---
    ("Cat Stevens",                 "The Wind"),
]


def normalize(s):
    """Lowercase, strip punctuation/articles for fuzzy comparison."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)      # remove punctuation
    s = re.sub(r"\s+", " ", s)         # collapse whitespace
    # strip common leading articles
    s = re.sub(r"^(the|a|an) ", "", s)
    return s


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def parse_table(table_path):
    """Parse pipe-delimited table into list of dicts."""
    records = []
    with open(table_path, encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return records

    # Parse header
    header_line = lines[0]
    columns = [c.strip() for c in header_line.strip().strip("|").split("|")]

    for line in lines[2:]:   # skip header + separator
        line = line.strip()
        if not line or line.startswith("|-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < len(columns):
            cells += [""] * (len(columns) - len(cells))
        records.append(dict(zip(columns, cells)))

    return records


def find_best_match(artist, title, records, artist_thresh=0.75, title_thresh=0.80):
    """Find the best matching record for (artist, title)."""
    norm_artist = normalize(artist)
    norm_title  = normalize(title)

    best_score = 0
    best_record = None

    for rec in records:
        rec_artist = normalize(rec.get("artist", ""))
        rec_title  = normalize(rec.get("title",  ""))

        a_score = similarity(norm_artist, rec_artist)
        t_score = similarity(norm_title,  rec_title)

        if a_score >= artist_thresh and t_score >= title_thresh:
            combined = (a_score + t_score) / 2
            if combined > best_score:
                best_score  = combined
                best_record = rec

    return best_record, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Copy setlist songs from music library to movie-content/music-playlist/."
    )
    parser.add_argument(
        "--table",
        default=r"d:\greg-memorial\gjsmovie\movie-content\fullmusictable.txt",
        help="Path to fullmusictable.txt"
    )
    parser.add_argument(
        "--dest",
        default=r"d:\greg-memorial\gjsmovie\movie-content\music-playlist",
        help="Destination folder"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying"
    )
    args = parser.parse_args()

    table_path = Path(args.table)
    dest_dir   = Path(args.dest)

    if not table_path.exists():
        print(f"ERROR: Table file not found: {table_path}")
        return

    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing table: {table_path}")
    records = parse_table(table_path)
    print(f"  {len(records)} records loaded.\n")

    found    = []
    not_found = []

    for artist, title in TARGET_SONGS:
        match, score = find_best_match(artist, title, records)

        if match:
            src_path_str = match.get("file", "").strip()
            src = Path(src_path_str)
            label = f"  [{score:.2f}] {artist} — {title}"
            print(f"FOUND:   {label}")
            print(f"         src: {src}")

            if src.exists():
                dest_file = dest_dir / src.name
                # Avoid collisions: prefix with artist if same filename
                if dest_file.exists() and dest_file != dest_dir / src.name:
                    dest_file = dest_dir / f"{artist} - {src.name}"
                if not args.dry_run:
                    shutil.copy2(src, dest_file)
                    print(f"         → copied to {dest_file.name}")
                else:
                    print(f"         → would copy to {dest_file.name}")
                found.append((artist, title, src.name))
            else:
                print(f"         *** SOURCE FILE NOT ACCESSIBLE: {src}")
                not_found.append((artist, title, f"File not accessible: {src_path_str}"))
        else:
            print(f"NO MATCH: {artist} — {title}")
            not_found.append((artist, title, "Not found in table"))

        print()

    # Summary
    print("=" * 60)
    print(f"SUMMARY: {len(found)} found / {len(not_found)} not found / {len(TARGET_SONGS)} total")
    print()

    if found:
        print("Copied:")
        for artist, title, fname in found:
            print(f"  ✓  {artist} — {title}  →  {fname}")

    if not_found:
        print("\nNot found:")
        for artist, title, reason in not_found:
            print(f"  ✗  {artist} — {title}  ({reason})")

    if not args.dry_run and found:
        print(f"\nFiles copied to: {dest_dir}")


if __name__ == "__main__":
    main()
