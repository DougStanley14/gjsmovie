#!/usr/bin/env python3
"""
randomize_photos.py — Randomize and rename photos for the slideshow.

Reads photos from  movie-content/pic-playlist/nonrandom/
Copies them into   movie-content/pic-playlist/

Cherry-picked photos (listed in a text file or via CLI) are placed first
in the specified order. All remaining photos are shuffled randomly.

Output filenames are sequential starting from 000100, always as .jpg:
    000100.jpg, 000101.jpg, 000102.jpg, ...

HEIC/HEIF files are automatically converted to JPG during copy.

Usage:
    # Basic — randomize all photos
    python randomize_photos.py

    # Cherry-pick first photos (by original filename)
    python randomize_photos.py --first "IMG_001.jpg, IMG_050.jpg, family.png"

    # Cherry-pick from a text file (one filename per line)
    python randomize_photos.py --first-file cherry_picks.txt

    # Preview without copying
    python randomize_photos.py --dry-run

    # Clear destination before copying
    python randomize_photos.py --clean

    # Change starting number (default 100)
    python randomize_photos.py --start 200
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

from PIL import Image
import pillow_heif

# Register HEIF/HEIC opener with Pillow
pillow_heif.register_heif_opener()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "movie-content" / "pic-playlist" / "nonrandom"
DEST_DIR = PROJECT_ROOT / "movie-content" / "pic-playlist"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_photos(directory: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    photos = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    photos.sort(key=lambda p: p.name.lower())
    return photos


def parse_cherry_picks(first_arg: str | None, first_file_arg: str | None) -> list[str]:
    """Parse cherry-pick filenames from CLI string and/or text file."""
    picks = []

    # From --first "file1.jpg, file2.jpg"
    if first_arg:
        for name in first_arg.split(","):
            name = name.strip()
            if name:
                picks.append(name)

    # From --first-file cherry_picks.txt
    if first_file_arg:
        fpath = Path(first_file_arg)
        if not fpath.is_absolute():
            fpath = PROJECT_ROOT / fpath
        if fpath.exists():
            for line in fpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    picks.append(line)
        else:
            logger.warning("Cherry-pick file not found: %s", fpath)

    return picks


def clean_destination(dest_dir: Path, source_dir: Path):
    """Remove previously numbered photos from destination (not the nonrandom folder)."""
    removed = 0
    for f in dest_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            # Don't touch the source subdirectory
            if f == source_dir or source_dir in f.parents:
                continue
            f.unlink()
            removed += 1
    if removed:
        logger.info("Cleaned %d existing photos from %s", removed, dest_dir)


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------
def randomize_and_copy(
    source_dir: Path,
    dest_dir: Path,
    cherry_picks: list[str],
    start_number: int = 100,
    dry_run: bool = False,
    clean: bool = False,
    seed: int | None = None,
):
    """Randomize photos and copy with sequential numbering."""

    # Validate source
    if not source_dir.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        logger.error("Move your base photos into: %s", source_dir)
        sys.exit(1)

    # Discover all photos
    all_photos = discover_photos(source_dir)
    if not all_photos:
        logger.error("No image files found in %s", source_dir)
        sys.exit(1)

    logger.info("Found %d photos in %s", len(all_photos), source_dir)

    # Build lookup by filename (case-insensitive)
    photo_map = {p.name.lower(): p for p in all_photos}

    # Separate cherry-picked from the rest
    first_photos = []
    remaining_photos = list(all_photos)

    for pick_name in cherry_picks:
        key = pick_name.lower()
        if key in photo_map:
            photo = photo_map[key]
            if photo in remaining_photos:
                remaining_photos.remove(photo)
                first_photos.append(photo)
                logger.info("  Cherry-picked: %s", photo.name)
            else:
                logger.warning("  Duplicate cherry-pick ignored: %s", pick_name)
        else:
            logger.warning("  Cherry-pick not found, skipping: %s", pick_name)

    # Shuffle remaining
    if seed is not None:
        random.seed(seed)
    random.shuffle(remaining_photos)

    # Final ordered list
    ordered = first_photos + remaining_photos

    logger.info("Order: %d cherry-picked + %d randomized = %d total",
                len(first_photos), len(remaining_photos), len(ordered))

    # Clean destination if requested
    if clean and not dry_run:
        clean_destination(dest_dir, source_dir)

    # Ensure destination exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy with sequential numbering
    num_width = 6  # 000100, 000101, ...
    copied = 0

    print()
    print("=" * 60)
    if dry_run:
        print("  DRY RUN — Photo Randomization Preview")
    else:
        print("  Photo Randomization")
    print("=" * 60)

    heic_converted = 0
    for i, photo in enumerate(ordered):
        num = start_number + i
        is_heic = photo.suffix.lower() in {".heic", ".heif"}
        out_ext = ".jpg" if is_heic else photo.suffix.lower()
        new_name = f"{num:0{num_width}d}{out_ext}"
        dest_path = dest_dir / new_name

        tag = "FIRST" if photo in first_photos else "     "
        conv = " [HEIC→JPG]" if is_heic else ""
        print(f"  {tag}  {new_name}  ←  {photo.name}{conv}")

        if not dry_run:
            if is_heic:
                img = Image.open(photo)
                img = img.convert("RGB")
                img.save(dest_path, "JPEG", quality=95)
                heic_converted += 1
            else:
                shutil.copy2(photo, dest_path)
            copied += 1

    print("=" * 60)

    if dry_run:
        print(f"\n  Dry run complete — {len(ordered)} photos would be copied.")
        print(f"  Starting number: {start_number:0{num_width}d}")
        print(f"  Ending number:   {start_number + len(ordered) - 1:0{num_width}d}")
    else:
        logger.info("Copied %d photos to %s", copied, dest_dir)
        if heic_converted:
            logger.info("Converted %d HEIC files to JPG", heic_converted)
        logger.info("Range: %s → %s",
                     f"{start_number:0{num_width}d}",
                     f"{start_number + len(ordered) - 1:0{num_width}d}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Randomize and sequentially rename photos for the slideshow."
    )
    parser.add_argument(
        "--first",
        type=str,
        default=None,
        help='Comma-separated filenames to place first, e.g. "IMG_001.jpg, IMG_050.jpg"'
    )
    parser.add_argument(
        "--first-file",
        type=str,
        default=None,
        help="Path to a text file listing cherry-pick filenames (one per line)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=100,
        help="Starting number for output filenames (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the randomized order without copying files"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing photos from destination before copying"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible ordering"
    )
    args = parser.parse_args()

    cherry_picks = parse_cherry_picks(args.first, args.first_file)

    randomize_and_copy(
        source_dir=SOURCE_DIR,
        dest_dir=DEST_DIR,
        cherry_picks=cherry_picks,
        start_number=args.start,
        dry_run=args.dry_run,
        clean=args.clean,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
