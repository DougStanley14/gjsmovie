#!/usr/bin/env python3
"""
Generate music-list.txt by traversing a folder and finding all music files.
Outputs a list of relative paths to music files.
"""

import argparse
import os
from pathlib import Path


def is_music_file(file_path):
    """Check if file is a music file based on extension."""
    music_extensions = {
        '.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', '.wma',
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.mpg', '.mpeg'
    }
    return file_path.suffix.lower() in music_extensions


def find_music_files(folder_path):
    """Recursively find all music files in folder and subfolders."""
    music_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"ERROR: Folder '{folder_path}' does not exist")
        return []
    
    if not folder.is_dir():
        print(f"ERROR: '{folder_path}' is not a directory")
        return []
    
    print(f"Scanning '{folder_path}' for music files...")
    
    for file_path in folder.rglob('*'):
        if file_path.is_file() and is_music_file(file_path):
            # Get relative path from the base folder
            rel_path = file_path.relative_to(folder)
            music_files.append(str(rel_path))
    
    music_files.sort()  # Sort alphabetically
    return music_files


def main():
    parser = argparse.ArgumentParser(description='Generate music-list.txt by traversing a folder for music files.')
    parser.add_argument('folder', 
                       help='Folder to scan for music files')
    parser.add_argument('-o', '--output', 
                       default='music-list.txt',
                       help='Output filename (default: music-list.txt)')
    
    args = parser.parse_args()
    
    music_files = find_music_files(args.folder)
    
    if not music_files:
        print("No music files found.")
        return
    
    output_path = Path(args.folder) / args.output
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for file_path in music_files:
                f.write(file_path + '\n')
        
        print(f"Found {len(music_files)} music files")
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR: Could not write to '{output_path}': {e}")


if __name__ == '__main__':
    main()
