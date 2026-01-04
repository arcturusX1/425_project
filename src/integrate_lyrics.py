"""
Integrate real lyrics data with GTZAN dataset.
Supports multiple lyrics data sources and formats.
"""
import os
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


def load_lyrics_from_csv(csv_path, track_id_col='track_id', lyrics_col='lyrics', genre_col=None):
    """
    Load lyrics from CSV file.
    
    Expected CSV format:
    - track_id: Identifier matching GTZAN filename (e.g., 'blues.00000' or '00000')
    - lyrics: Text lyrics for the track
    - genre: Optional genre column for validation
    
    Args:
        csv_path: Path to CSV file
        track_id_col: Column name for track identifier
        lyrics_col: Column name for lyrics text
        genre_col: Optional column name for genre (for validation)
    
    Returns:
        Dictionary mapping track_id -> lyrics
    """
    df = pd.read_csv(csv_path)
    
    lyrics_dict = {}
    for _, row in df.iterrows():
        track_id = str(row[track_id_col]).strip()
        lyrics = str(row[lyrics_col]).strip()
        
        if pd.isna(lyrics) or lyrics == '' or lyrics.lower() == 'nan':
            continue
        
        lyrics_dict[track_id] = lyrics
    
    print(f"Loaded {len(lyrics_dict)} lyrics from CSV")
    return lyrics_dict


def load_lyrics_from_json(json_path, track_id_key='track_id', lyrics_key='lyrics'):
    """
    Load lyrics from JSON file.
    
    Expected JSON format:
    [
        {"track_id": "blues.00000", "lyrics": "lyrics text here"},
        ...
    ]
    or
    {
        "blues.00000": "lyrics text here",
        ...
    }
    
    Args:
        json_path: Path to JSON file
        track_id_key: Key for track identifier (if list format)
        lyrics_key: Key for lyrics text (if list format)
    
    Returns:
        Dictionary mapping track_id -> lyrics
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lyrics_dict = {}
    
    if isinstance(data, list):
        # List format
        for item in data:
            track_id = str(item[track_id_key]).strip()
            lyrics = str(item[lyrics_key]).strip()
            if lyrics and lyrics.lower() != 'nan':
                lyrics_dict[track_id] = lyrics
    elif isinstance(data, dict):
        # Dictionary format
        for track_id, lyrics in data.items():
            track_id = str(track_id).strip()
            lyrics = str(lyrics).strip()
            if lyrics and lyrics.lower() != 'nan':
                lyrics_dict[track_id] = lyrics
    
    print(f"Loaded {len(lyrics_dict)} lyrics from JSON")
    return lyrics_dict


def match_lyrics_to_gtzan(lyrics_dict, gtzan_path, match_strategy='filename'):
    """
    Match lyrics to GTZAN tracks.
    
    Args:
        lyrics_dict: Dictionary of track_id -> lyrics
        gtzan_path: Path to GTZAN genres_original directory
        match_strategy: How to match tracks
            - 'filename': Match by filename (e.g., 'blues.00000' -> 'blues.00000.wav')
            - 'basename': Match by basename (e.g., '00000' matches 'blues.00000.wav')
            - 'index': Match by index within genre (0, 1, 2, ...)
    
    Returns:
        Dictionary mapping (genre, filename) -> lyrics
    """
    matched_lyrics = {}
    unmatched_tracks = []
    
    genres = sorted(os.listdir(gtzan_path))
    
    for genre in genres:
        genre_dir = os.path.join(gtzan_path, genre)
        track_files = sorted([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
        
        for idx, filename in enumerate(track_files):
            track_id = None
            lyrics = None
            
            if match_strategy == 'filename':
                # Try exact filename match (with/without extension)
                base_name = filename.replace('.wav', '')
                track_id = base_name  # e.g., 'blues.00000'
                if track_id in lyrics_dict:
                    lyrics = lyrics_dict[track_id]
                else:
                    # Try with genre prefix
                    track_id = f"{genre}.{base_name.split('.')[-1]}"  # e.g., 'blues.00000'
                    if track_id in lyrics_dict:
                        lyrics = lyrics_dict[track_id]
            
            elif match_strategy == 'basename':
                # Match by number only (e.g., '00000' matches 'blues.00000.wav')
                base_name = filename.replace('.wav', '').split('.')[-1]  # e.g., '00000'
                if base_name in lyrics_dict:
                    lyrics = lyrics_dict[base_name]
                else:
                    # Try with genre prefix
                    track_id = f"{genre}.{base_name}"
                    if track_id in lyrics_dict:
                        lyrics = lyrics_dict[track_id]
            
            elif match_strategy == 'index':
                # Match by index (first track = 0, second = 1, etc.)
                track_id = str(idx)
                if track_id in lyrics_dict:
                    lyrics = lyrics_dict[track_id]
            
            if lyrics:
                matched_lyrics[(genre, filename)] = lyrics
            else:
                unmatched_tracks.append((genre, filename))
    
    print(f"Matched {len(matched_lyrics)} tracks with lyrics")
    print(f"Unmatched {len(unmatched_tracks)} tracks")
    
    if len(unmatched_tracks) > 0 and len(unmatched_tracks) <= 10:
        print("Unmatched tracks (first 10):")
        for genre, filename in unmatched_tracks[:10]:
            print(f"  {genre}/{filename}")
    
    return matched_lyrics, unmatched_tracks


def save_lyrics_mapping(matched_lyrics, output_path):
    """
    Save lyrics mapping to JSON file.
    
    Format:
    {
        "genre/filename": "lyrics text",
        ...
    }
    """
    mapping = {}
    for (genre, filename), lyrics in matched_lyrics.items():
        key = f"{genre}/{filename}"
        mapping[key] = lyrics
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Saved lyrics mapping to {output_path}")
    return mapping


def main(args):
    """Main function to integrate lyrics with GTZAN."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load lyrics
    lyrics_dict = {}
    if args.lyrics_csv:
        lyrics_dict = load_lyrics_from_csv(
            args.lyrics_csv,
            track_id_col=args.track_id_col,
            lyrics_col=args.lyrics_col,
            genre_col=args.genre_col
        )
    elif args.lyrics_json:
        lyrics_dict = load_lyrics_from_json(
            args.lyrics_json,
            track_id_key=args.track_id_key,
            lyrics_key=args.lyrics_key
        )
    else:
        raise ValueError("Must provide either --lyrics-csv or --lyrics-json")
    
    # Match lyrics to GTZAN tracks
    gtzan_path = os.path.join(project_root, args.gtzan_path)
    matched_lyrics, unmatched = match_lyrics_to_gtzan(
        lyrics_dict,
        gtzan_path,
        match_strategy=args.match_strategy
    )
    
    # Save mapping
    output_path = os.path.join(project_root, 'data', 'lyrics', 'lyrics_mapping.json')
    save_lyrics_mapping(matched_lyrics, output_path)
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total lyrics loaded: {len(lyrics_dict)}")
    print(f"  Matched tracks: {len(matched_lyrics)}")
    print(f"  Unmatched tracks: {len(unmatched)}")
    print(f"  Match rate: {len(matched_lyrics) / (len(matched_lyrics) + len(unmatched)) * 100:.1f}%")
    
    return matched_lyrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate lyrics data with GTZAN dataset')
    parser.add_argument('--lyrics-csv', type=str, default=None,
                        help='Path to CSV file with lyrics')
    parser.add_argument('--lyrics-json', type=str, default=None,
                        help='Path to JSON file with lyrics')
    parser.add_argument('--gtzan-path', type=str, default='data/gtzan/genres_original',
                        help='Path to GTZAN genres_original directory')
    parser.add_argument('--match-strategy', type=str, default='filename',
                        choices=['filename', 'basename', 'index'],
                        help='Strategy for matching lyrics to tracks')
    parser.add_argument('--track-id-col', type=str, default='track_id',
                        help='CSV column name for track identifier')
    parser.add_argument('--lyrics-col', type=str, default='lyrics',
                        help='CSV column name for lyrics text')
    parser.add_argument('--genre-col', type=str, default=None,
                        help='CSV column name for genre (optional)')
    parser.add_argument('--track-id-key', type=str, default='track_id',
                        help='JSON key for track identifier')
    parser.add_argument('--lyrics-key', type=str, default='lyrics',
                        help='JSON key for lyrics text')
    args = parser.parse_args()
    
    main(args)

