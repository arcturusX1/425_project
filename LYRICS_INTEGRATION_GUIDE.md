# Lyrics Integration Guide

This guide explains how to integrate real lyrics data with the GTZAN dataset to create authentic lyrics embeddings for multimodal clustering.

## Overview

Instead of using synthetic genre-based embeddings, you can integrate real lyrics data to create more meaningful multimodal features. This guide covers:

1. Where to get lyrics data
2. How to format lyrics data
3. How to match lyrics to GTZAN tracks
4. How to generate embeddings from real lyrics

## Step 1: Obtain Lyrics Data

### Option 1: Lyrics APIs

**Genius API** (Recommended):
- Website: https://genius.com/api-clients
- Free tier available
- Good coverage of popular songs
- Python library: `lyricsgenius`

**Musixmatch API**:
- Website: https://developer.musixmatch.com/
- Free tier: 500 requests/day
- Large database

**Example using Genius API**:
```python
import lyricsgenius

genius = lyricsgenius.Genius("YOUR_API_KEY")
song = genius.search_song("Title", "Artist")
lyrics = song.lyrics
```

### Option 2: Lyrics Datasets

**Kaggle Datasets**:
- Search for "lyrics dataset" or "song lyrics"
- Many datasets available with track metadata

**Million Song Dataset (MSD)**:
- Includes lyrics for many tracks
- Requires matching to GTZAN tracks

### Option 3: Manual Collection

For a small dataset like GTZAN (1000 tracks), you could manually collect lyrics or use web scraping (respecting terms of service).

## Step 2: Format Lyrics Data

### CSV Format

Create a CSV file with columns:
- `track_id`: Identifier matching GTZAN filename
- `lyrics`: Full lyrics text
- `genre`: Optional, for validation

**Example CSV** (`data/lyrics/lyrics.csv`):
```csv
track_id,lyrics,genre
blues.00000,"I woke up this morning, feeling so blue...",blues
blues.00001,"The sun is shining, but I'm feeling down...",blues
classical.00000,"[Instrumental - no lyrics]",classical
```

### JSON Format

**List format** (`data/lyrics/lyrics.json`):
```json
[
  {
    "track_id": "blues.00000",
    "lyrics": "I woke up this morning, feeling so blue...",
    "genre": "blues"
  },
  {
    "track_id": "blues.00001",
    "lyrics": "The sun is shining, but I'm feeling down...",
    "genre": "blues"
  }
]
```

**Dictionary format** (`data/lyrics/lyrics.json`):
```json
{
  "blues.00000": "I woke up this morning, feeling so blue...",
  "blues.00001": "The sun is shining, but I'm feeling down...",
  "classical.00000": "[Instrumental - no lyrics]"
}
```

## Step 3: Match Lyrics to GTZAN Tracks

GTZAN files are named like: `blues.00000.wav`, `classical.00001.wav`, etc.

### Matching Strategies

1. **Filename matching** (default):
   - Track ID: `blues.00000` matches `blues.00000.wav`
   - Most precise but requires exact naming

2. **Basename matching**:
   - Track ID: `00000` matches any `*.00000.wav`
   - More flexible if your lyrics dataset uses numeric IDs

3. **Index matching**:
   - Track ID: `0`, `1`, `2`, ... matches tracks in order
   - Use if lyrics are ordered by genre

### Run Integration Script

```bash
# From CSV
python3 src/integrate_lyrics.py \
    --lyrics-csv data/lyrics/lyrics.csv \
    --track-id-col track_id \
    --lyrics-col lyrics \
    --match-strategy filename

# From JSON
python3 src/integrate_lyrics.py \
    --lyrics-json data/lyrics/lyrics.json \
    --match-strategy filename
```

This creates `data/lyrics/lyrics_mapping.json` mapping GTZAN tracks to lyrics.

## Step 4: Generate Embeddings from Real Lyrics

### Option 1: Sentence Transformers (Recommended)

**Install**:
```bash
pip install sentence-transformers
```

**Generate embeddings**:
```bash
python3 src/generate_lyrics_embeddings_real.py \
    --method sentence_transformer \
    --model-name all-MiniLM-L6-v2 \
    --embedding-dim 32
```

**Available models**:
- `all-MiniLM-L6-v2`: Fast, 384-dim (default)
- `all-mpnet-base-v2`: Better quality, 768-dim
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support

### Option 2: TF-IDF (No Dependencies)

**Generate embeddings**:
```bash
python3 src/generate_lyrics_embeddings_real.py \
    --method tfidf \
    --embedding-dim 32 \
    --max-features 500
```

### Handling Missing Lyrics

If some tracks don't have lyrics, you can handle them with:

- `--missing-strategy zero`: Set to zero vector
- `--missing-strategy mean`: Set to mean of available embeddings (default)
- `--missing-strategy genre_mean`: Set to mean of same-genre embeddings

## Step 5: Use in Multimodal Clustering

Once embeddings are generated, they're automatically used by:

```bash
python3 src/multimodal_clustering.py --n-clusters 10
```

The script will load `data/processed/lyrics_embeddings.npy` if it exists.

## Complete Workflow Example

```bash
# 1. Collect lyrics (manual or API) and save to CSV/JSON
#    Example: data/lyrics/lyrics.csv

# 2. Integrate lyrics with GTZAN
python3 src/integrate_lyrics.py \
    --lyrics-csv data/lyrics/lyrics.csv \
    --match-strategy filename

# 3. Generate embeddings
python3 src/generate_lyrics_embeddings_real.py \
    --method sentence_transformer \
    --embedding-dim 32

# 4. Run multimodal clustering
python3 src/multimodal_clustering.py --n-clusters 10

# 5. Evaluate
python3 src/evaluate_hard_task.py
```

## Tips and Best Practices

### 1. Track ID Formatting

Ensure your track IDs match GTZAN filenames:
- GTZAN: `blues.00000.wav`
- Track ID should be: `blues.00000` (for filename matching)
- Or: `00000` (for basename matching)

### 2. Lyrics Preprocessing

Consider preprocessing lyrics:
- Remove metadata (e.g., "[Verse 1]", "[Chorus]")
- Normalize whitespace
- Handle special characters
- Remove very short lyrics (< 10 words might be instrumental)

### 3. Missing Lyrics

For tracks without lyrics (e.g., classical instrumental):
- Mark as empty string `""` or `"[Instrumental]"`
- The embedding generator will handle them using your chosen strategy

### 4. Embedding Dimension

- Match audio feature dimensions for better fusion
- Common choices: 32, 64, 128
- Sentence transformers: Use model default or reduce with PCA

### 5. Validation

Check the integration:
```python
import json
import numpy as np

# Check mapping
with open('data/lyrics/lyrics_mapping.json', 'r') as f:
    mapping = json.load(f)
print(f"Total tracks with lyrics: {len(mapping)}")

# Check embeddings
embeddings = np.load('data/processed/lyrics_embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")
print(f"Non-zero embeddings: {np.count_nonzero(embeddings)}")
```

## Troubleshooting

### Low Match Rate

If few tracks are matched:
1. Check track ID format in your lyrics file
2. Try different `--match-strategy` (basename, index)
3. Verify GTZAN filenames match your track IDs

### Missing Lyrics

If many tracks lack lyrics:
1. Use `--missing-strategy mean` to fill with average
2. Consider collecting more lyrics
3. For instrumental genres, this is expected

### Embedding Quality

If embeddings don't improve clustering:
1. Try different sentence transformer models
2. Increase embedding dimension
3. Preprocess lyrics better (remove noise)
4. Check if lyrics are actually genre-discriminative

## Expected Results

With real lyrics embeddings:
- **Better than synthetic**: Real lyrics should capture actual semantic content
- **Realistic performance**: NMI ~0.4-0.6, ARI ~0.3-0.5 (not 0.99!)
- **Genre-dependent**: Some genres (pop, hip-hop) benefit more than others (classical, jazz)

## References

- Sentence Transformers: https://www.sbert.net/
- Genius API: https://docs.genius.com/
- GTZAN Dataset: https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection

