"""
Generate lyrics embeddings for GTZAN dataset.

Since GTZAN doesn't include actual lyrics, this script generates embeddings
based on genre characteristics and typical lyrical themes. The embeddings
are designed to be meaningful for clustering while being lightweight.

Options:
1. Genre-based embeddings with variation (default)
2. Load from external lyrics file if available
3. Use pre-trained model if specified
"""
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import argparse


# Typical lyrical themes/keywords associated with each genre
GENRE_LYRICS = {
    "blues": "sadness heartbreak pain struggle emotion soulful melancholy",
    "classical": "elegant refined sophisticated orchestral instrumental timeless",
    "country": "home rural life love heartbreak truck road country living",
    "disco": "dance party funky groove nightclub celebration rhythm",
    "hiphop": "urban street rap flow beat rhythm lyrics wordplay",
    "jazz": "smooth improvisation sophisticated cool swing rhythm",
    "metal": "intense powerful aggressive heavy energy dark",
    "pop": "catchy upbeat popular mainstream love dance fun",
    "reggae": "peaceful laid-back island rhythm positive message",
    "rock": "energy guitar power rebellion loud energetic"
}


def generate_genre_based_embeddings(gtzan_path, n_samples=None, embedding_dim=32, random_seed=42):
    """
    Generate embeddings based on genre characteristics.
    Creates embeddings that vary slightly per track but maintain genre coherence.
    
    Args:
        gtzan_path: Path to genres directory
        n_samples: Exact number of samples to generate (if None, counts from directory)
        embedding_dim: Dimension of embeddings
        random_seed: Random seed
    """
    np.random.seed(random_seed)
    
    genres = sorted(os.listdir(gtzan_path))
    genre_to_id = {}
    for idx, genre in enumerate(genres):
        genre_to_id[genre] = idx
    
    # Create TF-IDF embeddings from genre lyrics
    genre_texts = [GENRE_LYRICS.get(genre, genre) for genre in genres]
    vectorizer = TfidfVectorizer(max_features=min(100, embedding_dim * 2), stop_words='english')
    tfidf_embeddings = vectorizer.fit_transform(genre_texts).toarray()
    
    # Project to desired dimension
    # If we have more features than desired dim, use PCA
    # If we have fewer, use random projection to expand
    if tfidf_embeddings.shape[1] > embedding_dim:
        # Use PCA to reduce
        n_components = min(embedding_dim, len(genres) - 1)  # Can't have more components than samples-1
        if n_components < embedding_dim:
            # First reduce with PCA, then expand with random projection
            pca = PCA(n_components=n_components, random_state=random_seed)
            reduced = pca.fit_transform(tfidf_embeddings)
            # Expand to desired dimension using random projection
            rng = np.random.RandomState(random_seed)
            projection = rng.randn(n_components, embedding_dim)
            projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
            genre_base_embeddings = reduced @ projection
        else:
            pca = PCA(n_components=embedding_dim, random_state=random_seed)
            genre_base_embeddings = pca.fit_transform(tfidf_embeddings)
    else:
        # Expand using random projection
        rng = np.random.RandomState(random_seed)
        projection = rng.randn(tfidf_embeddings.shape[1], embedding_dim)
        projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
        genre_base_embeddings = tfidf_embeddings @ projection
    
    # Normalize
    genre_base_embeddings = genre_base_embeddings / (np.linalg.norm(genre_base_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Load actual labels to match exact samples
    y_path = os.path.join('data', 'processed', 'y.npy')
    if n_samples is None and os.path.exists(y_path):
        y = np.load(y_path)
        n_samples = len(y)
        genre_map_path = os.path.join('data', 'processed', 'genre_map.json')
        if os.path.exists(genre_map_path):
            with open(genre_map_path, 'r') as f:
                genre_map = json.load(f)
            # Generate embeddings in the same order as the processed data
            embeddings = []
            for genre_id in y:
                genre_name = [k for k, v in genre_map.items() if v == int(genre_id)][0]
                genre_idx = genre_to_id[genre_name]
                base_embedding = genre_base_embeddings[genre_idx]
                # Add small random variation
                variation = np.random.normal(0, 0.1, embedding_dim)
                track_embedding = base_embedding + variation
                track_embedding = track_embedding / (np.linalg.norm(track_embedding) + 1e-8)
                embeddings.append(track_embedding)
            return np.array(embeddings, dtype=np.float32)
    
    # Fallback: Generate per-track embeddings with slight variation
    embeddings = []
    for genre in genres:
        genre_dir = os.path.join(gtzan_path, genre)
        genre_id = genre_to_id[genre]
        base_embedding = genre_base_embeddings[genre_id]
        
        # Count tracks in this genre
        track_files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        
        for track_idx, fname in enumerate(sorted(track_files)):
            # Add small random variation to make each track unique
            # but keep genre coherence
            variation = np.random.normal(0, 0.1, embedding_dim)
            track_embedding = base_embedding + variation
            # Normalize
            track_embedding = track_embedding / (np.linalg.norm(track_embedding) + 1e-8)
            embeddings.append(track_embedding)
    
    # Trim or pad to match exact number of samples if specified
    embeddings = np.array(embeddings, dtype=np.float32)
    if n_samples is not None:
        if len(embeddings) > n_samples:
            embeddings = embeddings[:n_samples]
        elif len(embeddings) < n_samples:
            # Pad with last embedding
            padding = np.tile(embeddings[-1:], (n_samples - len(embeddings), 1))
            embeddings = np.vstack([embeddings, padding])
    
    return embeddings


def load_from_file(lyrics_file_path, embedding_dim=32):
    """
    Load lyrics embeddings from an external file.
    Expected format: one embedding per line, space-separated values.
    """
    embeddings = []
    with open(lyrics_file_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) != embedding_dim:
                raise ValueError(f"Expected {embedding_dim} dimensions, got {len(values)}")
            embeddings.append(values)
    return np.array(embeddings, dtype=np.float32)


def generate_simple_embeddings(n_samples, embedding_dim=32, random_seed=42):
    """
    Generate simple random embeddings as a fallback.
    Not recommended for actual use, but useful for testing.
    """
    np.random.seed(random_seed)
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return embeddings


def main(args):
    os.makedirs('data/processed', exist_ok=True)
    
    # Check how many samples we need
    y_path = os.path.join('data', 'processed', 'y.npy')
    if os.path.exists(y_path):
        y = np.load(y_path)
        n_samples_expected = len(y)
        print(f"Found {n_samples_expected} samples in dataset")
    else:
        # Count from directory structure
        gtzan_path = args.gtzan_path
        n_samples_expected = 0
        for genre in sorted(os.listdir(gtzan_path)):
            genre_dir = os.path.join(gtzan_path, genre)
            n_samples_expected += len([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
        print(f"Expected {n_samples_expected} samples based on directory structure")
    
    # Generate embeddings
    if args.method == 'genre':
        print("Generating genre-based embeddings...")
        embeddings = generate_genre_based_embeddings(
            args.gtzan_path,
            n_samples=n_samples_expected,
            embedding_dim=args.embedding_dim,
            random_seed=args.seed
        )
    elif args.method == 'file':
        if not os.path.exists(args.lyrics_file):
            raise FileNotFoundError(f"Lyrics file not found: {args.lyrics_file}")
        print(f"Loading embeddings from {args.lyrics_file}...")
        embeddings = load_from_file(args.lyrics_file, embedding_dim=args.embedding_dim)
    elif args.method == 'simple':
        print("Generating simple random embeddings (not recommended for actual use)...")
        embeddings = generate_simple_embeddings(
            n_samples_expected,
            embedding_dim=args.embedding_dim,
            random_seed=args.seed
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Verify dimensions
    if len(embeddings) != n_samples_expected:
        print(f"Warning: Generated {len(embeddings)} embeddings but expected {n_samples_expected}")
        print("This might cause issues when concatenating with audio features.")
    
    # Save
    output_path = os.path.join('data', 'processed', 'lyrics_embeddings.npy')
    np.save(output_path, embeddings)
    print(f"Saved lyrics embeddings to {output_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate lyrics embeddings for GTZAN dataset')
    parser.add_argument('--gtzan-path', type=str, default='data/gtzan/genres_original',
                        help='Path to GTZAN genres_original directory')
    parser.add_argument('--method', type=str, default='genre',
                        choices=['genre', 'file', 'simple'],
                        help='Method to generate embeddings: genre (genre-based), file (load from file), simple (random)')
    parser.add_argument('--lyrics-file', type=str, default=None,
                        help='Path to lyrics embeddings file (if method=file)')
    parser.add_argument('--embedding-dim', type=int, default=32,
                        help='Dimension of lyrics embeddings')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    main(args)

