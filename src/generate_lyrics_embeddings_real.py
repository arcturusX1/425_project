"""
Generate lyrics embeddings from real lyrics data.
Uses text embedding models (sentence transformers or TF-IDF) to create embeddings.
"""
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using TF-IDF instead.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


def load_lyrics_mapping(mapping_path):
    """Load lyrics mapping from JSON file."""
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping


def get_lyrics_for_tracks(lyrics_mapping, gtzan_path, genre_map):
    """
    Get lyrics for each track in the same order as processed data.
    
    Returns:
        List of lyrics texts in the same order as y.npy
    """
    # Load processed labels to get order
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    y_path = os.path.join(project_root, 'data', 'processed', 'y.npy')
    
    if not os.path.exists(y_path):
        raise FileNotFoundError("y.npy not found. Run dataset.py first to establish track order.")
    
    y = np.load(y_path)
    
    # Load genre map
    genre_map_path = os.path.join(project_root, 'data', 'processed', 'genre_map.json')
    if os.path.exists(genre_map_path):
        with open(genre_map_path, 'r') as f:
            processed_genre_map = json.load(f)
    else:
        processed_genre_map = genre_map
    
    id_to_genre = {v: k for k, v in processed_genre_map.items()}
    
    # Get track order from GTZAN directory structure
    lyrics_list = []
    track_order = []
    
    for genre_id in y:
        genre_name = id_to_genre[int(genre_id)]
        genre_dir = os.path.join(gtzan_path, genre_name)
        track_files = sorted([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
        
        for filename in track_files:
            key = f"{genre_name}/{filename}"
            lyrics = lyrics_mapping.get(key, None)
            lyrics_list.append(lyrics)
            track_order.append((genre_name, filename))
    
    return lyrics_list, track_order


def generate_embeddings_sentence_transformer(lyrics_list, model_name='all-MiniLM-L6-v2', embedding_dim=None):
    """
    Generate embeddings using sentence transformers.
    
    Args:
        lyrics_list: List of lyrics texts (can contain None for missing lyrics)
        model_name: Sentence transformer model name
        embedding_dim: Target dimension (if None, uses model default)
    
    Returns:
        numpy array of embeddings [N, D]
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
    
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Replace None with empty string
    lyrics_clean = [lyrics if lyrics else "" for lyrics in lyrics_list]
    
    print("Generating embeddings...")
    embeddings = model.encode(lyrics_clean, show_progress_bar=True, convert_to_numpy=True)
    
    # Reduce dimension if needed
    if embedding_dim is not None and embeddings.shape[1] != embedding_dim:
        print(f"Reducing dimension from {embeddings.shape[1]} to {embedding_dim}")
        pca = PCA(n_components=embedding_dim, random_state=42)
        embeddings = pca.fit_transform(embeddings)
    
    return embeddings.astype(np.float32)


def generate_embeddings_tfidf(lyrics_list, embedding_dim=32, max_features=500):
    """
    Generate embeddings using TF-IDF.
    
    Args:
        lyrics_list: List of lyrics texts (can contain None for missing lyrics)
        embedding_dim: Target embedding dimension
        max_features: Maximum TF-IDF features before PCA
    
    Returns:
        numpy array of embeddings [N, D]
    """
    # Replace None with empty string
    lyrics_clean = [lyrics if lyrics else "" for lyrics in lyrics_list]
    
    print("Computing TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95  # Maximum document frequency
    )
    
    tfidf_matrix = vectorizer.fit_transform(lyrics_clean)
    tfidf_features = tfidf_matrix.toarray()
    
    print(f"TF-IDF features shape: {tfidf_features.shape}")
    
    # Reduce to target dimension
    if tfidf_features.shape[1] > embedding_dim:
        print(f"Reducing dimension from {tfidf_features.shape[1]} to {embedding_dim} using PCA")
        pca = PCA(n_components=embedding_dim, random_state=42)
        embeddings = pca.fit_transform(tfidf_features)
    else:
        # Pad if smaller
        embeddings = np.zeros((tfidf_features.shape[0], embedding_dim))
        embeddings[:, :tfidf_features.shape[1]] = tfidf_features
        if tfidf_features.shape[1] < embedding_dim:
            print(f"Warning: TF-IDF features ({tfidf_features.shape[1]}) < target dimension ({embedding_dim}). Padding with zeros.")
    
    return embeddings.astype(np.float32)


def handle_missing_lyrics(embeddings, lyrics_list, strategy='zero'):
    """
    Handle missing lyrics (None values).
    
    Args:
        embeddings: Embedding array [N, D]
        lyrics_list: List of lyrics (may contain None)
        strategy: How to handle missing lyrics
            - 'zero': Set to zero vector
            - 'mean': Set to mean of available embeddings
            - 'genre_mean': Set to mean of same-genre embeddings (requires genre info)
    
    Returns:
        Updated embeddings
    """
    missing_indices = [i for i, lyrics in enumerate(lyrics_list) if lyrics is None or lyrics == ""]
    
    if len(missing_indices) == 0:
        return embeddings
    
    print(f"Handling {len(missing_indices)} missing lyrics using strategy: {strategy}")
    
    if strategy == 'zero':
        # Already zero from empty string, no action needed
        pass
    elif strategy == 'mean':
        available_embeddings = embeddings[[i for i in range(len(embeddings)) if i not in missing_indices]]
        mean_embedding = available_embeddings.mean(axis=0)
        for idx in missing_indices:
            embeddings[idx] = mean_embedding
    elif strategy == 'genre_mean':
        # Would need genre information - for now, use mean
        available_embeddings = embeddings[[i for i in range(len(embeddings)) if i not in missing_indices]]
        mean_embedding = available_embeddings.mean(axis=0)
        for idx in missing_indices:
            embeddings[idx] = mean_embedding
    
    return embeddings


def main(args):
    """Main function to generate embeddings from real lyrics."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load lyrics mapping
    mapping_path = os.path.join(project_root, 'data', 'lyrics', 'lyrics_mapping.json')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Lyrics mapping not found at {mapping_path}. "
            "Run integrate_lyrics.py first to create the mapping."
        )
    
    print(f"Loading lyrics mapping from {mapping_path}")
    lyrics_mapping = load_lyrics_mapping(mapping_path)
    
    # Get lyrics for tracks in correct order
    gtzan_path = os.path.join(project_root, args.gtzan_path)
    genre_map_path = os.path.join(project_root, 'data', 'processed', 'genre_map.json')
    
    if os.path.exists(genre_map_path):
        with open(genre_map_path, 'r') as f:
            genre_map = json.load(f)
    else:
        # Create genre map from directory
        genres = sorted(os.listdir(gtzan_path))
        genre_map = {genre: idx for idx, genre in enumerate(genres)}
    
    lyrics_list, track_order = get_lyrics_for_tracks(lyrics_mapping, gtzan_path, genre_map)
    
    # Count missing lyrics
    missing_count = sum(1 for lyrics in lyrics_list if lyrics is None or lyrics == "")
    print(f"Lyrics available: {len(lyrics_list) - missing_count}/{len(lyrics_list)}")
    
    # Generate embeddings
    if args.method == 'sentence_transformer':
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("sentence-transformers not available. Falling back to TF-IDF.")
            args.method = 'tfidf'
        else:
            embeddings = generate_embeddings_sentence_transformer(
                lyrics_list,
                model_name=args.model_name,
                embedding_dim=args.embedding_dim
            )
    
    if args.method == 'tfidf':
        embeddings = generate_embeddings_tfidf(
            lyrics_list,
            embedding_dim=args.embedding_dim,
            max_features=args.max_features
        )
    
    # Handle missing lyrics
    embeddings = handle_missing_lyrics(embeddings, lyrics_list, strategy=args.missing_strategy)
    
    # Save embeddings
    output_path = os.path.join(project_root, 'data', 'processed', 'lyrics_embeddings.npy')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    
    print(f"\nSaved lyrics embeddings to {output_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Missing lyrics: {missing_count}/{len(lyrics_list)} ({missing_count/len(lyrics_list)*100:.1f}%)")
    
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate lyrics embeddings from real lyrics data')
    parser.add_argument('--gtzan-path', type=str, default='data/gtzan/genres_original',
                        help='Path to GTZAN genres_original directory')
    parser.add_argument('--method', type=str, default='sentence_transformer',
                        choices=['sentence_transformer', 'tfidf'],
                        help='Embedding method')
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name (if using sentence_transformer)')
    parser.add_argument('--embedding-dim', type=int, default=32,
                        help='Target embedding dimension')
    parser.add_argument('--max-features', type=int, default=500,
                        help='Max TF-IDF features (if using tfidf)')
    parser.add_argument('--missing-strategy', type=str, default='mean',
                        choices=['zero', 'mean', 'genre_mean'],
                        help='How to handle missing lyrics')
    args = parser.parse_args()
    
    main(args)

