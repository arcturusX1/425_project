"""
Multi-modal clustering combining audio, lyrics, and genre information.
Supports various fusion strategies: concatenation, weighted fusion, and late fusion.
"""
import os
import argparse
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def concatenate_features(audio_features, lyrics_features, genre_features=None, normalize=True):
    """
    Concatenate multi-modal features.
    
    Args:
        audio_features: Audio feature embeddings [N, D_audio]
        lyrics_features: Lyrics embeddings [N, D_lyrics]
        genre_features: Optional genre one-hot encodings [N, D_genre]
        normalize: Whether to normalize features before concatenation
    """
    features = [audio_features]
    
    if lyrics_features is not None:
        features.append(lyrics_features)
    
    if genre_features is not None:
        features.append(genre_features)
    
    if normalize:
        # Normalize each modality separately
        normalized_features = []
        for feat in features:
            scaler = StandardScaler()
            normalized_features.append(scaler.fit_transform(feat))
        features = normalized_features
    
    return np.concatenate(features, axis=1)


def weighted_fusion(audio_features, lyrics_features, genre_features=None, 
                   audio_weight=0.5, lyrics_weight=0.3, genre_weight=0.2, normalize=True):
    """
    Weighted fusion of multi-modal features.
    
    Args:
        audio_features: Audio feature embeddings [N, D_audio]
        lyrics_features: Lyrics embeddings [N, D_lyrics]
        genre_features: Optional genre one-hot encodings [N, D_genre]
        audio_weight: Weight for audio features
        lyrics_weight: Weight for lyrics features
        genre_weight: Weight for genre features
        normalize: Whether to normalize features before fusion
    """
    weights = [audio_weight]
    features = [audio_features]
    
    if lyrics_features is not None:
        weights.append(lyrics_weight)
        features.append(lyrics_features)
    else:
        # Renormalize weights if lyrics not available
        total = audio_weight + (genre_weight if genre_features is not None else 0)
        weights = [w / total for w in weights]
    
    if genre_features is not None:
        weights.append(genre_weight)
        features.append(genre_features)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    if normalize:
        # Normalize each modality separately
        normalized_features = []
        for feat in features:
            scaler = StandardScaler()
            normalized_features.append(scaler.fit_transform(feat))
        features = normalized_features
    
    # Weighted concatenation
    weighted_features = []
    for feat, weight in zip(features, weights):
        weighted_features.append(feat * weight)
    
    return np.concatenate(weighted_features, axis=1)


def create_genre_features(y, n_classes=None):
    """
    Create one-hot encoded genre features.
    
    Args:
        y: Genre labels [N]
        n_classes: Number of classes (if None, inferred from y)
    """
    if n_classes is None:
        n_classes = len(np.unique(y))
    
    genre_features = np.zeros((len(y), n_classes))
    genre_features[np.arange(len(y)), y] = 1.0
    return genre_features


def main(args):
    os.makedirs('data/processed', exist_ok=True)
    
    # Load features
    audio_features = None
    lyrics_features = None
    genre_labels = None
    
    # Load audio features (try multiple sources)
    audio_sources = [
        ('Z_cvae.npy', 'CVAE'),
        ('Z_betavae.npy', 'BetaVAE'),
        ('Z_vae.npy', 'VAE'),
        ('Z_conv.npy', 'ConvVAE'),
    ]
    
    for filename, name in audio_sources:
        path = os.path.join('data', 'processed', filename)
        if os.path.exists(path):
            audio_features = np.load(path)
            print(f"Loaded audio features from {name}: {audio_features.shape}")
            break
    
    if audio_features is None:
        raise FileNotFoundError("No audio features found. Train a VAE model first.")
    
    # Load lyrics features
    lyrics_path = os.path.join('data', 'processed', 'lyrics_embeddings.npy')
    if os.path.exists(lyrics_path):
        lyrics_features = np.load(lyrics_path)
        print(f"Loaded lyrics features: {lyrics_features.shape}")
    else:
        print("Warning: Lyrics features not found. Using audio-only features.")
    
    # Load genre labels
    y_path = os.path.join('data', 'processed', 'y.npy')
    if os.path.exists(y_path):
        genre_labels = np.load(y_path)
        print(f"Loaded genre labels: {genre_labels.shape}")
    
    # Create multi-modal features
    if args.fusion_method == 'concat':
        if lyrics_features is not None:
            if genre_labels is not None and args.include_genre:
                genre_features = create_genre_features(genre_labels)
                multimodal_features = concatenate_features(
                    audio_features, lyrics_features, genre_features, normalize=args.normalize
                )
            else:
                multimodal_features = concatenate_features(
                    audio_features, lyrics_features, None, normalize=args.normalize
                )
        else:
            multimodal_features = audio_features
    elif args.fusion_method == 'weighted':
        if lyrics_features is not None:
            if genre_labels is not None and args.include_genre:
                genre_features = create_genre_features(genre_labels)
                multimodal_features = weighted_fusion(
                    audio_features, lyrics_features, genre_features,
                    args.audio_weight, args.lyrics_weight, args.genre_weight,
                    normalize=args.normalize
                )
            else:
                # Renormalize weights
                total = args.audio_weight + args.lyrics_weight
                multimodal_features = weighted_fusion(
                    audio_features, lyrics_features, None,
                    args.audio_weight / total, args.lyrics_weight / total, 0.0,
                    normalize=args.normalize
                )
        else:
            multimodal_features = audio_features
    else:
        raise ValueError(f"Unknown fusion method: {args.fusion_method}")
    
    print(f"Multi-modal features shape: {multimodal_features.shape}")
    
    # Perform clustering
    if args.clustering_method == 'kmeans':
        clusterer = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    elif args.clustering_method == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=args.n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {args.clustering_method}")
    
    cluster_labels = clusterer.fit_predict(multimodal_features)
    
    # Save results
    output_path = os.path.join('data', 'processed', 'multimodal_clusters.npy')
    np.save(output_path, cluster_labels)
    print(f"Saved cluster labels to {output_path}")
    
    # Save multimodal features
    features_path = os.path.join('data', 'processed', 'multimodal_features.npy')
    np.save(features_path, multimodal_features)
    print(f"Saved multimodal features to {features_path}")
    
    return multimodal_features, cluster_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-modal clustering')
    parser.add_argument('--fusion-method', type=str, default='concat',
                        choices=['concat', 'weighted'],
                        help='Feature fusion method')
    parser.add_argument('--clustering-method', type=str, default='kmeans',
                        choices=['kmeans', 'agglomerative'],
                        help='Clustering algorithm')
    parser.add_argument('--n-clusters', type=int, default=10,
                        help='Number of clusters')
    parser.add_argument('--include-genre', action='store_true',
                        help='Include genre one-hot encoding in features')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize features before fusion')
    parser.add_argument('--audio-weight', type=float, default=0.5,
                        help='Weight for audio features (weighted fusion)')
    parser.add_argument('--lyrics-weight', type=float, default=0.3,
                        help='Weight for lyrics features (weighted fusion)')
    parser.add_argument('--genre-weight', type=float, default=0.2,
                        help='Weight for genre features (weighted fusion)')
    args = parser.parse_args()
    
    main(args)

