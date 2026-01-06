"""
Comprehensive visualization script for Hard Task.
Generates: latent space plots, cluster distribution over genres,
and reconstruction examples from VAE latent space.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import torch
from cvae import CVAE


def plot_latent_space(Z, y_true, y_pred, title, filename, method='tsne'):
    """
    Plot 2D visualization of latent space with true and predicted labels.
    
    Args:
        Z: Latent representations [N, D]
        y_true: True genre labels [N]
        y_pred: Predicted cluster labels [N]
        title: Plot title
        filename: Output filename
        method: 'tsne' or 'pca'
    """
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        Z_2d = reducer.fit_transform(Z)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
        Z_2d = reducer.fit_transform(Z)
    
    # Load genre map
    genre_map_path = os.path.join('data', 'processed', 'genre_map.json')
    if os.path.exists(genre_map_path):
        with open(genre_map_path, 'r') as f:
            genre_map = json.load(f)
        id_to_genre = {v: k for k, v in genre_map.items()}
    else:
        id_to_genre = {i: f"Genre {i}" for i in range(len(np.unique(y_true)))}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: True genre labels
    unique_genres = np.unique(y_true)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
    for i, genre_id in enumerate(unique_genres):
        mask = y_true == genre_id
        genre_name = id_to_genre.get(int(genre_id), f"Genre {genre_id}")
        ax1.scatter(Z_2d[mask, 0], Z_2d[mask, 1], 
                   c=[colors[i]], label=genre_name, alpha=0.6, s=20)
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_title(f'{title} - True Genre Labels')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted cluster labels
    unique_clusters = np.unique(y_pred)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    for i, cluster_id in enumerate(unique_clusters):
        mask = y_pred == cluster_id
        # Find majority genre in this cluster
        cluster_genres = y_true[mask]
        if len(cluster_genres) > 0:
            counts = np.bincount(cluster_genres)
            maj_genre_id = int(np.argmax(counts))
            maj_genre_name = id_to_genre.get(maj_genre_id, f"Genre {maj_genre_id}")
            perc = int(counts[maj_genre_id] / len(cluster_genres) * 100)
            label = f"Cluster {cluster_id} ({maj_genre_name} {perc}%)"
        else:
            label = f"Cluster {cluster_id}"
        ax2.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                   c=[cluster_colors[i]], label=label, alpha=0.6, s=20)
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.set_title(f'{title} - Predicted Clusters')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved latent space plot to {filename}")


def plot_cluster_distribution(y_true, y_pred, title, filename):
    """
    Plot cluster distribution over genres.
    
    Args:
        y_true: True genre labels [N]
        y_pred: Predicted cluster labels [N]
        title: Plot title
        filename: Output filename
    """
    # Load genre map
    genre_map_path = os.path.join('data', 'processed', 'genre_map.json')
    if os.path.exists(genre_map_path):
        with open(genre_map_path, 'r') as f:
            genre_map = json.load(f)
        id_to_genre = {v: k for k, v in genre_map.items()}
    else:
        id_to_genre = {i: f"Genre {i}" for i in range(len(np.unique(y_true)))}
    
    # Create confusion matrix
    n_genres = len(np.unique(y_true))
    n_clusters = len(np.unique(y_pred))
    cm = np.zeros((n_clusters, n_genres))
    
    for i in range(len(y_pred)):
        cm[y_pred[i], y_true[i]] += 1
    
    # Normalize by cluster size (percentage)
    cluster_sizes = cm.sum(axis=1, keepdims=True)
    cluster_sizes = np.where(cluster_sizes == 0, 1, cluster_sizes)
    cm_normalized = cm / cluster_sizes
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
    
    # Set labels
    genre_names = [id_to_genre.get(i, f"Genre {i}") for i in range(n_genres)]
    cluster_labels = [f"Cluster {i}" for i in range(n_clusters)]
    
    ax.set_xticks(np.arange(n_genres))
    ax.set_yticks(np.arange(n_clusters))
    ax.set_xticklabels(genre_names, rotation=45, ha='right')
    ax.set_yticklabels(cluster_labels)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(n_clusters):
        for j in range(n_genres):
            text = ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if cm_normalized[i, j] > thresh else "black")
    
    ax.set_xlabel('True Genre')
    ax.set_ylabel('Predicted Cluster')
    ax.set_title(f'{title} - Cluster Distribution over Genres')
    
    plt.colorbar(im, ax=ax, label='Percentage')
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster distribution plot to {filename}")


def plot_reconstructions(model, X, y, n_examples=8, filename='results/figures/hard_task_reconstructions.png'):
    """
    Plot original and reconstructed examples from VAE.
    
    Args:
        model: Trained VAE model
        X: Input features [N, D]
        y: Genre labels (for CVAE) or None
        n_examples: Number of examples to show
        filename: Output filename
    """
    import torch
    
    model.eval()
    device = next(model.parameters()).device
    
    # Select random examples
    indices = np.random.choice(len(X), n_examples, replace=False)
    X_samples = X[indices]
    X_samples_torch = torch.from_numpy(X_samples).float().to(device)
    
    with torch.no_grad():
        if y is not None:
            # CVAE
            y_samples = torch.from_numpy(y[indices]).long().to(device)
            recon, _, _ = model(X_samples_torch, y_samples)
        else:
            # Standard VAE or Beta-VAE
            recon, _, _ = model(X_samples_torch)
    
    recon = recon.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, n_examples, figsize=(2*n_examples, 4))
    
    for i in range(n_examples):
        # Original
        axes[0, i].plot(X_samples[i])
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].set_ylabel('Feature Value')
        axes[0, i].grid(True, alpha=0.3)
        
        # Reconstruction
        axes[1, i].plot(recon[i])
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].set_xlabel('Feature Index')
        axes[1, i].set_ylabel('Feature Value')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle('Hard Task - VAE Reconstructions', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction plot to {filename}")


def main():
    os.makedirs('results/figures', exist_ok=True)
    
    # Load data
    y_true = np.load(os.path.join('data', 'processed', 'y.npy'))
    
    # Try to load cluster labels
    cluster_path = os.path.join('data', 'processed', 'multimodal_clusters.npy')
    if os.path.exists(cluster_path):
        y_pred = np.load(cluster_path)
    else:
        print("Warning: Cluster labels not found. Generating from K-Means on features.")
        from sklearn.cluster import KMeans
        features_path = os.path.join('data', 'processed', 'multimodal_features.npy')
        if os.path.exists(features_path):
            Z = np.load(features_path)
        else:
            # Fallback to CVAE features
            Z = np.load(os.path.join('data', 'processed', 'Z_cvae.npy'))
        kmeans = KMeans(n_clusters=10, random_state=42)
        y_pred = kmeans.fit_predict(Z)
    
    # Load latent representations (try multiple sources)
    Z = None
    z_sources = [
        ('Z_cvae.npy', 'CVAE'),
        ('Z_betavae.npy', 'BetaVAE'),
        ('Z_vae.npy', 'VAE'),
        ('multimodal_features.npy', 'Multimodal'),
    ]
    
    for filename, name in z_sources:
        path = os.path.join('data', 'processed', filename)
        if os.path.exists(path):
            Z = np.load(path)
            print(f"Using {name} features for visualization")
            break
    
    if Z is None:
        raise FileNotFoundError("No latent representations found.")
    
    # Generate visualizations
    plot_latent_space(Z, y_true, y_pred, 'Hard Task - Latent Space', 
                     'results/figures/hard_task_latent_space.png', method='tsne')
    
    plot_cluster_distribution(y_true, y_pred, 'Hard Task - Cluster Distribution',
                             'results/figures/hard_task_cluster_distribution.png')
    
    print("Visualization complete!")

    # Attempt to load a trained CVAE model and generate reconstructions
    # Prefer CVAE checkpoints; fall back to other VAE checkpoints if necessary.
    try:
        # Load input features for reconstruction (use original features X)
        X_path = os.path.join('data', 'processed', 'X.npy')
        if os.path.exists(X_path):
            X = np.load(X_path)
        else:
            # If original features missing, try multimodal features or Z as proxy
            fallback = os.path.join('data', 'processed', 'multimodal_features.npy')
            if os.path.exists(fallback):
                X = np.load(fallback)
            else:
                print("No suitable input features found for reconstructions. Skipping reconstructions.")
                return

        # Determine checkpoint path
        results_dir = os.path.join('results')
        ckpt_candidates = [
            os.path.join(results_dir, 'cvae_best.pth'),
            os.path.join(results_dir, 'cvae_final.pth'),
            os.path.join(results_dir, 'vae_best.pth'),
            os.path.join(results_dir, 'vae_final.pth')
        ]
        ckpt_path = None
        for p in ckpt_candidates:
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            print("No model checkpoint found in results/. Skipping reconstructions.")
            return

        # Prepare model instance matching data
        input_dim = X.shape[1]
        # try to infer latent dim from loaded Z if available
        latent_dim = None
        try:
            if 'Z' in locals() and Z is not None:
                latent_dim = Z.shape[1]
        except Exception:
            latent_dim = None
        if latent_dim is None:
            latent_dim = 10

        n_classes = len(np.unique(y_true))
        model = CVAE(input_dim=input_dim, latent_dim=latent_dim, n_classes=n_classes).to('cpu')

        # Load state dict
        state = torch.load(ckpt_path, map_location='cpu')
        try:
            model.load_state_dict(state)
        except Exception:
            # If the saved file contains a full model dict, attempt to load directly
            try:
                model = state
            except Exception:
                print(f"Failed to load model from {ckpt_path}. Skipping reconstructions.")
                return

        # Call reconstruction plotting
        try:
            plot_reconstructions(model, X, y_true, n_examples=8,
                                 filename=os.path.join('results', 'figures', 'hard_task_reconstructions.png'))
        except Exception as e:
            print(f"Error during plotting reconstructions: {e}")
    except Exception as e:
        print(f"Reconstruction step skipped due to error: {e}")


if __name__ == '__main__':
    main()

