import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import json

def plot_tsne(X, title, filename):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=20, cmap="tab10", alpha=0.8)

    # attempt to map clusters to genre names if labels and genre map exist
    legend_title = "Clusters"
    handles = None
    try:
        y = np.load("data/processed/y.npy")
        with open("data/processed/genre_map.json", "r") as f:
            genre_map = json.load(f)
        # invert genre_map (genre -> id) to id -> genre
        id_to_genre = {v: k for k, v in genre_map.items()}

        n_clusters = len(np.unique(labels))
        from matplotlib.lines import Line2D
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_clusters)]
        handles = []
        for i in range(n_clusters):
            idx = np.where(labels == i)[0]
            if len(idx) == 0:
                label_text = f"Cluster {i}"
            else:
                # find majority genre id among points in this cluster
                cluster_genres = y[idx]
                counts = np.bincount(cluster_genres)
                maj_id = int(np.argmax(counts))
                maj_name = id_to_genre.get(maj_id, str(maj_id))
                perc = int(counts[maj_id] / len(cluster_genres) * 100)
                label_text = f"{maj_name} ({perc}%)"
            handles.append(Line2D([0], [0], marker='o', color='w', label=label_text,
                                  markerfacecolor=colors[i], markersize=8))
        legend_title = "Cluster → Genre"
    except Exception:
        # fallback: numeric cluster labels
        cmap = plt.get_cmap("tab10")
        n_clusters = len(np.unique(labels))
        from matplotlib.lines import Line2D
        colors = [cmap(i) for i in range(n_clusters)]
        handles = [Line2D([0], [0], marker='o', color='w', label=f"Cluster {i}",
                          markerfacecolor=colors[i], markersize=8) for i in range(n_clusters)]

    plt.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    figures_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Easy Task visualizations
    X = np.load(os.path.join(project_root, "data", "processed", "X.npy"))
    Z_vae = np.load(os.path.join(project_root, "data", "processed", "Z_vae.npy"))
    
    plot_tsne(X, "PCA / Raw Feature Clusters", os.path.join(figures_dir, "easy_task_pca_tsne.png"))
    plot_tsne(Z_vae, "VAE Latent Clusters", os.path.join(figures_dir, "easy_task_vae_tsne.png"))
    
    # Medium Task visualizations
    # ConvVAE latents
    try:
        Z_conv = np.load(os.path.join(project_root, "data", "processed", "Z_conv.npy"))
        plot_tsne(Z_conv, "ConvVAE Latent Clusters", os.path.join(figures_dir, "medium_task_conv_vae_tsne.png"))
    except FileNotFoundError:
        print("Warning: Z_conv.npy not found. Skipping ConvVAE visualization.")
    
    # Hybrid features (VAE + Lyrics)
    try:
        lyrics = np.load(os.path.join(project_root, "data", "processed", "lyrics_embeddings.npy"))
        # Ensure dimensions match
        min_samples = min(len(Z_vae), len(lyrics))
        if len(Z_vae) != len(lyrics):
            print(f"Warning: Z_vae has {len(Z_vae)} samples, lyrics has {len(lyrics)}. Using {min_samples} samples.")
        Z_vae_lyrics = np.concatenate([Z_vae[:min_samples], lyrics[:min_samples]], axis=1)
        plot_tsne(Z_vae_lyrics, "VAE + Lyrics Hybrid Clusters", os.path.join(figures_dir, "medium_task_vae_lyrics_tsne.png"))
        
        # ConvVAE + Lyrics
        if 'Z_conv' in locals():
            min_samples_conv = min(len(Z_conv), len(lyrics))
            if len(Z_conv) != len(lyrics):
                print(f"Warning: Z_conv has {len(Z_conv)} samples, lyrics has {len(lyrics)}. Using {min_samples_conv} samples.")
            Z_conv_lyrics = np.concatenate([Z_conv[:min_samples_conv], lyrics[:min_samples_conv]], axis=1)
            plot_tsne(Z_conv_lyrics, "ConvVAE + Lyrics Hybrid Clusters", os.path.join(figures_dir, "medium_task_conv_vae_lyrics_tsne.png"))
    except FileNotFoundError:
        print("Warning: lyrics_embeddings.npy not found. Skipping hybrid feature visualizations.")
