import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_clustering(X, labels):
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels)
    }

if __name__ == "__main__":
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    X_path = os.path.join(project_root, "data", "processed", "X.npy")
    X = np.load(X_path)
    
    # attempt both common filename variants
    z_paths = [
        os.path.join(project_root, "data", "processed", "Z_vae.npy"),
        os.path.join(project_root, "data", "processed", "z_vae.npy"),
    ]
    Z = None
    for p in z_paths:
        if os.path.exists(p):
            Z = np.load(p)
            break
    if Z is None:
        raise FileNotFoundError(
            "No VAE latent file found. Run src/train_vae.py to create data/processed/Z_vae.npy"
        )

    results = []

    # PCA baseline
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    kmeans_pca = KMeans(n_clusters=10, random_state=42)
    labels_pca = kmeans_pca.fit_predict(X_pca)

    metrics_pca = evaluate_clustering(X_pca, labels_pca)
    results.append({
        "method": "PCA + KMeans",
        **metrics_pca
    })

    # VAE latent clustering
    kmeans_vae = KMeans(n_clusters=10, random_state=42)
    labels_vae = kmeans_vae.fit_predict(Z)

    metrics_vae = evaluate_clustering(Z, labels_vae)
    results.append({
        "method": "VAE + KMeans",
        **metrics_vae
    })

    df = pd.DataFrame(results)
    metrics_dir = os.path.join(project_root, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    df.to_csv(os.path.join(metrics_dir, "easy_task_metrics.csv"), index=False)

    print(df)
