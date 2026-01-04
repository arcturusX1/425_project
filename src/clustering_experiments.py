import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score


def evaluate(X, labels, y_true=None):
    metrics = {}
    # silhouette requires >1 cluster and less than n_samples
    try:
        metrics['silhouette'] = float(silhouette_score(X, labels))
    except Exception:
        metrics['silhouette'] = float('nan')
    try:
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
    except Exception:
        metrics['calinski_harabasz'] = float('nan')
    try:
        metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
    except Exception:
        metrics['davies_bouldin'] = float('nan')
    if y_true is not None:
        try:
            metrics['adjusted_rand'] = float(adjusted_rand_score(y_true, labels))
        except Exception:
            metrics['adjusted_rand'] = float('nan')
    else:
        metrics['adjusted_rand'] = float('nan')
    return metrics


def run_all(args):
    os.makedirs('results/metrics', exist_ok=True)

    X_path = os.path.join('data', 'processed', 'X.npy')
    X = np.load(X_path) if os.path.exists(X_path) else None
    X_spec = np.load(os.path.join('data', 'processed', 'X_spec.npy')) if os.path.exists(os.path.join('data', 'processed', 'X_spec.npy')) else None
    Z_vae = np.load(os.path.join('data', 'processed', 'Z_vae.npy')) if os.path.exists(os.path.join('data', 'processed', 'Z_vae.npy')) else None
    Z_conv = np.load(os.path.join('data', 'processed', 'Z_conv.npy')) if os.path.exists(os.path.join('data', 'processed', 'Z_conv.npy')) else None
    lyrics = np.load(os.path.join('data', 'processed', 'lyrics_embeddings.npy')) if os.path.exists(os.path.join('data', 'processed', 'lyrics_embeddings.npy')) else None
    y = np.load(os.path.join('data', 'processed', 'y.npy')) if os.path.exists(os.path.join('data', 'processed', 'y.npy')) else None

    datasets = []
    if X is not None:
        datasets.append(('RawFeatures_PCA', PCA(n_components=10).fit_transform(X)))
    if Z_vae is not None:
        datasets.append(('VAE_Latents', Z_vae))
    if Z_conv is not None:
        datasets.append(('ConvVAE_Latents', Z_conv))
    if Z_vae is not None and lyrics is not None:
        # Ensure dimensions match
        min_samples = min(len(Z_vae), len(lyrics))
        if len(Z_vae) != len(lyrics):
            print(f"Warning: Z_vae has {len(Z_vae)} samples, lyrics has {len(lyrics)}. Trimming to {min_samples}.")
        datasets.append(('VAE+Lyrics', np.concatenate([Z_vae[:min_samples], lyrics[:min_samples]], axis=1)))
    if Z_conv is not None and lyrics is not None:
        # Ensure dimensions match
        min_samples = min(len(Z_conv), len(lyrics))
        if len(Z_conv) != len(lyrics):
            print(f"Warning: Z_conv has {len(Z_conv)} samples, lyrics has {len(lyrics)}. Trimming to {min_samples}.")
        datasets.append(('ConvVAE+Lyrics', np.concatenate([Z_conv[:min_samples], lyrics[:min_samples]], axis=1)))

    clustering_algs = {
        'KMeans': lambda X: KMeans(n_clusters=args.n_clusters, random_state=42).fit_predict(X),
        'Agglomerative': lambda X: AgglomerativeClustering(n_clusters=args.n_clusters).fit_predict(X),
        'DBSCAN': lambda X: DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples).fit_predict(X)
    }

    results = []
    for name, data in datasets:
        for alg_name, alg in clustering_algs.items():
            labels = alg(data)
            metrics = evaluate(data, labels, y_true=y)
            results.append({
                'dataset': name,
                'algorithm': alg_name,
                'n_samples': data.shape[0],
                'n_features': data.shape[1],
                **metrics
            })

    df = pd.DataFrame(results)
    out = os.path.join('results', 'metrics', 'medium_task_metrics.csv')
    df.to_csv(out, index=False)
    print('Saved metrics to', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--dbscan-eps', type=float, default=0.5)
    parser.add_argument('--dbscan-min-samples', type=int, default=5)
    args = parser.parse_args()
    run_all(args)
