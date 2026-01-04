"""
Comprehensive comparison script for Hard Task.
Compares: PCA + K-Means, Autoencoder + K-Means, Direct spectral feature clustering,
and VAE-based methods (CVAE, BetaVAE, standard VAE).
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score
)
from evaluate_hard_task import evaluate_clustering, cluster_purity


def direct_spectral_clustering(X_spec, n_clusters=10):
    """
    Direct clustering on spectral features (mel-spectrograms).
    Flattens spectrograms and applies PCA before clustering.
    
    Args:
        X_spec: Spectrograms [N, 1, n_mels, time_steps]
        n_clusters: Number of clusters
    """
    # Flatten spectrograms
    X_flat = X_spec.reshape(X_spec.shape[0], -1)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_flat)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    return labels, X_pca


def compare_all_methods(args):
    """
    Compare all baseline and VAE-based methods.
    """
    os.makedirs('results/metrics', exist_ok=True)
    
    # Load ground truth
    y_path = os.path.join('data', 'processed', 'y.npy')
    y_true = np.load(y_path) if os.path.exists(y_path) else None
    
    results = []
    
    # 1. PCA + K-Means (baseline)
    print("Evaluating PCA + K-Means...")
    X = np.load(os.path.join('data', 'processed', 'X.npy'))
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)
    kmeans_pca = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    y_pred_pca = kmeans_pca.fit_predict(X_pca)
    
    metrics_pca = evaluate_clustering(X_pca, y_pred_pca, y_true)
    results.append({
        'method': 'PCA + K-Means',
        'n_samples': len(y_pred_pca),
        'n_clusters': len(np.unique(y_pred_pca)),
        **metrics_pca
    })
    
    # 2. Autoencoder + K-Means
    print("Evaluating Autoencoder + K-Means...")
    z_ae_path = os.path.join('data', 'processed', 'Z_autoencoder.npy')
    if os.path.exists(z_ae_path):
        Z_ae = np.load(z_ae_path)
        kmeans_ae = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        y_pred_ae = kmeans_ae.fit_predict(Z_ae)
        
        metrics_ae = evaluate_clustering(Z_ae, y_pred_ae, y_true)
        results.append({
            'method': 'Autoencoder + K-Means',
            'n_samples': len(y_pred_ae),
            'n_clusters': len(np.unique(y_pred_ae)),
            **metrics_ae
        })
    else:
        print("Warning: Autoencoder features not found. Train autoencoder first.")
    
    # 3. Direct Spectral Feature Clustering
    print("Evaluating Direct Spectral Feature Clustering...")
    X_spec_path = os.path.join('data', 'processed', 'X_spec.npy')
    if os.path.exists(X_spec_path):
        X_spec = np.load(X_spec_path)
        y_pred_spec, X_spec_pca = direct_spectral_clustering(X_spec, args.n_clusters)
        
        metrics_spec = evaluate_clustering(X_spec_pca, y_pred_spec, y_true)
        results.append({
            'method': 'Direct Spectral + K-Means',
            'n_samples': len(y_pred_spec),
            'n_clusters': len(np.unique(y_pred_spec)),
            **metrics_spec
        })
    else:
        print("Warning: Spectrogram data not found. Run dataset_spectrogram.py first.")
    
    # 4. VAE + K-Means
    print("Evaluating VAE + K-Means...")
    z_vae_path = os.path.join('data', 'processed', 'Z_vae.npy')
    if os.path.exists(z_vae_path):
        Z_vae = np.load(z_vae_path)
        kmeans_vae = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        y_pred_vae = kmeans_vae.fit_predict(Z_vae)
        
        metrics_vae = evaluate_clustering(Z_vae, y_pred_vae, y_true)
        results.append({
            'method': 'VAE + K-Means',
            'n_samples': len(y_pred_vae),
            'n_clusters': len(np.unique(y_pred_vae)),
            **metrics_vae
        })
    
    # 5. CVAE + K-Means
    print("Evaluating CVAE + K-Means...")
    z_cvae_path = os.path.join('data', 'processed', 'Z_cvae.npy')
    if os.path.exists(z_cvae_path):
        Z_cvae = np.load(z_cvae_path)
        kmeans_cvae = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        y_pred_cvae = kmeans_cvae.fit_predict(Z_cvae)
        
        metrics_cvae = evaluate_clustering(Z_cvae, y_pred_cvae, y_true)
        results.append({
            'method': 'CVAE + K-Means',
            'n_samples': len(y_pred_cvae),
            'n_clusters': len(np.unique(y_pred_cvae)),
            **metrics_cvae
        })
    else:
        print("Warning: CVAE features not found. Train CVAE first.")
    
    # 6. BetaVAE + K-Means
    print("Evaluating BetaVAE + K-Means...")
    z_betavae_path = os.path.join('data', 'processed', 'Z_betavae.npy')
    if os.path.exists(z_betavae_path):
        Z_betavae = np.load(z_betavae_path)
        kmeans_betavae = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        y_pred_betavae = kmeans_betavae.fit_predict(Z_betavae)
        
        metrics_betavae = evaluate_clustering(Z_betavae, y_pred_betavae, y_true)
        results.append({
            'method': 'BetaVAE + K-Means',
            'n_samples': len(y_pred_betavae),
            'n_clusters': len(np.unique(y_pred_betavae)),
            **metrics_betavae
        })
    else:
        print("Warning: BetaVAE features not found. Train BetaVAE first.")
    
    # 7. Multimodal Clustering
    print("Evaluating Multimodal Clustering...")
    multimodal_clusters_path = os.path.join('data', 'processed', 'multimodal_clusters.npy')
    multimodal_features_path = os.path.join('data', 'processed', 'multimodal_features.npy')
    if os.path.exists(multimodal_clusters_path) and os.path.exists(multimodal_features_path):
        y_pred_multimodal = np.load(multimodal_clusters_path)
        Z_multimodal = np.load(multimodal_features_path)
        
        metrics_multimodal = evaluate_clustering(Z_multimodal, y_pred_multimodal, y_true)
        results.append({
            'method': 'Multimodal (Audio+Lyrics)',
            'n_samples': len(y_pred_multimodal),
            'n_clusters': len(np.unique(y_pred_multimodal)),
            **metrics_multimodal
        })
    else:
        print("Warning: Multimodal clustering results not found. Run multimodal_clustering.py first.")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join('results', 'metrics', 'hard_task_comparison.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("Comparison Results:")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\nSaved results to {output_path}")
    
    # Print summary
    if y_true is not None:
        print(f"\n{'='*80}")
        print("Ranking by Adjusted Rand Index (ARI):")
        print(f"{'='*80}")
        df_sorted = df.sort_values('ari', ascending=False, na_position='last')
        for i, row in df_sorted.iterrows():
            ari = row['ari']
            method = row['method']
            if not np.isnan(ari):
                print(f"{method:30s} ARI: {ari:.4f}")
        
        print(f"\n{'='*80}")
        print("Ranking by Normalized Mutual Information (NMI):")
        print(f"{'='*80}")
        df_sorted = df.sort_values('nmi', ascending=False, na_position='last')
        for i, row in df_sorted.iterrows():
            nmi = row['nmi']
            method = row['method']
            if not np.isnan(nmi):
                print(f"{method:30s} NMI: {nmi:.4f}")
        
        print(f"\n{'='*80}")
        print("Ranking by Cluster Purity:")
        print(f"{'='*80}")
        df_sorted = df.sort_values('purity', ascending=False, na_position='last')
        for i, row in df_sorted.iterrows():
            purity = row['purity']
            method = row['method']
            if not np.isnan(purity):
                print(f"{method:30s} Purity: {purity:.4f}")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare all Hard Task methods')
    parser.add_argument('--n-clusters', type=int, default=10,
                        help='Number of clusters')
    args = parser.parse_args()
    
    compare_all_methods(args)

