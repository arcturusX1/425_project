"""
Comprehensive evaluation script for Hard Task.
Computes: Silhouette Score, Normalized Mutual Information (NMI),
Adjusted Rand Index (ARI), and Cluster Purity.
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix
)


def cluster_purity(y_true, y_pred):
    """
    Compute cluster purity.
    
    Purity = (1/N) * sum(max_j |C_i ∩ L_j|)
    where C_i is cluster i and L_j is true class j.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted cluster labels [N]
    
    Returns:
        Purity score [0, 1] (higher is better)
    """
    cm = confusion_matrix(y_true, y_pred)
    # For each cluster, find the majority class
    cluster_sizes = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    cluster_sizes = np.where(cluster_sizes == 0, 1, cluster_sizes)
    # Purity for each cluster
    cluster_purities = cm.max(axis=1) / cluster_sizes.flatten()
    # Weighted average by cluster size
    weights = cm.sum(axis=1)
    weights = weights / weights.sum()
    purity = np.sum(cluster_purities * weights)
    return purity


def evaluate_clustering(X, y_pred, y_true=None):
    """
    Evaluate clustering performance using multiple metrics.
    
    Args:
        X: Feature matrix [N, D]
        y_pred: Predicted cluster labels [N]
        y_true: True labels [N] (optional, for supervised metrics)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Silhouette Score (internal metric)
    try:
        n_clusters = len(np.unique(y_pred))
        n_samples = len(y_pred)
        if n_clusters > 1 and n_clusters < n_samples:
            metrics['silhouette'] = float(silhouette_score(X, y_pred))
        else:
            metrics['silhouette'] = float('nan')
    except Exception as e:
        metrics['silhouette'] = float('nan')
    
    # Supervised metrics (require ground truth)
    if y_true is not None:
        try:
            metrics['nmi'] = float(normalized_mutual_info_score(y_true, y_pred))
        except Exception:
            metrics['nmi'] = float('nan')
        
        try:
            metrics['ari'] = float(adjusted_rand_score(y_true, y_pred))
        except Exception:
            metrics['ari'] = float('nan')
        
        try:
            metrics['purity'] = float(cluster_purity(y_true, y_pred))
        except Exception:
            metrics['purity'] = float('nan')
    else:
        metrics['nmi'] = float('nan')
        metrics['ari'] = float('nan')
        metrics['purity'] = float('nan')
    
    return metrics


def main(args):
    os.makedirs('results/metrics', exist_ok=True)
    
    # Load ground truth labels
    y_path = os.path.join('data', 'processed', 'y.npy')
    y_true = np.load(y_path) if os.path.exists(y_path) else None
    
    # Load cluster labels
    cluster_path = os.path.join('data', 'processed', 'multimodal_clusters.npy')
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(f"Cluster labels not found at {cluster_path}. Run multimodal_clustering.py first.")
    
    y_pred = np.load(cluster_path)
    
    # Load features for silhouette score
    features_path = os.path.join('data', 'processed', 'multimodal_features.npy')
    if os.path.exists(features_path):
        X = np.load(features_path)
    else:
        # Fallback: try to load from other sources
        audio_sources = [
            'Z_cvae.npy', 'Z_betavae.npy', 'Z_vae.npy', 'Z_conv.npy'
        ]
        X = None
        for source in audio_sources:
            path = os.path.join('data', 'processed', source)
            if os.path.exists(path):
                X = np.load(path)
                break
        if X is None:
            raise FileNotFoundError("No feature matrix found for silhouette score computation.")
    
    # Evaluate
    metrics = evaluate_clustering(X, y_pred, y_true)
    
    # Create results dataframe
    results = {
        'method': args.method_name if args.method_name else 'multimodal_clustering',
        'n_samples': len(y_pred),
        'n_clusters': len(np.unique(y_pred)),
        **metrics
    }
    
    df = pd.DataFrame([results])
    
    # Save results
    output_path = os.path.join('results', 'metrics', 'hard_task_metrics.csv')
    
    # Append to existing file or create new
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if not np.isnan(value) else f"  {key}: N/A")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Hard Task clustering')
    parser.add_argument('--method-name', type=str, default=None,
                        help='Name of the method for results table')
    args = parser.parse_args()
    
    main(args)

