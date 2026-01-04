"""
Driver script to run Easy, Medium, and Hard tasks sequentially.
Provides clear console output identifying which task is running and what outputs are generated.
"""
import os
import sys
import subprocess
import time
from pathlib import Path


# Get project root directory
script_dir = Path(__file__).parent
project_root = script_dir.parent


def print_header(title, char="=", width=80):
    """Print a formatted header."""
    print("\n" + char * width)
    print(f"{title:^{width}}")
    print(char * width + "\n")


def print_section(title, char="-", width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(char * width + "\n")


def run_command(cmd, description, task_name):
    """Run a command and handle output."""
    print_section(f"[{task_name}] {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ Successfully completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}")
        print(f"Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found: {cmd[0]}")
        print("Make sure Python 3 is installed and scripts are in the correct location.")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    full_path = project_root / filepath
    if full_path.exists():
        size = full_path.stat().st_size
        print(f"  ✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {description}: {filepath} (NOT FOUND)")
        return False


def run_easy_task():
    """Run Easy Task pipeline."""
    print_header("EASY TASK", "=")
    
    outputs = []
    success = True
    
    # Step 1: Extract features
    if not (project_root / "data/processed/X.npy").exists():
        print_section("Step 1: Extracting MFCC Features")
        success = run_command(
            ["python3", "src/dataset.py"],
            "Extract MFCC features from audio files",
            "EASY"
        )
        if not success:
            return False
    else:
        print("Step 1: MFCC features already exist, skipping...")
    
    # Step 2: Train VAE
    if not (project_root / "data/processed/Z_vae.npy").exists():
        print_section("Step 2: Training VAE")
        success = run_command(
            ["python3", "src/train_vae.py", "--epochs", "100", "--batch-size", "32", "--val-split", "0.1", "--patience", "10"],
            "Train Variational Autoencoder",
            "EASY"
        )
        if not success:
            return False
    else:
        print("Step 2: VAE already trained, skipping...")
    
    # Step 3: Run clustering experiments
    print_section("Step 3: Running Clustering Experiments")
    success = run_command(
        ["python3", "src/clustering.py"],
        "Compare PCA + K-Means vs VAE + K-Means",
        "EASY"
    )
    if not success:
        return False
    
    # Step 4: Generate visualizations
    print_section("Step 4: Generating Visualizations")
    success = run_command(
        ["python3", "src/visualize.py"],
        "Generate t-SNE visualizations",
        "EASY"
    )
    if not success:
        return False
    
    # Check outputs
    print_section("Easy Task Outputs")
    outputs.append(check_file_exists("results/metrics/easy_task_metrics.csv", "Metrics"))
    outputs.append(check_file_exists("results/figures/easy_task_pca_tsne.png", "PCA t-SNE plot"))
    outputs.append(check_file_exists("results/figures/easy_task_vae_tsne.png", "VAE t-SNE plot"))
    
    if all(outputs):
        print("\n✓ Easy Task completed successfully!")
        return True
    else:
        print("\n✗ Some Easy Task outputs are missing.")
        return False


def run_medium_task():
    """Run Medium Task pipeline."""
    print_header("MEDIUM TASK", "=")
    
    outputs = []
    success = True
    
    # Step 1: Extract spectrograms
    if not (project_root / "data/processed/X_spec.npy").exists():
        print_section("Step 1: Extracting Mel-Spectrograms")
        success = run_command(
            ["python3", "src/dataset_spectrogram.py"],
            "Extract mel-spectrograms from audio files",
            "MEDIUM"
        )
        if not success:
            return False
    else:
        print("Step 1: Spectrograms already exist, skipping...")
    
    # Step 2: Train ConvVAE
    if not (project_root / "data/processed/Z_conv.npy").exists():
        print_section("Step 2: Training Convolutional VAE")
        success = run_command(
            ["python3", "src/train_conv_vae.py", "--epochs", "100", "--batch-size", "32", "--latent-dim", "64", "--val-split", "0.1", "--patience", "10"],
            "Train Convolutional VAE on spectrograms",
            "MEDIUM"
        )
        if not success:
            return False
    else:
        print("Step 2: ConvVAE already trained, skipping...")
    
    # Step 3: Generate lyrics embeddings
    if not (project_root / "data/processed/lyrics_embeddings.npy").exists():
        print_section("Step 3: Generating Lyrics Embeddings")
        success = run_command(
            ["python3", "src/generate_lyrics_embeddings.py", "--embedding-dim", "32"],
            "Generate lyrics embeddings for hybrid features",
            "MEDIUM"
        )
        if not success:
            return False
    else:
        print("Step 3: Lyrics embeddings already exist, skipping...")
    
    # Step 4: Run clustering experiments
    print_section("Step 4: Running Clustering Experiments")
    success = run_command(
        ["python3", "src/clustering_experiments.py", "--n-clusters", "10"],
        "Run clustering with multiple algorithms and feature sets",
        "MEDIUM"
    )
    if not success:
        return False
    
    # Step 5: Generate visualizations
    print_section("Step 5: Generating Visualizations")
    success = run_command(
        ["python3", "src/visualize.py"],
        "Generate t-SNE visualizations for all feature sets",
        "MEDIUM"
    )
    if not success:
        return False
    
    # Check outputs
    print_section("Medium Task Outputs")
    outputs.append(check_file_exists("results/metrics/medium_task_metrics.csv", "Metrics"))
    outputs.append(check_file_exists("results/figures/medium_task_conv_vae_tsne.png", "ConvVAE t-SNE plot"))
    outputs.append(check_file_exists("results/figures/medium_task_vae_lyrics_tsne.png", "VAE+Lyrics t-SNE plot"))
    outputs.append(check_file_exists("results/figures/medium_task_conv_vae_lyrics_tsne.png", "ConvVAE+Lyrics t-SNE plot"))
    
    if all(outputs):
        print("\n✓ Medium Task completed successfully!")
        return True
    else:
        print("\n✗ Some Medium Task outputs are missing.")
        return False


def run_hard_task():
    """Run Hard Task pipeline."""
    print_header("HARD TASK", "=")
    
    outputs = []
    success = True
    
    # Step 1: Train CVAE
    if not (project_root / "data/processed/Z_cvae.npy").exists():
        print_section("Step 1: Training Conditional VAE (CVAE)")
        success = run_command(
            ["python3", "src/train_cvae.py", "--model-type", "cvae", "--epochs", "100", "--batch-size", "32", "--latent-dim", "10"],
            "Train Conditional VAE with genre conditioning",
            "HARD"
        )
        if not success:
            return False
    else:
        print("Step 1: CVAE already trained, skipping...")
    
    # Step 2: Train BetaVAE
    if not (project_root / "data/processed/Z_betavae.npy").exists():
        print_section("Step 2: Training Beta-VAE")
        success = run_command(
            ["python3", "src/train_cvae.py", "--model-type", "betavae", "--epochs", "100", "--batch-size", "32", "--latent-dim", "10", "--beta", "4.0"],
            "Train Beta-VAE for disentangled representations",
            "HARD"
        )
        if not success:
            return False
    else:
        print("Step 2: BetaVAE already trained, skipping...")
    
    # Step 3: Train Autoencoder (for comparison)
    if not (project_root / "data/processed/Z_autoencoder.npy").exists():
        print_section("Step 3: Training Autoencoder (Baseline)")
        success = run_command(
            ["python3", "src/train_autoencoder.py", "--epochs", "100", "--batch-size", "32", "--latent-dim", "10"],
            "Train standard Autoencoder for comparison",
            "HARD"
        )
        if not success:
            return False
    else:
        print("Step 3: Autoencoder already trained, skipping...")
    
    # Step 4: Multi-modal clustering
    if not (project_root / "data/processed/multimodal_clusters.npy").exists():
        print_section("Step 4: Multi-modal Clustering")
        success = run_command(
            ["python3", "src/multimodal_clustering.py", "--fusion-method", "concat", "--clustering-method", "kmeans", "--n-clusters", "10"],
            "Perform multi-modal clustering (audio + lyrics + genre)",
            "HARD"
        )
        if not success:
            return False
    else:
        print("Step 4: Multi-modal clustering already completed, skipping...")
    
    # Step 5: Evaluate methods
    print_section("Step 5: Evaluating Hard Task Methods")
    success = run_command(
        ["python3", "src/evaluate_hard_task.py", "--method-name", "Multimodal_Concat"],
        "Evaluate clustering with NMI, ARI, Purity, Silhouette",
        "HARD"
    )
    if not success:
        return False
    
    # Step 6: Compare all methods
    print_section("Step 6: Comparing All Methods")
    success = run_command(
        ["python3", "src/compare_hard_task.py", "--n-clusters", "10"],
        "Compare all baseline and VAE-based methods",
        "HARD"
    )
    if not success:
        return False
    
    # Step 7: Generate visualizations
    print_section("Step 7: Generating Visualizations")
    success = run_command(
        ["python3", "src/visualize_hard_task.py"],
        "Generate latent space plots, cluster distributions, and reconstructions",
        "HARD"
    )
    if not success:
        return False
    
    # Check outputs
    print_section("Hard Task Outputs")
    outputs.append(check_file_exists("results/metrics/hard_task_metrics.csv", "Metrics"))
    outputs.append(check_file_exists("results/metrics/hard_task_comparison.csv", "Comparison results"))
    outputs.append(check_file_exists("results/figures/hard_task_latent_space.png", "Latent space plot"))
    outputs.append(check_file_exists("results/figures/hard_task_cluster_distribution.png", "Cluster distribution"))
    
    if all(outputs):
        print("\n✓ Hard Task completed successfully!")
        return True
    else:
        print("\n✗ Some Hard Task outputs are missing.")
        return False


def main():
    """Main driver function."""
    print_header("MUSIC GENRE CLUSTERING - COMPLETE PIPELINE", "=")
    print("This script will run Easy, Medium, and Hard tasks sequentially.")
    print("Each task will generate labeled outputs in results/metrics/ and results/figures/")
    print("\nPress Ctrl+C to cancel at any time.\n")
    
    start_time = time.time()
    results = {}
    
    try:
        # Run Easy Task
        results['easy'] = run_easy_task()
        time.sleep(1)  # Brief pause between tasks
        
        # Run Medium Task
        results['medium'] = run_medium_task()
        time.sleep(1)
        
        # Run Hard Task
        results['hard'] = run_hard_task()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user.")
        sys.exit(1)
    
    # Final summary
    elapsed_time = time.time() - start_time
    print_header("FINAL SUMMARY", "=")
    
    print("Task Completion Status:")
    print(f"  Easy Task:   {'✓ COMPLETED' if results.get('easy') else '✗ FAILED/INCOMPLETE'}")
    print(f"  Medium Task: {'✓ COMPLETED' if results.get('medium') else '✗ FAILED/INCOMPLETE'}")
    print(f"  Hard Task:   {'✓ COMPLETED' if results.get('hard') else '✗ FAILED/INCOMPLETE'}")
    
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    
    print("\nOutput Locations:")
    print("  Metrics:    results/metrics/")
    print("    - easy_task_metrics.csv")
    print("    - medium_task_metrics.csv")
    print("    - hard_task_metrics.csv")
    print("    - hard_task_comparison.csv")
    print("\n  Figures:    results/figures/")
    print("    - easy_task_*.png")
    print("    - medium_task_*.png")
    print("    - hard_task_*.png")
    
    if all(results.values()):
        print("\n" + "=" * 80)
        print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("⚠ SOME TASKS FAILED OR ARE INCOMPLETE")
        print("Check the output above for details.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

