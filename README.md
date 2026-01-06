# Music Genre Clustering with Variational Autoencoders

A comprehensive project implementing Easy, Medium, and Hard task pipelines for music genre clustering using Variational Autoencoders (VAEs), Convolutional VAEs, Conditional VAEs, and multi-modal feature fusion.

## Table of Contents

- [Overview](#overview)
- [Dependencies and Libraries](#dependencies-and-libraries)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Quick Start](#quick-start)
- [Task Descriptions](#task-descriptions)
- [Output Files](#output-files)

## Overview

This project implements three levels of complexity for music genre clustering:

1. **Easy Task**: Basic VAE with PCA baseline comparison
2. **Medium Task**: Convolutional VAE with hybrid features (audio + lyrics) and multiple clustering algorithms
3. **Hard Task**: Conditional VAE, Beta-VAE, multi-modal clustering, and comprehensive evaluation

## Dependencies and Libraries

### Core Libraries

- **NumPy** (`numpy`): Numerical computing and array operations
  - Used for: Data manipulation, array operations, saving/loading `.npy` files
  - Key usage: Feature arrays, latent representations, matrix operations

- **PyTorch** (`torch`): Deep learning framework
  - Used for: Neural network implementations (VAE, CVAE, Autoencoder)
  - Key modules: `torch.nn` (neural network layers), `torch.optim` (optimizers), `torch.utils.data` (data loaders)
  - GPU acceleration support via CUDA

- **Librosa** (`librosa`): Audio processing library
  - Used for: Audio feature extraction (MFCC, mel-spectrograms)
  - Key functions: `librosa.load()` (audio loading), `librosa.feature.mfcc()` (MFCC extraction), `librosa.feature.melspectrogram()` (spectrogram generation)

- **scikit-learn** (`sklearn`): Machine learning utilities
  - Used for: Dimensionality reduction, clustering, evaluation metrics
  - Key modules:
    - `sklearn.decomposition.PCA`: Principal Component Analysis
    - `sklearn.cluster`: KMeans, AgglomerativeClustering, DBSCAN
    - `sklearn.manifold.TSNE`: t-SNE visualization
    - `sklearn.metrics`: Silhouette Score, NMI, ARI, Davies-Bouldin Index
    - `sklearn.preprocessing.StandardScaler`: Feature normalization

- **Matplotlib** (`matplotlib`): Plotting and visualization
  - Used for: t-SNE plots, cluster distributions, reconstruction examples
  - Key module: `matplotlib.pyplot` for figure generation

- **Pandas** (`pandas`): Data manipulation and CSV handling
  - Used for: Storing and organizing evaluation metrics in CSV format
  - Key usage: DataFrames for metrics tables

- **tqdm** (`tqdm`): Progress bars
  - Used for: Visual progress indicators during data processing and training

### Dataset
- **GTZAN Genre Collection** : https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection
- Download the dataset and extract the contents under data/gtzan to continue

## Project Structure

```
425_project/
├── data/
│   ├── gtzan/
│   │   ├── genres_original/     # Original audio files (WAV format)
│   │   └── images_original/      # Pre-generated spectrogram images (not used)
│   └── processed/               # Generated feature files
│       ├── X.npy                # MFCC features
│       ├── X_spec.npy           # Mel-spectrograms
│       ├── y.npy                # Genre labels
│       ├── Z_*.npy              # Latent representations
│       └── lyrics_embeddings.npy # Lyrics embeddings
├── results/
│   ├── metrics/                 # Evaluation metrics (CSV files)
│   └── figures/                 # Visualization plots (PNG files)
├── src/                         # Source code modules
└── README.md                    # This file
```

## Module Documentation

### Data Processing Modules

#### `src/dataset.py`
**Purpose**: Extract MFCC (Mel-Frequency Cepstral Coefficients) features from audio files.

**Libraries Used**:
- `librosa`: Audio loading and MFCC feature extraction
- `numpy`: Array operations
- `sklearn.preprocessing.StandardScaler`: Feature normalization
- `tqdm`: Progress bars

**Key Functions**:
- `extract_mfcc_features()`: Processes audio files, extracts MFCC features (mean and std), normalizes them

**Outputs**: `data/processed/X.npy`, `data/processed/y.npy`, `data/processed/genre_map.json`

---

#### `src/dataset_spectrogram.py`
**Purpose**: Extract mel-spectrograms from audio files for convolutional VAE input.

**Libraries Used**:
- `librosa`: Audio loading and mel-spectrogram computation
- `numpy`: Array operations and padding/truncation
- `tqdm`: Progress bars

**Key Functions**:
- `extract_melspectrograms()`: Generates mel-spectrograms (64 mel bins × 128 time steps), converts to dB scale, pads/truncates to fixed dimensions

**Outputs**: `data/processed/X_spec.npy` (shape: N × 1 × 64 × 128)

---

#### `src/generate_lyrics_embeddings.py`
**Purpose**: Generate synthetic lyrics embeddings for hybrid feature fusion (Medium/Hard tasks).

**Libraries Used**:
- `numpy`: Array operations
- `sklearn.feature_extraction.text.TfidfVectorizer`: Text feature extraction
- `sklearn.decomposition.PCA`: Dimensionality reduction

**Key Functions**:
- `generate_genre_based_embeddings()`: Creates embeddings from genre lyrical themes using TF-IDF and random projection
- `load_from_file()`: Loads embeddings from external file
- `generate_simple_embeddings()`: Fallback random embeddings

**Outputs**: `data/processed/lyrics_embeddings.npy`

**Note**: For real lyrics data integration, use `src/integrate_lyrics.py` and `src/generate_lyrics_embeddings_real.py` instead. See `LYRICS_INTEGRATION_GUIDE.md` for details.

---

#### `src/integrate_lyrics.py`
**Purpose**: Integrate real lyrics data with GTZAN dataset by matching lyrics to audio tracks.

**Libraries Used**:
- `pandas`: CSV/DataFrame handling
- `json`: JSON file parsing
- `numpy`: Array operations

**Key Functions**:
- `load_lyrics_from_csv()`: Load lyrics from CSV file
- `load_lyrics_from_json()`: Load lyrics from JSON file
- `match_lyrics_to_gtzan()`: Match lyrics to GTZAN tracks using various strategies
- `save_lyrics_mapping()`: Save matched lyrics to JSON

**Matching Strategies**:
- `filename`: Exact filename match (e.g., 'blues.00000' → 'blues.00000.wav')
- `basename`: Match by number only (e.g., '00000' → '*.00000.wav')
- `index`: Match by position within genre

**Outputs**: `data/lyrics/lyrics_mapping.json`

---

#### `src/generate_lyrics_embeddings_real.py`
**Purpose**: Generate lyrics embeddings from real lyrics text using sentence transformers or TF-IDF.

**Libraries Used**:
- `sentence_transformers` (optional): Pre-trained text embedding models
- `sklearn.feature_extraction.text.TfidfVectorizer`: TF-IDF feature extraction
- `sklearn.decomposition.PCA`: Dimensionality reduction
- `numpy`: Array operations

**Key Functions**:
- `load_lyrics_mapping()`: Load lyrics mapping from JSON
- `get_lyrics_for_tracks()`: Get lyrics in same order as processed data
- `generate_embeddings_sentence_transformer()`: Use sentence transformers (better quality)
- `generate_embeddings_tfidf()`: Use TF-IDF (no external dependencies)
- `handle_missing_lyrics()`: Handle tracks without lyrics

**Embedding Methods**:
- `sentence_transformer`: Pre-trained models (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- `tfidf`: Traditional TF-IDF with PCA reduction

**Outputs**: `data/processed/lyrics_embeddings.npy`

---

### Model Architecture Modules

#### `src/vae.py`
**Purpose**: Standard Variational Autoencoder (VAE) implementation.

**Libraries Used**:
- `torch.nn`: Neural network layers (Linear, ReLU activation)
- `torch.nn.functional`: Activation functions

**Architecture**:
- Encoder: Input → 128 → 64 → (μ, log σ²)
- Decoder: Latent → 64 → 128 → Output
- Reparameterization trick for sampling

**Key Classes**:
- `VAE`: Dense VAE for MFCC features

---

#### `src/conv_vae.py`
**Purpose**: Convolutional VAE for spectrogram inputs (2D data).

**Libraries Used**:
- `torch.nn`: Convolutional layers (Conv2d, ConvTranspose2d)

**Architecture**:
- Encoder: 3-layer 2D convolutions with stride 2
- Decoder: 3-layer 2D transposed convolutions
- Designed for 64×128 spectrogram inputs

**Key Classes**:
- `ConvVAE`: Convolutional VAE for mel-spectrograms

---

#### `src/cvae.py`
**Purpose**: Conditional VAE (CVAE) and Beta-VAE implementations.

**Libraries Used**:
- `torch.nn`: Neural network layers, Embedding layer for genre conditioning

**Architecture**:
- **CVAE**: Conditions encoding/decoding on genre labels via embedding layer
- **BetaVAE**: Standard VAE with adjustable β parameter for disentanglement

**Key Classes**:
- `CVAE`: Genre-conditioned VAE
- `BetaVAE`: Disentangled representation learning

---

#### `src/autoencoder.py`
**Purpose**: Standard Autoencoder (non-variational) for baseline comparison.

**Libraries Used**:
- `torch.nn`: Neural network layers

**Architecture**: Similar to VAE but without probabilistic latent space

**Key Classes**:
- `Autoencoder`: Deterministic encoder-decoder

---

### Training Modules

#### `src/train_vae.py`
**Purpose**: Training script for standard VAE.

**Libraries Used**:
- `torch`: Model training, optimizers, data loaders
- `torch.optim.Adam`: Adam optimizer
- `numpy`: Data loading

**Features**:
- Train/validation split
- Early stopping with patience
- Checkpoint saving
- Latent representation encoding

**Outputs**: `results/vae_best.pth`, `data/processed/Z_vae.npy`

---

#### `src/train_conv_vae.py`
**Purpose**: Training script for Convolutional VAE.

**Libraries Used**:
- `torch`: Model training infrastructure
- Same training features as `train_vae.py`

**Outputs**: `results/conv_vae_best.pth`, `data/processed/Z_conv.npy`

---

#### `src/train_cvae.py`
**Purpose**: Training script for CVAE and BetaVAE.

**Libraries Used**:
- `torch`: Model training infrastructure
- Supports both CVAE and BetaVAE via command-line arguments

**Features**:
- Model type selection (cvae/betavae)
- Beta parameter tuning for BetaVAE
- Genre embedding dimension configuration

**Outputs**: `results/cvae_best.pth` or `results/betavae_best.pth`, `data/processed/Z_cvae.npy` or `data/processed/Z_betavae.npy`

---

#### `src/train_autoencoder.py`
**Purpose**: Training script for standard Autoencoder.

**Libraries Used**:
- `torch`: Model training infrastructure

**Outputs**: `results/autoencoder_best.pth`, `data/processed/Z_autoencoder.npy`

---

### Clustering and Evaluation Modules

#### `src/clustering.py`
**Purpose**: Easy Task clustering comparison (PCA + K-Means vs VAE + K-Means).

**Libraries Used**:
- `sklearn.decomposition.PCA`: Dimensionality reduction
- `sklearn.cluster.KMeans`: K-Means clustering
- `sklearn.metrics`: Silhouette Score, Calinski-Harabasz Index
- `pandas`: Results storage

**Outputs**: `results/metrics/easy_task_metrics.csv`

---

#### `src/clustering_experiments.py`
**Purpose**: Medium Task comprehensive clustering experiments.

**Libraries Used**:
- `sklearn.decomposition.PCA`: Feature reduction
- `sklearn.cluster`: KMeans, AgglomerativeClustering, DBSCAN
- `sklearn.metrics`: Silhouette, Davies-Bouldin, Adjusted Rand Index, Calinski-Harabasz
- `pandas`: Results storage
- `numpy`: Data manipulation

**Features**:
- Multiple feature sets: RawFeatures_PCA, VAE_Latents, ConvVAE_Latents, VAE+Lyrics, ConvVAE+Lyrics
- Multiple clustering algorithms
- Comprehensive metric computation

**Outputs**: `results/metrics/medium_task_metrics.csv`

---

#### `src/multimodal_clustering.py`
**Purpose**: Multi-modal clustering combining audio, lyrics, and genre features.

**Libraries Used**:
- `sklearn.cluster`: KMeans, AgglomerativeClustering
- `sklearn.preprocessing.StandardScaler`: Feature normalization
- `numpy`: Feature concatenation and manipulation

**Key Functions**:
- `concatenate_features()`: Simple feature concatenation
- `weighted_fusion()`: Weighted combination of modalities
- `create_genre_features()`: One-hot genre encoding

**Outputs**: `data/processed/multimodal_features.npy`, `data/processed/multimodal_clusters.npy`

---

#### `src/evaluate_hard_task.py`
**Purpose**: Comprehensive evaluation for Hard Task with all required metrics.

**Libraries Used**:
- `sklearn.metrics`: Silhouette Score, Normalized Mutual Information (NMI), Adjusted Rand Index (ARI)
- `numpy`: Array operations, confusion matrix computation
- `pandas`: Results storage

**Key Functions**:
- `cluster_purity()`: Computes cluster purity metric
- `evaluate_clustering()`: Comprehensive metric evaluation

**Metrics Computed**:
- Silhouette Score (internal metric)
- NMI (Normalized Mutual Information)
- ARI (Adjusted Rand Index)
- Cluster Purity

**Outputs**: `results/metrics/hard_task_metrics.csv`

---

#### `src/compare_hard_task.py`
**Purpose**: Compare all baseline and VAE-based methods for Hard Task.

**Libraries Used**:
- `sklearn.decomposition.PCA`: Dimensionality reduction
- `sklearn.cluster.KMeans`: Clustering
- `sklearn.metrics`: Evaluation metrics
- `pandas`: Results organization and sorting
- `numpy`: Data manipulation

**Methods Compared**:
1. PCA + K-Means (baseline)
2. Autoencoder + K-Means
3. Direct Spectral Feature Clustering
4. VAE + K-Means
5. CVAE + K-Means
6. BetaVAE + K-Means
7. Multimodal (Audio+Lyrics)

**Outputs**: `results/metrics/hard_task_comparison.csv`

---

### Visualization Modules

#### `src/visualize.py`
**Purpose**: Generate t-SNE visualizations for Easy and Medium tasks.

**Libraries Used**:
- `sklearn.manifold.TSNE`: 2D dimensionality reduction for visualization
- `sklearn.cluster.KMeans`: Cluster assignment for visualization
- `matplotlib.pyplot`: Plot generation
- `matplotlib.lines.Line2D`: Custom legend creation
- `numpy`: Data manipulation
- `json`: Genre map loading

**Key Functions**:
- `plot_tsne()`: Creates t-SNE plots with cluster coloring and genre mapping

**Outputs**: 
- `results/figures/easy_task_pca_tsne.png`
- `results/figures/easy_task_vae_tsne.png`
- `results/figures/medium_task_conv_vae_tsne.png`
- `results/figures/medium_task_vae_lyrics_tsne.png`
- `results/figures/medium_task_conv_vae_lyrics_tsne.png`

---

#### `src/visualize_hard_task.py`
**Purpose**: Comprehensive visualizations for Hard Task.

**Libraries Used**:
- `sklearn.manifold.TSNE`: Latent space visualization
- `sklearn.decomposition.PCA`: Alternative dimensionality reduction
- `matplotlib.pyplot`: Plotting
- `numpy`: Data manipulation
- `json`: Genre map loading

**Key Functions**:
- `plot_latent_space()`: 2D visualization with true labels and predicted clusters
- `plot_cluster_distribution()`: Confusion matrix showing cluster distribution over genres
- `plot_reconstructions()`: VAE reconstruction examples

**Outputs**:
- `results/figures/hard_task_latent_space.png`
- `results/figures/hard_task_cluster_distribution.png`
- `results/figures/hard_task_reconstructions.png`

---

### Driver Module

#### `src/run_all_tasks.py`
**Purpose**: Main driver script to run all tasks sequentially with clear console output.

**Libraries Used**:
- `subprocess`: Running Python scripts
- `pathlib.Path`: Path manipulation
- `os`: File system operations
- `time`: Execution time tracking

**Features**:
- Sequential execution of Easy, Medium, and Hard tasks
- Clear console output with task identification
- File existence checking to skip completed steps
- Output verification
- Final summary report

**Usage**: `python3 src/run_all_tasks.py`

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or if using virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run All Tasks

```bash
# Run complete pipeline
python3 src/run_all_tasks.py
```

### Run Individual Tasks

**Easy Task:**
```bash
# Extract features
python3 src/dataset.py

# Train VAE
python3 src/train_vae.py --epochs 100 --batch-size 32

# Run clustering
python3 src/clustering.py

# Visualize
python3 src/visualize.py
```

**Medium Task:**
```bash
# Extract spectrograms
python3 src/dataset_spectrogram.py

# Train ConvVAE
python3 src/train_conv_vae.py --epochs 100 --latent-dim 64

# Generate lyrics embeddings
python3 src/generate_lyrics_embeddings.py --embedding-dim 32

# Run experiments
python3 src/clustering_experiments.py --n-clusters 10

# Visualize
python3 src/visualize.py
```

**Hard Task:**
```bash
# Train models
python3 src/train_cvae.py --model-type cvae --epochs 100
python3 src/train_cvae.py --model-type betavae --epochs 100 --beta 4.0
python3 src/train_autoencoder.py --epochs 100

# Multi-modal clustering
python3 src/multimodal_clustering.py --n-clusters 10

# Evaluate
python3 src/evaluate_hard_task.py

# Compare all methods
python3 src/compare_hard_task.py --n-clusters 10

# Visualize
python3 src/visualize_hard_task.py
```

Note: `visualize_hard_task.py` attempts to load a trained CVAE (or other VAE) checkpoint
from `results/` and will generate reconstruction example images in addition to latent-space
and cluster-distribution plots when a compatible checkpoint (e.g. `cvae_best.pth` or
`cvae_final.pth`) is available.

## Task Descriptions

### Easy Task
- **Objective**: Compare PCA + K-Means baseline with VAE + K-Means
- **Metrics**: Silhouette Score, Calinski-Harabasz Index
- **Outputs**: `easy_task_metrics.csv`, `easy_task_*.png`

### Medium Task
- **Objective**: Advanced VAE architectures with hybrid features and multiple clustering algorithms
- **Features**: ConvVAE, hybrid audio+lyrics features
- **Algorithms**: K-Means, Agglomerative Clustering, DBSCAN
- **Metrics**: Silhouette, Davies-Bouldin, Adjusted Rand Index
- **Outputs**: `medium_task_metrics.csv`, `medium_task_*.png`

### Hard Task
- **Objective**: Conditional VAE, Beta-VAE, multi-modal clustering, comprehensive evaluation
- **Models**: CVAE, BetaVAE, Autoencoder
- **Clustering**: Multi-modal fusion (audio + lyrics + genre)
- **Metrics**: Silhouette, NMI, ARI, Cluster Purity
- **Outputs**: `hard_task_metrics.csv`, `hard_task_comparison.csv`, `hard_task_*.png`

## Output Files

All outputs are organized by task with clear naming:

### Metrics (`results/metrics/`)
- `easy_task_metrics.csv` - Easy Task comparison
- `medium_task_metrics.csv` - Medium Task comprehensive results
- `hard_task_metrics.csv` - Hard Task individual evaluations
- `hard_task_comparison.csv` - Hard Task method comparison

### Figures (`results/figures/`)
- `easy_task_*.png` - Easy Task visualizations
- `medium_task_*.png` - Medium Task visualizations
- `hard_task_*.png` - Hard Task visualizations
 - `hard_task_reconstructions.png` - Example original vs reconstructed feature plots (generated when a model checkpoint is available)

### Model Checkpoints (`results/`)
- `vae_best.pth`, `conv_vae_best.pth`, `cvae_best.pth`, `betavae_best.pth`, `autoencoder_best.pth`

Note: `src/visualize_hard_task.py` prefers `cvae_best.pth` / `cvae_final.pth` and will fall back
to other VAE checkpoints when attempting to create reconstructions.

### Latent Representations (`data/processed/`)
- `Z_vae.npy`, `Z_conv.npy`, `Z_cvae.npy`, `Z_betavae.npy`, `Z_autoencoder.npy`

## Important Notes and Known Issues

### Data Leakage Fix (v2.0)

**Issue**: Previous versions of `generate_lyrics_embeddings.py` used ground truth labels to generate embeddings, causing data leakage and unrealistically high multimodal clustering performance.

**Fix**: The lyrics embedding generation has been updated to generate embeddings based on file order, not label order. This eliminates data leakage but may result in lower (more realistic) multimodal clustering performance.

**Impact**: 
- Previous multimodal results (NMI ~0.99, ARI ~0.99) were artificially high due to label leakage
- New results should show more realistic performance (NMI ~0.4-0.6, ARI ~0.3-0.5)
- To regenerate results, delete `data/processed/lyrics_embeddings.npy` and re-run the generation script

### CVAE Performance Analysis

CVAE may show high silhouette scores but low label alignment (NMI, ARI). This suggests the model learns well-separated clusters that don't correspond to genre boundaries. See `ANALYSIS_ISSUES.md` for detailed analysis.

## License

This project is for educational/research purposes.

