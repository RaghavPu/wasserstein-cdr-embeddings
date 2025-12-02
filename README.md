# Wasserstein Cross-Domain Recommendation

A modular framework for training BPR-based recommendation models and learning optimal transport mappings for cross-domain recommendation. This repository implements the embedding generation and OT-based transfer learning approach for the Books and Movies & TV domains.

**Code Repository:** https://github.com/RaghavPu/wasserstein-cdr-embeddings

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Information](#dataset-information)
3. [Generating BPR Embeddings](#generating-bpr-embeddings)
4. [Running OT Experiments](#running-ot-experiments)
5. [Additional Scripts and Utilities](#additional-scripts-and-utilities)
6. [Output Structure](#output-structure)
7. [Reproducibility Details](#reproducibility-details)
8. [Citation](#citation)

---

## Environment Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, scikit-learn
- POT (Python Optimal Transport)
- Matplotlib, seaborn (for visualization)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RaghavPu/wasserstein-cdr-embeddings.git
cd wasserstein-cdr-embeddings
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
pyyaml>=5.4.0
tqdm>=4.62.0
POT>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## Dataset Information

### Amazon Dense Subset

We use a curated **dense subset** derived from the Amazon-CrossDomain-Overlapping dataset. The dataset is optimized for fast training while maintaining real 1-5 star ratings.

**Dataset Statistics:**
- **Books:** 1,426 users, 723 items, 38,149 interactions (density: 3.70%, mean rating: 4.31)
- **Movies:** 1,381 users, 1,338 items, 68,023 interactions (density: 3.68%, mean rating: 4.01)
- **Overlapping Users:** 1,329 users appear in both domains (93%)

**Location:**
```
data/Amazon-Dense-Subset/
├── Amazon-Books-Dense.inter      # Books interactions (user_id, item_id, rating)
├── Amazon-Movies-Dense.inter     # Movies interactions (user_id, item_id, rating)
├── dense_users.txt               # List of selected users
└── README.md                     # Dataset documentation
```

**File Format:** Each `.inter` file contains tab-separated columns:
```
user_id    item_id    rating
A1234...   B5678...   5.0
A1234...   B9012...   4.0
...
```

### Dataset Creation

The dense subset was created using `create_dense_subset.py`:
1. Selected top 1,500 most active users across both domains
2. Filtered items with < 30 interactions (reduces sparsity from 99.9% to ~3.7%)
3. Result: 743× denser for Books, 149× denser for Movies

To recreate the dataset:
```bash
python create_dense_subset.py
```

This reads from `data/Amazon-CrossDomain-Overlapping/` and creates `data/Amazon-Dense-Subset/`.

---

## Generating BPR Embeddings

### Overview

We use **Bayesian Personalized Ranking (BPR)** to generate user and item embeddings for each domain. BPR optimizes for ranking quality (NDCG, Recall, Precision) rather than rating prediction.

### Configuration

All training parameters are managed via `config.yaml`. Key settings:

```yaml
data:
  books_path: "data/Amazon-Dense-Subset/Amazon-Books-Dense.inter"
  movies_path: "data/Amazon-Dense-Subset/Amazon-Movies-Dense.inter"
  test_ratio: 0.2
  val_ratio: 0.1
  random_seed: 42

model:
  name: "BPR"
  embedding_dim: 64          # Embedding dimension (d=64)
  learning_rate: 0.01
  weight_decay: 0.0001

bpr:
  n_negatives: 8             # Negative samples per positive
  min_item_interactions: 10  # Item filtering for evaluation
  negative_sampling: "popularity"

training:
  num_epochs: 20
  batch_size: 512
  device: "cpu"              # or "cuda"
```

### Training Books Domain

```bash
python src/train_bpr.py --domain books
```

**Expected Output:**
```
Books Domain BPR Training Results:
  NDCG@10:      0.1574
  Recall@10:    0.1876
  Precision@10: 0.0917
  Training time: ~1-2 minutes (20 epochs)
```

**Saved Files:**
```
outputs/books_bpr_YYYYMMDD_HHMMSS/
├── embeddings.npz         # User & item embeddings + mappings
├── model.pt               # Trained model checkpoint
├── test_metrics.json      # NDCG, Recall, Precision, RMSE, MAE
├── training_history.npz   # Loss curves
└── item_counts.npy        # Item popularity for evaluation
```

### Training Movies Domain

```bash
python src/train_bpr.py --domain movies
```

**Expected Output:**
```
Movies Domain BPR Training Results:
  NDCG@10:      0.0957
  Recall@10:    0.0872
  Precision@10: 0.0607
  Training time: ~1-2 minutes (20 epochs)
```

### Loading Embeddings

After training, embeddings are stored in `.npz` format:

```python
import numpy as np

# Load embeddings
emb = np.load('outputs/books_bpr_YYYYMMDD_HHMMSS/embeddings.npz', allow_pickle=True)

user_embeddings = emb['user_embeddings']  # Shape: (n_users, 64)
item_embeddings = emb['item_embeddings']  # Shape: (n_items, 64)
user2idx = emb['user2idx'].item()         # Dict: user_id -> index
item2idx = emb['item2idx'].item()         # Dict: item_id -> index

# Get embedding for specific user
user_id = "A2S166WSCFIFP5"
user_idx = user2idx[user_id]
user_emb = user_embeddings[user_idx]  # Shape: (64,)
```

### Expected Embedding Quality

**Books Embeddings:**
- NDCG@10: 0.1574 (high quality)
- Effective dimensionality: 31/64 (48% variance captured)
- Diverse representations (avg cosine similarity: 0.032)

**Movies Embeddings:**
- NDCG@10: 0.0957 (acceptable quality)
- Effective dimensionality: 23/64 (36% variance captured)
- Diverse representations (avg cosine similarity: 0.013)

**Cross-Domain Alignment:**
- Same-user embeddings across domains have near-zero correlation (mean: -0.008)
- Confirms domains occupy distinct embedding spaces → need for OT mapping

---

## Running OT Experiments

### Overview

After generating embeddings for both domains, we learn an optimal transport (OT) mapping from Books to Movies user embeddings using the 1,329 overlapping users.

### Step 1: Extract Overlapping User Embeddings

The framework automatically identifies overlapping users when computing OT distributions:

```bash
python src/run_ot_distributions.py \
  --books_embeddings outputs/books_bpr_good_results/embeddings.npz \
  --movies_embeddings outputs/movies_bpr_good_results/embeddings.npz \
  --output outputs/ot_distributions.npz
```

**What this does:**
1. Loads user embeddings from both domains
2. Identifies 1,329 overlapping users
3. Extracts their embeddings: `X` (Books) and `Y` (Movies)
4. Normalizes embeddings: `(emb - mean) / std`
5. Computes empirical distributions on `X` and `Y`
6. Saves distributions and metadata to `outputs/ot_distributions.npz`

**Output Structure:**
```python
ot_data = np.load('outputs/ot_distributions.npz')
X = ot_data['X']  # Books embeddings (1329, 64)
Y = ot_data['Y']  # Movies embeddings (1329, 64)
overlapping_users = ot_data['user_ids']  # List of 1,329 user IDs
```

### Step 2: Learn OT Mapping and Evaluate

```bash
python src/evaluate_ot.py \
  --ot_distributions outputs/ot_distributions.npz \
  --books_embeddings outputs/books_bpr_good_results/embeddings.npz \
  --movies_embeddings outputs/movies_bpr_good_results/embeddings.npz \
  --books_data data/Amazon-Dense-Subset/Amazon-Books-Dense.inter \
  --movies_data data/Amazon-Dense-Subset/Amazon-Movies-Dense.inter
```

**What this does:**
1. Loads OT distributions (`X`, `Y`) and original embeddings
2. Splits overlapping users into train/test (e.g., 80/20)
3. Learns OT mapping on training users:
   - Computes OT transport plan between `X_train` and `Y_train`
   - Uses Sinkhorn algorithm or exact linear program
4. Applies mapping to test users: `Y_pred = OT_map(X_test)`
5. Evaluates transfer quality:
   - **Wasserstein Distance:** Measures distribution alignment
   - **Embedding MSE:** `||Y_pred - Y_test||^2`
   - **Downstream NDCG:** Recommendation quality on Movies domain using transferred embeddings
6. Saves results to JSON and generates plots

**Expected Output:**
```
Optimal Transport Evaluation Results:
  Wasserstein Distance: 0.XX (lower = better alignment)
  Embedding MSE:        0.XX (lower = better reconstruction)
  Cross-Domain NDCG@10: 0.XX (improvement over cold-start baseline)
  
Results saved to: outputs/ot_evaluation_results.json
Plots saved to: outputs/ot_*.png
```

### Full OT Pipeline Example

```bash
# 1. Train BPR models (if not already done)
python src/train_bpr.py --domain books
python src/train_bpr.py --domain movies

# 2. Extract OT distributions
python src/run_ot_distributions.py \
  --books_embeddings outputs/books_bpr_good_results/embeddings.npz \
  --movies_embeddings outputs/movies_bpr_good_results/embeddings.npz \
  --output outputs/ot_distributions.npz

# 3. Learn OT mapping and evaluate
python src/evaluate_ot.py \
  --ot_distributions outputs/ot_distributions.npz \
  --books_embeddings outputs/books_bpr_good_results/embeddings.npz \
  --movies_embeddings outputs/movies_bpr_good_results/embeddings.npz \
  --books_data data/Amazon-Dense-Subset/Amazon-Books-Dense.inter \
  --movies_data data/Amazon-Dense-Subset/Amazon-Movies-Dense.inter
```

---

## Additional Scripts and Utilities

### Embedding Visualization

Visualize embedding quality, distribution, and cross-domain alignment:

```bash
python visualize_embeddings.py
```

**Generates:**
- `embedding_analysis.png`: 12-plot comprehensive analysis
  - Embedding value distributions (Books vs Movies)
  - L2 norms comparison
  - PCA variance explained
  - User-user similarity heatmaps
  - PCA 2D projections
  - t-SNE 2D projections
  - Overlapping users in both embedding spaces
  - Cross-domain similarity histogram

**Output includes:**
- Effective dimensionality (PCA)
- Pairwise similarity analysis
- Cross-domain alignment metrics
- Quality assessment for OT research

### Simple MF Baseline (Alternative to BPR)

Train a standard matrix factorization model optimized for rating prediction:

```bash
python train_simple_baseline.py --domain books
```

**Differences from BPR:**
- Optimizes MSE loss (rating prediction)
- Better RMSE (~0.88) but worse NDCG (~0.04)
- Not recommended for OT (low embedding quality)

### Demo and Usage Examples

Explore the API and common usage patterns:

```bash
python src/demo.py
```

Shows how to:
- Load embeddings
- Access user/item representations
- Compute similarities
- Make predictions
- Extract overlapping users

### Exploratory Data Analysis

Analyze the original Amazon datasets:

```bash
python eda_overlapping.py
```

Shows statistics on:
- Original dataset size and sparsity
- Rating distributions
- User/item coverage
- Overlap analysis

---

## Output Structure

Each experiment creates a timestamped directory in `outputs/`:

```
outputs/
├── books_bpr_YYYYMMDD_HHMMSS/
│   ├── embeddings.npz           # User & item embeddings + ID mappings
│   ├── model.pt                 # PyTorch model checkpoint
│   ├── test_metrics.json        # NDCG, Recall, Precision, RMSE, MAE
│   ├── training_history.npz     # Train/val loss curves
│   └── item_counts.npy          # Item popularity statistics
├── movies_bpr_YYYYMMDD_HHMMSS/  # Same structure for Movies
├── ot_distributions.npz         # OT input data (X, Y, user IDs)
├── ot_evaluation_results.json   # OT mapping results
└── ot_*.png                     # OT visualization plots
```

### Key Files

**`embeddings.npz`** contains:
```python
{
  'user_embeddings': ndarray (n_users, 64),
  'item_embeddings': ndarray (n_items, 64),
  'user2idx': dict {user_id: index},
  'item2idx': dict {item_id: index}
}
```

**`test_metrics.json`** contains:
```json
{
  "rmse": 3.5593,
  "mae": 3.3236,
  "ndcg@10": 0.1574,
  "recall@10": 0.1876,
  "precision@10": 0.0917
}
```

**`ot_distributions.npz`** contains:
```python
{
  'X': ndarray (1329, 64),        # Books embeddings
  'Y': ndarray (1329, 64),        # Movies embeddings
  'user_ids': list of 1329 IDs,  # Overlapping user IDs
  'X_mean': ndarray (64,),        # Books normalization stats
  'X_std': ndarray (64,),
  'Y_mean': ndarray (64,),        # Movies normalization stats
  'Y_std': ndarray (64,)
}
```

---

## Reproducibility Details

### Code and Configuration

- **Repository:** https://github.com/RaghavPu/wasserstein-cdr-embeddings
- **Configuration:** All experiments use `config.yaml` for parameter management
- **Single-domain models:** Trained with `src/train_bpr.py`
- **OT experiments:** Use `src/run_ot_distributions.py` and `src/evaluate_ot.py`

### Dataset

- **Source:** Amazon Review Data (2018), Books and Movies & TV domains
- **Preprocessing:** Dense subset created via `create_dense_subset.py`
- **Format:** Recbole-style `*.inter` files (tab-separated: user_id, item_id, rating)
- **Location:**
  - `data/Amazon-Dense-Subset/Amazon-Books-Dense.inter`
  - `data/Amazon-Dense-Subset/Amazon-Movies-Dense.inter`
- **Statistics:** See [Dataset Information](#dataset-information) section above

### Model Architecture

- **Base recommender:** BPR-style matrix factorization (implemented in `src/models.py`)
- **Embedding dimension:** `d = 64` for both domains
- **Loss function:** BPR pairwise ranking loss with L2 regularization
- **Negative sampling:** 8 negative samples per positive, popularity-based sampling

### Training Details

- **Optimizer:** Adam with learning rate `lr = 0.01` (default from `config.yaml`)
- **Regularization:** L2 weight decay `λ = 10^-4`
- **Batch size:** 512
- **Epochs:** 20 (with early stopping on validation NDCG)
- **Train/val/test split:** 70/10/20
- **Hardware:** Experiments run on CPU (typical time: 1-2 min per domain per epoch)

### OT Configuration

- **Overlapping users:** 1,329 users in both domains
- **OT algorithm:** Sinkhorn (entropic regularization) or exact linear program
- **Normalization:** Embeddings standardized: `(X - mean) / std` before OT
- **Train/test split:** 80/20 split of overlapping users for OT evaluation

### Random Seeds

- **Data splitting:** `random_seed = 42` (set in `config.yaml`)
- **Model initialization:** PyTorch default (can be set via `torch.manual_seed()`)
- **OT evaluation:** Uses same seed for train/test split consistency

### Expected Results

**Books BPR:**
- NDCG@10: 0.157 ± 0.01
- Recall@10: 0.188 ± 0.01
- Precision@10: 0.092 ± 0.005
- Training time: ~2 minutes (20 epochs)

**Movies BPR:**
- NDCG@10: 0.096 ± 0.01
- Recall@10: 0.087 ± 0.01
- Precision@10: 0.061 ± 0.005
- Training time: ~2 minutes (20 epochs)

**Note on RMSE/MAE:** BPR optimizes for ranking, not rating prediction. RMSE/MAE values (~3.5) are expected and should be ignored. Focus on NDCG, Recall, and Precision for BPR models.

---

## Project Structure

```
wasserstein-cdr-embeddings/
├── data/
│   ├── Amazon-Dense-Subset/           # Primary dataset (used in experiments)
│   │   ├── Amazon-Books-Dense.inter
│   │   ├── Amazon-Movies-Dense.inter
│   │   └── README.md
│   ├── Amazon-CrossDomain-Overlapping/  # Original large dataset
│   └── Amazon-KG-5core-*/             # Alternative datasets
├── src/
│   ├── train_bpr.py                   # Main BPR training script
│   ├── run_ot_distributions.py        # Extract OT distributions
│   ├── evaluate_ot.py                 # OT mapping evaluation
│   ├── models.py                      # BPR model implementation
│   ├── bpr_data_loader.py             # BPR triplet generation
│   ├── bpr_trainer.py                 # BPR training loop
│   ├── evaluator.py                   # Metrics computation
│   ├── config.py                      # Configuration management
│   └── utils.py                       # Utility functions
├── outputs/                           # Experiment results
│   ├── books_bpr_*/                   # Books embeddings & metrics
│   ├── movies_bpr_*/                  # Movies embeddings & metrics
│   └── ot_*.npz                       # OT results
├── config.yaml                        # Main configuration file
├── create_dense_subset.py             # Dataset preprocessing
├── visualize_embeddings.py            # Embedding analysis
├── train_simple_baseline.py           # MF baseline (alternative)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Citation

### This Work

If you use this code for your research, please cite:

```
@misc{wasserstein-cdr-embeddings,
  author = {Raghav Punnam, Herry Rao},
  title = {Wasserstein Alignment of User Embeddings for Cross-Domain Recommendation},
  year = {2024},
  url = {https://github.com/RaghavPu/wasserstein-cdr-embeddings}
}
```

### Amazon Dataset

```bibtex
@inproceedings{wang2024amazon,
  title={Amazon-KG: A Knowledge Graph Enhanced Cross-Domain Recommendation Dataset},
  author={Wang, Yuhan and Xie, Qing and Tang, Mengzi and Li, Lin and Yuan, Jingling and Liu, Yongjian},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={123--130},
  year={2024}
}
```

### Optimal Transport

```bibtex
@article{flamary2021pot,
  title={POT: Python optimal transport},
  author={Flamary, R{\'e}mi and Courty, Nicolas and Gramfort, Alexandre and Alaya, Mokhtar Z and Boisbunon, Aur{\'e}lie and Chambon, Stanislas and Chapel, Laetitia and Corenflos, Adrien and Fatras, Kilian and Fournier, Nemo and others},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={78},
  pages={1--8},
  year={2021}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

**Repository:** https://github.com/RaghavPu/wasserstein-cdr-embeddings
