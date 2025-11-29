# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Train on Books Domain

```bash
python src/train_single_domain.py --domain books
```

This will:
- Train a Matrix Factorization model
- Perform 3-fold cross-validation
- Evaluate on test set
- Save embeddings, model, and metrics to `outputs/books_*/`

### 2. Train on Movies Domain

```bash
python src/train_single_domain.py --domain movies
```

### 3. View Results

```bash
python src/demo.py
```

This demo script shows:
- How to load embeddings
- Model performance metrics
- Cross-domain statistics (if both domains are trained)

## Outputs

Each training run creates a timestamped directory with:

```
outputs/books_YYYYMMDD_HHMMSS/
├── embeddings.npz          # User and item embeddings
├── model.pt                # Trained model checkpoint
├── test_metrics.json       # Test performance (RMSE, MAE)
├── cv_results.json         # Cross-validation results
└── training_history.npz    # Loss curves
```

## Using Embeddings in Your Research

### Load Embeddings

```python
import numpy as np

# Load embeddings
data = np.load('outputs/books_20251128_234105/embeddings.npz', allow_pickle=True)
user_embeddings = data['user_embeddings']  # Shape: (10485, 64)
user2idx = data['user2idx'].item()  # user_id -> index mapping

# Get specific user embedding
user_id = "A2S166WSCFIFP5"
user_idx = user2idx[user_id]
embedding = user_embeddings[user_idx]
```

### Extract Overlapping Users

```python
from src.utils import extract_overlapping_embeddings

# Get embeddings for users that appear in both domains
result = extract_overlapping_embeddings(
    'outputs/books_*/embeddings.npz',
    'outputs/movies_*/embeddings.npz',
    output_path='overlapping_embeddings.npz'
)

books_emb = result['books_embeddings']    # Shape: (1062, 64)
movies_emb = result['movies_embeddings']  # Shape: (1062, 64)
user_ids = result['overlapping_user_ids']
```

Now you can learn a mapping: `f: books_emb → movies_emb`

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  embedding_dim: 64        # Embedding dimension
  learning_rate: 0.001     # Learning rate
  
training:
  num_epochs: 10           # Max epochs
  batch_size: 1024         # Batch size
  early_stopping_patience: 3  # Early stopping patience

cross_validation:
  enabled: true
  n_folds: 3              # Number of CV folds
```

## Dataset Statistics

**Books Domain:**
- Users: 10,485
- Items: 8,318
- Interactions: 115,572

**Movies Domain:**
- Users: 5,648
- Items: ~10,000 (estimated)
- Interactions: 67,067

**Overlapping Users:** 1,062 (users in both domains)

## Next Steps for Your Research

1. **Train both domains:**
   ```bash
   python src/train_single_domain.py --domain books
   python src/train_single_domain.py --domain movies
   ```

2. **Extract overlapping embeddings:**
   ```python
   from src.utils import extract_overlapping_embeddings
   result = extract_overlapping_embeddings(
       'outputs/books_*/embeddings.npz',
       'outputs/movies_*/embeddings.npz'
   )
   ```

3. **Learn a mapping:**
   - Use the paired embeddings to learn a Wasserstein mapping
   - Example: `f(user_books_emb) → user_movies_emb`
   - Evaluate transfer: Can Books preferences predict Movies preferences?

## Extending the Framework

### Add a New Model

Edit `src/models.py`:

```python
class YourModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        # Your model architecture
    
    def forward(self, user_ids, item_ids):
        # Return predictions
        pass
    
    def get_user_embeddings(self):
        # Return user embeddings
        return self.user_embeddings.weight.data
```

Then update `config.yaml`:
```yaml
model:
  name: "YourModel"
```

### Add Custom Metrics

Edit `src/evaluator.py` to add your custom evaluation metrics.

## Troubleshooting

**Issue:** Out of memory
- **Solution:** Reduce `batch_size` in `config.yaml`

**Issue:** Training too slow
- **Solution:** 
  - Reduce `num_epochs`
  - Set `cross_validation.enabled: false`
  - Use GPU: `training.device: "cuda"`

**Issue:** Want faster testing
- **Solution:** Use `--no-cv` flag to skip cross-validation

## Support

For questions about the dataset, see `data/README.md` or the SIGIR 2024 paper.

