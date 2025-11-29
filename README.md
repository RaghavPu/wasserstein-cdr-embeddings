# Wasserstein Cross-Domain Recommendation

A modular framework for training recommendation models on cross-domain datasets.

## Project Structure

```
wasserstein-cdr/
├── data/                           # Dataset directory
│   ├── Amazon-KG-5core-Books/
│   └── Amazon-KG-5core-Movies_and_TV/
├── src/                            # Source code
│   ├── config.py                   # Configuration loader
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── models.py                   # Recommendation models (MF, NeuMF)
│   ├── trainer.py                  # Training and cross-validation
│   ├── evaluator.py                # Evaluation metrics
│   └── train_single_domain.py      # Main experiment runner
├── outputs/                        # Experiment outputs
├── config.yaml                     # Configuration file
└── requirements.txt                # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train on Books Domain

```bash
python src/train_single_domain.py --domain books
```

### Train on Movies Domain

```bash
python src/train_single_domain.py --domain movies
```

### Train without Cross-Validation

```bash
python src/train_single_domain.py --domain books --no-cv
```

### Custom Configuration

```bash
python src/train_single_domain.py --domain books --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- **Data settings**: Test/validation split ratios, random seed
- **Model settings**: Embedding dimension, learning rate, dropout
- **Training settings**: Batch size, epochs, early stopping patience
- **Cross-validation**: Number of folds
- **Evaluation metrics**: RMSE, MAE, NDCG@k, Recall@k, Precision@k

## Outputs

Each experiment creates a timestamped directory in `outputs/` containing:
- `model.pt` - Trained model checkpoint
- `embeddings.npz` - User and item embeddings
- `test_metrics.json` - Test set performance metrics
- `cv_results.json` - Cross-validation results (if enabled)
- `training_history.npz` - Training and validation loss curves

## Modular Design

The codebase is designed to be modular and extensible:

1. **Data Loader** (`data_loader.py`): Easily adapt to new datasets
2. **Models** (`models.py`): Add new recommendation models by inheriting from `nn.Module`
3. **Trainer** (`trainer.py`): Reusable training loop with early stopping
4. **Evaluator** (`evaluator.py`): Comprehensive evaluation metrics
5. **Config** (`config.py`): Centralized configuration management

## Getting User Embeddings

After training, embeddings are saved and can be loaded:

```python
import numpy as np

# Load embeddings
data = np.load('outputs/books_20250129_120000/embeddings.npz', allow_pickle=True)
user_embeddings = data['user_embeddings']  # Shape: (n_users, embedding_dim)
user2idx = data['user2idx'].item()  # Mapping: user_id -> index

# Get embedding for specific user
user_id = "A2S166WSCFIFP5"
user_idx = user2idx[user_id]
embedding = user_embeddings[user_idx]
```

## Cross-Domain Usage

Train models on both domains separately:

```bash
python src/train_single_domain.py --domain books
python src/train_single_domain.py --domain movies
```

Then use the saved embeddings to learn mappings between domains for your research.

## Evaluation Metrics

The framework computes:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Recall@k**: Recall at top-k
- **Precision@k**: Precision at top-k

## Models

### Matrix Factorization (MF)
- Standard collaborative filtering baseline
- Learns latent user and item embeddings
- Prediction: `rating = user_embedding · item_embedding + biases`

### Neural Matrix Factorization (NeuMF)
- Combines MF with Multi-Layer Perceptron
- More expressive than standard MF
- To use: Set `model.name: "NeuralMF"` in `config.yaml`

## Extending the Framework

### Add a New Model

```python
# In src/models.py
class YourModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, **kwargs):
        super().__init__()
        # Define your model
    
    def forward(self, user_ids, item_ids):
        # Return predictions
        pass
    
    def get_user_embeddings(self):
        # Return user embeddings
        pass
```

Register in `get_model()` function.

### Add New Metrics

```python
# In src/evaluator.py
def compute_your_metric(self, predictions, targets):
    # Compute your metric
    return metric_value
```

## Citation

If you use this code, please cite the Amazon-KG dataset:

```
Yuhan Wang, Qing Xie, Mengzi Tang, Lin Li, Jingling Yuan, and Yongjian Liu. 2024.
Amazon-KG: A Knowledge Graph Enhanced Cross-Domain Recommendation Dataset.
In SIGIR '24. ACM, 123–130.
```

