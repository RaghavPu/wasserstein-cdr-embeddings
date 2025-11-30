"""
Simple Baseline: Regularized Matrix Factorization (Netflix-style)

This is THE standard academic baseline for recommendation systems.
Produces user and item embeddings from explicit ratings.

Model: r_hat = μ + b_u + b_i + p_u^T · q_i
Where:
  - μ = global mean rating
  - b_u, b_i = user/item biases
  - p_u, q_i = user/item embeddings (what we want!)

Loss: MSE + L2 regularization on all parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime


class SimpleMatrixFactorization(nn.Module):
    """
    Standard regularized matrix factorization baseline.
    The most common academic baseline for recommendation systems.
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64):
        """
        Initialize standard MF model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of embeddings (k in the formula)
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings (p_u and q_i)
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # User and item biases (b_u and b_i)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        # Global mean (μ) - learned as parameter
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize with small random values
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings: r_hat = μ + b_u + b_i + p_u^T · q_i
        
        Args:
            user_ids: User indices
            item_ids: Item indices
        
        Returns:
            Predicted ratings
        """
        # Get embeddings
        p_u = self.user_embeddings(user_ids)  # [batch, embedding_dim]
        q_i = self.item_embeddings(item_ids)  # [batch, embedding_dim]
        
        # Get biases
        b_u = self.user_bias(user_ids).squeeze()  # [batch]
        b_i = self.item_bias(item_ids).squeeze()  # [batch]
        
        # Dot product
        interaction = (p_u * q_i).sum(dim=1)  # [batch]
        
        # Full prediction
        prediction = self.global_bias + b_u + b_i + interaction
        
        return prediction
    
    def get_user_embeddings(self) -> np.ndarray:
        """Get all user embeddings (p_u)."""
        return self.user_embeddings.weight.detach().cpu().numpy()
    
    def get_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings (q_i)."""
        return self.item_embeddings.weight.detach().cpu().numpy()


def compute_ndcg_sample(model, df, user2idx, n_items, n_sample_users=100, k=10, device='cpu'):
    """
    Compute NDCG@k on a sample of users for monitoring during training.
    
    Args:
        model: Trained model
        df: DataFrame with user_idx, item_idx, rating
        user2idx: User ID to index mapping
        n_items: Total number of items
        n_sample_users: Number of users to sample for evaluation
        k: Top-k for NDCG
        device: Device for computation
    
    Returns:
        Average NDCG@k
    """
    model.eval()
    
    # Group by user
    user_items = df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    # Sample users who have at least k interactions
    valid_users = [u for u, items in user_items.items() if len(items) >= 3]
    if len(valid_users) > n_sample_users:
        sampled_users = np.random.choice(valid_users, n_sample_users, replace=False)
    else:
        sampled_users = valid_users
    
    ndcg_scores = []
    
    with torch.no_grad():
        for user_id in sampled_users:
            # Get all item scores for this user
            user_tensor = torch.full((n_items,), user_id, dtype=torch.long).to(device)
            item_tensor = torch.arange(n_items, dtype=torch.long).to(device)
            
            scores = model(user_tensor, item_tensor).cpu().numpy()
            
            # Get top-k items
            top_k_items = np.argsort(scores)[-k:][::-1]
            
            # Get ground truth items
            ground_truth = user_items[user_id]
            
            # Compute hits
            hits = [1 if item in ground_truth else 0 for item in top_k_items]
            
            # Compute NDCG
            dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
            idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(ground_truth), k))])
            ndcg = dcg / idcg if idcg > 0 else 0
            
            ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def train_simple_mf(domain: str = 'books', config_path: str = 'config.yaml'):
    """
    Train simple MF baseline on one domain.
    
    Args:
        domain: 'books' or 'movies'
        config_path: Path to config file
    """
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print(f"Simple Baseline: Regularized Matrix Factorization")
    print(f"Domain: {domain.upper()}")
    print("="*70)
    
    # Paths
    if domain == 'books':
        data_path = config['data']['books_path']
    else:
        data_path = config['data']['movies_path']
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(data_path, sep='\t')
    df.columns = ['user_id', 'item_id', 'rating']
    
    # Create mappings
    user2idx = {user: idx for idx, user in enumerate(df['user_id'].unique())}
    item2idx = {item: idx for idx, item in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user2idx)
    df['item_idx'] = df['item_id'].map(item2idx)
    
    n_users = len(user2idx)
    n_items = len(item2idx)
    
    print(f"  Users: {n_users:,}")
    print(f"  Items: {n_items:,}")
    print(f"  Ratings: {len(df):,}")
    print(f"  Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
    print(f"  Mean rating: {df['rating'].mean():.2f}")
    
    # Train/val/test split
    print("\n📊 Splitting data (70% train, 10% val, 20% test)...")
    train_df = df.sample(frac=0.7, random_state=42)
    remaining = df.drop(train_df.index)
    val_df = remaining.sample(frac=0.5, random_state=42)
    test_df = remaining.drop(val_df.index)
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    
    # Create model
    print("\n🔧 Creating model...")
    embedding_dim = config['model']['embedding_dim']
    model = SimpleMatrixFactorization(n_users, n_items, embedding_dim)
    
    # Setup training
    device = config['training'].get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("  ⚠️  CUDA requested but not available, using CPU")
        device = 'cpu'
    model = model.to(device)
    
    learning_rate = config['model']['learning_rate']
    weight_decay = config['model']['weight_decay']  # L2 regularization (λ in formula)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay (λ): {weight_decay}")
    print(f"  Device: {device}")
    
    # Training loop
    print("\n🚀 Training...")
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    best_val_ndcg = 0.0
    
    print(f"\nNote: Computing NDCG@10 on 100 sample users per epoch")
    print(f"(Full NDCG evaluation on test set at the end)\n")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        # Shuffle training data
        train_df_shuffled = train_df.sample(frac=1)
        
        # Create progress bar for training
        n_train_batches = len(train_df_shuffled) // batch_size + 1
        pbar = tqdm(range(0, len(train_df_shuffled), batch_size), 
                   desc=f'Epoch {epoch+1}/{num_epochs} [Train]',
                   total=n_train_batches)
        
        for i in pbar:
            batch = train_df_shuffled.iloc[i:i+batch_size]
            
            user_ids = torch.LongTensor(batch['user_idx'].values).to(device)
            item_ids = torch.LongTensor(batch['item_idx'].values).to(device)
            ratings = torch.FloatTensor(batch['rating'].values).to(device)
            
            # Forward pass
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': train_loss / n_batches})
        
        train_loss /= n_batches
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        n_batches = 0
        
        # Create progress bar for validation
        n_val_batches = len(val_df) // batch_size + 1
        pbar = tqdm(range(0, len(val_df), batch_size),
                   desc=f'Epoch {epoch+1}/{num_epochs} [Val]  ',
                   total=n_val_batches)
        
        with torch.no_grad():
            for i in pbar:
                batch = val_df.iloc[i:i+batch_size]
                
                user_ids = torch.LongTensor(batch['user_idx'].values).to(device)
                item_ids = torch.LongTensor(batch['item_idx'].values).to(device)
                ratings = torch.FloatTensor(batch['rating'].values).to(device)
                
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                val_loss += loss.item()
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': val_loss / n_batches})
        
        val_loss /= n_batches
        
        # Compute NDCG on sample (for monitoring)
        val_ndcg = compute_ndcg_sample(model, val_df, user2idx, n_items, 
                                       n_sample_users=100, k=10, device=device)
        
        # Format output with colors
        improvement = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improvement += " 🔽 Best Loss"
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            improvement += " 🔼 Best NDCG"
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val NDCG@10: {val_ndcg:.4f}"
              f"{improvement}")
    
    # Test evaluation
    print("\n📈 Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i:i+batch_size]
            
            user_ids = torch.LongTensor(batch['user_idx'].values).to(device)
            item_ids = torch.LongTensor(batch['item_idx'].values).to(device)
            ratings = torch.FloatTensor(batch['rating'].values).to(device)
            
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            
            test_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
    
    test_loss /= (len(test_df) // batch_size + 1)
    rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets))**2))
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    
    # Compute NDCG on more users for final evaluation
    print(f"\nComputing NDCG@10 on 500 test users...")
    test_ndcg = compute_ndcg_sample(model, test_df, user2idx, n_items, 
                                    n_sample_users=500, k=10, device=device)
    
    print(f"\n✓ Test Results:")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  NDCG@10:  {test_ndcg:.4f}")
    print(f"\n  📊 Interpretation:")
    print(f"     RMSE ~1.0 = good rating prediction")
    print(f"     NDCG ~0.02-0.05 = decent ranking for simple MF")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = config['logging']['output_dir']
    output_dir = Path(f'{output_base}/{domain}_simple_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    user_embeddings = model.get_user_embeddings()
    item_embeddings = model.get_item_embeddings()
    
    np.savez(
        output_dir / 'embeddings.npz',
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user2idx=user2idx,
        item2idx=item2idx
    )
    
    # Save metrics
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'ndcg@10': float(test_ndcg),
        'test_loss': float(test_loss),
        'best_val_loss': float(best_val_loss),
        'best_val_ndcg': float(best_val_ndcg)
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Saved to: {output_dir}")
    print(f"  User embeddings: {user_embeddings.shape}")
    print(f"  Item embeddings: {item_embeddings.shape}")
    
    print("\n" + "="*70)
    print("✓ Training Complete!")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train simple MF baseline')
    parser.add_argument('--domain', type=str, default='books', choices=['books', 'movies'])
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    train_simple_mf(args.domain, args.config)

