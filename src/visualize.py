"""
Visualization utilities for training results and embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import json


def plot_training_history(history_path: str, output_path: Optional[str] = None):
    """
    Plot training and validation loss curves.
    
    Args:
        history_path: Path to training_history.npz
        output_path: Optional path to save plot
    """
    history = np.load(history_path)
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = np.min(val_losses)
    plt.scatter([best_epoch], [best_val_loss], c='red', s=100, 
               marker='*', zorder=5, label=f'Best (Epoch {best_epoch})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embedding_distribution(embeddings_path: str, output_path: Optional[str] = None):
    """
    Plot distribution of embedding values.
    
    Args:
        embeddings_path: Path to embeddings.npz
        output_path: Optional path to save plot
    """
    data = np.load(embeddings_path, allow_pickle=True)
    user_embeddings = data['user_embeddings']
    item_embeddings = data['item_embeddings']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # User embedding distribution
    axes[0, 0].hist(user_embeddings.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('User Embedding Values Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Item embedding distribution
    axes[0, 1].hist(item_embeddings.flatten(), bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Item Embedding Values Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # User embedding norms
    user_norms = np.linalg.norm(user_embeddings, axis=1)
    axes[1, 0].hist(user_norms, bins=50, alpha=0.7, color='blue')
    axes[1, 0].set_title('User Embedding L2 Norms', fontweight='bold')
    axes[1, 0].set_xlabel('L2 Norm')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Item embedding norms
    item_norms = np.linalg.norm(item_embeddings, axis=1)
    axes[1, 1].hist(item_norms, bins=50, alpha=0.7, color='green')
    axes[1, 1].set_title('Item Embedding L2 Norms', fontweight='bold')
    axes[1, 1].set_xlabel('L2 Norm')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cross_domain_comparison(books_embeddings_path: str, 
                                 movies_embeddings_path: str,
                                 output_path: Optional[str] = None):
    """
    Plot comparison between Books and Movies domain embeddings.
    
    Args:
        books_embeddings_path: Path to books embeddings
        movies_embeddings_path: Path to movies embeddings
        output_path: Optional path to save plot
    """
    # Load embeddings
    books_data = np.load(books_embeddings_path, allow_pickle=True)
    movies_data = np.load(movies_embeddings_path, allow_pickle=True)
    
    books_user_emb = books_data['user_embeddings']
    movies_user_emb = movies_data['user_embeddings']
    
    # Get overlapping users
    books_user2idx = books_data['user2idx'].item()
    movies_user2idx = movies_data['user2idx'].item()
    
    books_users = set(books_user2idx.keys())
    movies_users = set(movies_user2idx.keys())
    overlapping_users = list(books_users & movies_users)
    
    # Extract overlapping embeddings
    books_indices = [books_user2idx[user] for user in overlapping_users]
    movies_indices = [movies_user2idx[user] for user in overlapping_users]
    
    books_overlap_emb = books_user_emb[books_indices]
    movies_overlap_emb = movies_user_emb[movies_indices]
    
    # Compute statistics
    cosine_sims = []
    for b_emb, m_emb in zip(books_overlap_emb, movies_overlap_emb):
        cos_sim = np.dot(b_emb, m_emb) / (np.linalg.norm(b_emb) * np.linalg.norm(m_emb))
        cosine_sims.append(cos_sim)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Cosine similarity distribution
    axes[0].hist(cosine_sims, bins=30, alpha=0.7, color='purple')
    axes[0].axvline(np.mean(cosine_sims), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(cosine_sims):.3f}')
    axes[0].set_title('Cosine Similarity\n(Same User, Different Domains)', fontweight='bold')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # L2 distance distribution
    l2_dists = np.linalg.norm(books_overlap_emb - movies_overlap_emb, axis=1)
    axes[1].hist(l2_dists, bins=30, alpha=0.7, color='orange')
    axes[1].axvline(np.mean(l2_dists), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(l2_dists):.3f}')
    axes[1].set_title('L2 Distance\n(Same User, Different Domains)', fontweight='bold')
    axes[1].set_xlabel('L2 Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Embedding norm comparison
    books_norms = np.linalg.norm(books_overlap_emb, axis=1)
    movies_norms = np.linalg.norm(movies_overlap_emb, axis=1)
    axes[2].scatter(books_norms, movies_norms, alpha=0.5, s=20)
    axes[2].plot([0, max(books_norms.max(), movies_norms.max())],
                [0, max(books_norms.max(), movies_norms.max())],
                'r--', linewidth=2, label='y=x')
    axes[2].set_title('Embedding Norm Comparison', fontweight='bold')
    axes[2].set_xlabel('Books Embedding Norm')
    axes[2].set_ylabel('Movies Embedding Norm')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_experiment(experiment_dir: str):
    """
    Create all visualizations for an experiment.
    
    Args:
        experiment_dir: Path to experiment output directory
    """
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        return
    
    print(f"Creating visualizations for: {experiment_dir}")
    
    # Plot training history
    history_path = exp_path / "training_history.npz"
    if history_path.exists():
        plot_training_history(
            str(history_path),
            output_path=str(exp_path / "training_curves.png")
        )
    
    # Plot embedding distributions
    embeddings_path = exp_path / "embeddings.npz"
    if embeddings_path.exists():
        plot_embedding_distribution(
            str(embeddings_path),
            output_path=str(exp_path / "embedding_distributions.png")
        )
    
    print("✓ Visualizations complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Visualize specific experiment
        visualize_experiment(sys.argv[1])
    else:
        # Visualize most recent experiment
        output_dir = Path("outputs")
        all_dirs = sorted(output_dir.glob("*_*"))
        
        if all_dirs:
            latest = all_dirs[-1]
            print(f"Visualizing most recent experiment: {latest}")
            visualize_experiment(str(latest))
        else:
            print("No experiments found. Please train a model first:")
            print("  python src/train_single_domain.py --domain books")

