"""
Demo script showing how to use the trained model and embeddings.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import load_embeddings, get_user_embedding, extract_overlapping_embeddings


def demo_load_embeddings():
    """Demonstrate loading embeddings."""
    print("="*60)
    print("Demo: Loading Embeddings")
    print("="*60)
    
    # Find the most recent books output directory
    output_dir = Path("outputs")
    books_dirs = sorted(output_dir.glob("books_*"))
    
    if not books_dirs:
        print("No trained models found. Please run training first:")
        print("  python src/train_single_domain.py --domain books")
        return
    
    latest_dir = books_dirs[-1]
    embeddings_path = latest_dir / "embeddings.npz"
    
    print(f"\nLoading embeddings from: {embeddings_path}")
    
    # Load embeddings
    user_embeddings, item_embeddings, user2idx, item2idx = load_embeddings(str(embeddings_path))
    
    print(f"\nEmbedding Statistics:")
    print(f"  Number of users: {len(user2idx)}")
    print(f"  Number of items: {len(item2idx)}")
    print(f"  Embedding dimension: {user_embeddings.shape[1]}")
    print(f"  User embeddings shape: {user_embeddings.shape}")
    print(f"  Item embeddings shape: {item_embeddings.shape}")
    
    # Show sample users
    sample_users = list(user2idx.keys())[:5]
    print(f"\nSample user IDs: {sample_users}")
    
    # Get embedding for first user
    first_user = sample_users[0]
    user_idx = user2idx[first_user]
    embedding = user_embeddings[user_idx]
    
    print(f"\nEmbedding for user '{first_user}':")
    print(f"  Shape: {embedding.shape}")
    print(f"  First 10 dimensions: {embedding[:10]}")
    print(f"  L2 norm: {np.linalg.norm(embedding):.4f}")


def demo_cross_domain():
    """Demonstrate loading embeddings from both domains for cross-domain research."""
    print("\n" + "="*60)
    print("Demo: Cross-Domain Embeddings")
    print("="*60)
    
    output_dir = Path("outputs")
    books_dirs = sorted(output_dir.glob("books_*"))
    movies_dirs = sorted(output_dir.glob("movies_*"))
    
    if not books_dirs:
        print("\nBooks domain not trained yet. Train with:")
        print("  python src/train_single_domain.py --domain books")
        return
    
    if not movies_dirs:
        print("\nMovies domain not trained yet. Train with:")
        print("  python src/train_single_domain.py --domain movies")
        print("\nFor now, showing only Books domain statistics.")
        return
    
    books_path = books_dirs[-1] / "embeddings.npz"
    movies_path = movies_dirs[-1] / "embeddings.npz"
    
    print(f"\nBooks embeddings: {books_path}")
    print(f"Movies embeddings: {movies_path}")
    
    # Extract overlapping embeddings
    result = extract_overlapping_embeddings(
        str(books_path),
        str(movies_path),
        output_path="outputs/overlapping_embeddings.npz"
    )
    
    print(f"\nCross-Domain Statistics:")
    print(f"  Overlapping users: {result['n_overlapping_users']}")
    print(f"  Books embeddings shape: {result['books_embeddings'].shape}")
    print(f"  Movies embeddings shape: {result['movies_embeddings'].shape}")
    
    # Compute some simple statistics
    books_emb = result['books_embeddings']
    movies_emb = result['movies_embeddings']
    
    # Compute cosine similarities between corresponding embeddings
    cosine_sims = []
    for i in range(len(books_emb)):
        book_vec = books_emb[i]
        movie_vec = movies_emb[i]
        cos_sim = np.dot(book_vec, movie_vec) / (
            np.linalg.norm(book_vec) * np.linalg.norm(movie_vec)
        )
        cosine_sims.append(cos_sim)
    
    print(f"\nCosine Similarity Statistics (same user, different domains):")
    print(f"  Mean: {np.mean(cosine_sims):.4f}")
    print(f"  Std:  {np.std(cosine_sims):.4f}")
    print(f"  Min:  {np.min(cosine_sims):.4f}")
    print(f"  Max:  {np.max(cosine_sims):.4f}")
    
    print(f"\nℹ️  You can now use these paired embeddings to learn a mapping between domains!")
    print(f"   For example: Learn f: books_embedding → movies_embedding")


def demo_model_metrics():
    """Show model performance metrics."""
    print("\n" + "="*60)
    print("Demo: Model Performance Metrics")
    print("="*60)
    
    import json
    
    output_dir = Path("outputs")
    books_dirs = sorted(output_dir.glob("books_*"))
    
    if not books_dirs:
        print("No trained models found.")
        return
    
    latest_dir = books_dirs[-1]
    metrics_path = latest_dir / "test_metrics.json"
    
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        print(f"\nTest Set Performance:")
        for metric, value in metrics.items():
            print(f"  {metric.upper():10s}: {value:.4f}")
    
    # Load training history if available
    history_path = latest_dir / "training_history.npz"
    if history_path.exists():
        history = np.load(history_path)
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        
        print(f"\nTraining History:")
        print(f"  Total epochs: {len(train_losses)}")
        print(f"  Final train loss: {train_losses[-1]:.4f}")
        print(f"  Final val loss: {val_losses[-1]:.4f}")
        print(f"  Best val loss: {np.min(val_losses):.4f} (epoch {np.argmin(val_losses) + 1})")


if __name__ == "__main__":
    # Run all demos
    demo_load_embeddings()
    demo_model_metrics()
    demo_cross_domain()
    
    print("\n" + "="*60)
    print("✓ Demo Complete!")
    print("="*60)

