"""Utility functions for working with embeddings and results."""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_embeddings(embeddings_path: str) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Load saved embeddings.
    
    Args:
        embeddings_path: Path to embeddings.npz file
    
    Returns:
        Tuple of (user_embeddings, item_embeddings, user2idx, item2idx)
    """
    data = np.load(embeddings_path, allow_pickle=True)
    return (
        data['user_embeddings'],
        data['item_embeddings'],
        data['user2idx'].item(),
        data['item2idx'].item()
    )


def get_user_embedding(user_id: str, embeddings_path: str) -> Optional[np.ndarray]:
    """
    Get embedding for a specific user.
    
    Args:
        user_id: User ID
        embeddings_path: Path to embeddings file
    
    Returns:
        User embedding vector or None if user not found
    """
    user_embeddings, _, user2idx, _ = load_embeddings(embeddings_path)
    
    if user_id not in user2idx:
        return None
    
    user_idx = user2idx[user_id]
    return user_embeddings[user_idx]


def get_overlapping_users(books_embeddings_path: str, 
                         movies_embeddings_path: str) -> Tuple[list, Dict]:
    """
    Find users that appear in both domains.
    
    Args:
        books_embeddings_path: Path to books embeddings
        movies_embeddings_path: Path to movies embeddings
    
    Returns:
        Tuple of (overlapping_user_ids, user_indices)
    """
    # Load mappings
    _, _, books_user2idx, _ = load_embeddings(books_embeddings_path)
    _, _, movies_user2idx, _ = load_embeddings(movies_embeddings_path)
    
    # Find overlap
    books_users = set(books_user2idx.keys())
    movies_users = set(movies_user2idx.keys())
    overlapping_users = list(books_users & movies_users)
    
    # Get indices
    user_indices = {
        'books': {user: books_user2idx[user] for user in overlapping_users},
        'movies': {user: movies_user2idx[user] for user in overlapping_users}
    }
    
    return overlapping_users, user_indices


def extract_overlapping_embeddings(books_embeddings_path: str,
                                   movies_embeddings_path: str,
                                   output_path: Optional[str] = None) -> Dict:
    """
    Extract embeddings for overlapping users.
    
    Args:
        books_embeddings_path: Path to books embeddings
        movies_embeddings_path: Path to movies embeddings
        output_path: Optional path to save extracted embeddings
    
    Returns:
        Dictionary with overlapping embeddings
    """
    # Load embeddings
    books_emb, _, books_user2idx, _ = load_embeddings(books_embeddings_path)
    movies_emb, _, movies_user2idx, _ = load_embeddings(movies_embeddings_path)
    
    # Get overlapping users
    overlapping_users, user_indices = get_overlapping_users(
        books_embeddings_path, movies_embeddings_path
    )
    
    # Extract embeddings
    books_indices = [user_indices['books'][user] for user in overlapping_users]
    movies_indices = [user_indices['movies'][user] for user in overlapping_users]
    
    overlapping_books_emb = books_emb[books_indices]
    overlapping_movies_emb = movies_emb[movies_indices]
    
    result = {
        'overlapping_user_ids': overlapping_users,
        'books_embeddings': overlapping_books_emb,
        'movies_embeddings': overlapping_movies_emb,
        'n_overlapping_users': len(overlapping_users)
    }
    
    # Save if output path provided
    if output_path:
        np.savez(
            output_path,
            overlapping_user_ids=overlapping_users,
            books_embeddings=overlapping_books_emb,
            movies_embeddings=overlapping_movies_emb
        )
        print(f"Saved overlapping embeddings to {output_path}")
        print(f"  Number of overlapping users: {len(overlapping_users)}")
        print(f"  Books embeddings shape: {overlapping_books_emb.shape}")
        print(f"  Movies embeddings shape: {overlapping_movies_emb.shape}")
    
    return result


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_results(results_paths: Dict[str, str]) -> None:
    """
    Compare results from multiple experiments.
    
    Args:
        results_paths: Dict mapping experiment names to metrics paths
    """
    print("\n" + "="*60)
    print("Experiment Comparison")
    print("="*60)
    
    results = {}
    for name, path in results_paths.items():
        results[name] = load_metrics(path)
    
    # Find all metrics
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    
    # Print comparison table
    print(f"\n{'Metric':<20}", end='')
    for name in results_paths.keys():
        print(f"{name:>15}", end='')
    print()
    print("-"*60)
    
    for metric in sorted(all_metrics):
        print(f"{metric:<20}", end='')
        for name in results_paths.keys():
            if metric in results[name]:
                print(f"{results[name][metric]:>15.4f}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()
    print("="*60)

