"""
BPR-specific data loaders with negative sampling.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
from collections import defaultdict


class BPRDataset(Dataset):
    """
    Dataset for BPR training with negative sampling.
    
    For each positive (user, item) pair, samples N negative items
    that the user hasn't interacted with.
    """
    
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray,
                 n_items: int, n_negatives: int = 4,
                 negative_sampling_strategy: str = 'uniform'):
        """
        Initialize BPR dataset.
        
        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices (positive items)
            n_items: Total number of items in the catalog
            n_negatives: Number of negative samples per positive
            negative_sampling_strategy: 'uniform' or 'popularity'
        """
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.strategy = negative_sampling_strategy
        
        # Build user interaction sets for fast negative sampling
        self.user_items = defaultdict(set)
        for user, item in zip(user_ids, item_ids):
            self.user_items[user].add(item)
        
        # For popularity-based sampling
        if negative_sampling_strategy == 'popularity':
            item_counts = np.bincount(item_ids, minlength=n_items)
            # Add smoothing to avoid zero probabilities
            self.item_probs = (item_counts + 1) / (item_counts.sum() + n_items)
        else:
            self.item_probs = None
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, np.ndarray]:
        """
        Get a training sample.
        
        Returns:
            (user_id, positive_item_id, negative_item_ids)
        """
        user = self.user_ids[idx]
        pos_item = self.item_ids[idx]
        
        # Sample negative items
        neg_items = self._sample_negatives(user)
        
        return user, pos_item, neg_items
    
    def _sample_negatives(self, user: int) -> np.ndarray:
        """Sample negative items for a user."""
        user_positives = self.user_items[user]
        neg_items = []
        
        attempts = 0
        max_attempts = self.n_negatives * 10  # Avoid infinite loop
        
        while len(neg_items) < self.n_negatives and attempts < max_attempts:
            if self.strategy == 'popularity' and self.item_probs is not None:
                # Popularity-based sampling
                candidates = np.random.choice(
                    self.n_items,
                    size=self.n_negatives * 2,
                    p=self.item_probs
                )
            else:
                # Uniform sampling
                candidates = np.random.randint(0, self.n_items, size=self.n_negatives * 2)
            
            for item in candidates:
                if item not in user_positives and item not in neg_items:
                    neg_items.append(item)
                    if len(neg_items) >= self.n_negatives:
                        break
            
            attempts += 1
        
        # If we couldn't sample enough, fill with random (rare edge case)
        while len(neg_items) < self.n_negatives:
            item = np.random.randint(0, self.n_items)
            if item not in user_positives:
                neg_items.append(item)
        
        return np.array(neg_items, dtype=np.int64)


def create_bpr_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame,
                           test_df: pd.DataFrame, n_items: int,
                           batch_size: int = 1024, n_negatives: int = 4,
                           num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    Create BPR DataLoaders for train/val/test.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        n_items: Total number of items
        batch_size: Batch size
        n_negatives: Number of negative samples per positive
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}
    
    # Training loader with BPR sampling
    train_dataset = BPRDataset(
        train_df['user_idx'].values,
        train_df['item_idx'].values,
        n_items=n_items,
        n_negatives=n_negatives,
        negative_sampling_strategy='popularity'  # Better than uniform
    )
    dataloaders['train'] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    # Val and test loaders (no negative sampling needed for evaluation)
    # These are regular datasets for evaluation purposes
    from data_loader import InteractionDataset
    
    val_dataset = InteractionDataset(
        val_df['user_idx'].values,
        val_df['item_idx'].values,
        val_df['rating'].values
    )
    dataloaders['val'] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_dataset = InteractionDataset(
        test_df['user_idx'].values,
        test_df['item_idx'].values,
        test_df['rating'].values
    )
    dataloaders['test'] = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return dataloaders

