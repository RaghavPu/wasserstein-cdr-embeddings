"""
Data loading and preprocessing utilities for recommendation datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    """PyTorch Dataset for user-item interactions."""
    
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray):
        """
        Initialize interaction dataset.
        
        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices  
            ratings: Array of ratings/interactions
        """
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class RecommendationDataLoader:
    """
    Data loader for recommendation datasets.
    Handles loading, preprocessing, and train/val/test splits.
    """
    
    def __init__(self, data_path: str, test_ratio: float = 0.2, 
                 val_ratio: float = 0.1, random_seed: int = 42):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to interaction data file
            test_ratio: Proportion of data for testing
            val_ratio: Proportion of training data for validation
            random_seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # Mappings
        self.user2idx: Dict[str, int] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2user: Dict[int, str] = {}
        self.idx2item: Dict[int, str] = {}
        
        # Load and process data
        self.df = self._load_data()
        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)
        
        print(f"Loaded data from {data_path}")
        print(f"  Users: {self.n_users}")
        print(f"  Items: {self.n_items}")
        print(f"  Interactions: {len(self.df)}")
        print(f"  Sparsity: {1 - len(self.df) / (self.n_users * self.n_items):.4f}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess interaction data."""
        # Load data
        df = pd.read_csv(self.data_path, sep='\t')
        
        # Rename columns (remove type annotations)
        df.columns = ['user_id', 'item_id', 'rating']
        
        # Create mappings
        unique_users = df['user_id'].unique()
        unique_items = df['item_id'].unique()
        
        self.user2idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item2idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        
        # Convert to indices
        df['user_idx'] = df['user_id'].map(self.user2idx)
        df['item_idx'] = df['item_id'].map(self.item2idx)
        
        return df
    
    def get_train_val_test_split(self, strategy: str = 'random') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            strategy: Split strategy ('random' or 'temporal')
        
        Returns:
            train_df, val_df, test_df
        """
        if strategy == 'random':
            # Random split
            train_val_df, test_df = train_test_split(
                self.df, test_size=self.test_ratio, random_state=self.random_seed
            )
            train_df, val_df = train_test_split(
                train_val_df, test_size=self.val_ratio / (1 - self.test_ratio),
                random_state=self.random_seed
            )
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
        
        print(f"\nData split:")
        print(f"  Train: {len(train_df)} ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/len(self.df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} ({len(test_df)/len(self.df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_kfold_splits(self, n_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate K-fold cross-validation splits.
        
        Args:
            n_folds: Number of folds
        
        Returns:
            List of (train_df, val_df) tuples
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        splits = []
        
        for train_idx, val_idx in kf.split(self.df):
            train_df = self.df.iloc[train_idx]
            val_df = self.df.iloc[val_idx]
            splits.append((train_df, val_df))
        
        print(f"\nCreated {n_folds}-fold cross-validation splits")
        return splits
    
    def create_dataloaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          test_df: Optional[pd.DataFrame] = None,
                          batch_size: int = 1024, num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for train, val, and test sets.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe (optional)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        
        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {}
        
        # Training loader
        train_dataset = InteractionDataset(
            train_df['user_idx'].values,
            train_df['item_idx'].values,
            train_df['rating'].values
        )
        dataloaders['train'] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        # Validation loader
        val_dataset = InteractionDataset(
            val_df['user_idx'].values,
            val_df['item_idx'].values,
            val_df['rating'].values
        )
        dataloaders['val'] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        # Test loader (if provided)
        if test_df is not None:
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
    
    def get_user_embedding_mapping(self) -> Dict[str, int]:
        """Get mapping from original user IDs to embedding indices."""
        return self.user2idx.copy()
    
    def get_item_embedding_mapping(self) -> Dict[str, int]:
        """Get mapping from original item IDs to embedding indices."""
        return self.item2idx.copy()

