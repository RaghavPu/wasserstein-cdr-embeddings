"""
Evaluation metrics for recommendation systems.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm


class RecommendationEvaluator:
    """
    Evaluator for recommendation models.
    Computes various metrics including RMSE, MAE, NDCG, Recall, and Precision.
    Supports item filtering for fair evaluation on large catalogs.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu', k: int = 10,
                 min_item_interactions: int = 0, valid_items: Optional[np.ndarray] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained recommendation model
            device: Device to run evaluation on
            k: Top-k for ranking metrics (NDCG@k, Recall@k, Precision@k)
            min_item_interactions: Minimum interactions for item to be considered (filtering)
            valid_items: Optional array of valid item indices to consider
        """
        self.model = model.to(device)
        self.device = device
        self.k = k
        self.min_item_interactions = min_item_interactions
        self.valid_items = valid_items
        self.model.eval()
        
        # Check if model has predict method (BPR) or use forward
        self.is_bpr = hasattr(model, 'predict')
    
    def get_predictions(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and ground truth from dataloader.
        
        Args:
            dataloader: DataLoader with test data
        
        Returns:
            Tuple of (predictions, ground_truth)
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in tqdm(dataloader, desc='Getting predictions'):
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                
                # Use appropriate prediction method
                if self.is_bpr:
                    predictions = self.model.predict(user_ids, item_ids)
                else:
                    predictions = self.model(user_ids, item_ids)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(ratings.numpy())
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        return predictions, targets
    
    def compute_rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Root Mean Squared Error.
        
        Args:
            predictions: Predicted ratings
            targets: Ground truth ratings
        
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(targets, predictions))
    
    def compute_mae(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Mean Absolute Error.
        
        Args:
            predictions: Predicted ratings
            targets: Ground truth ratings
        
        Returns:
            MAE value
        """
        return mean_absolute_error(targets, predictions)
    
    def compute_ranking_metrics(self, dataloader: DataLoader, 
                                n_users: int, n_items: int,
                                item_counts: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute ranking metrics (NDCG@k, Recall@k, Precision@k).
        With optional item filtering for fair evaluation.
        
        Args:
            dataloader: DataLoader with test data
            n_users: Total number of users
            n_items: Total number of items
            item_counts: Array of interaction counts per item (for filtering)
        
        Returns:
            Dictionary with ranking metrics
        """
        # Determine valid items for evaluation
        if item_counts is not None and self.min_item_interactions > 0:
            valid_items = np.where(item_counts >= self.min_item_interactions)[0]
            print(f"Item filtering: {len(valid_items)}/{n_items} items "
                  f"(≥{self.min_item_interactions} interactions)")
        elif self.valid_items is not None:
            valid_items = self.valid_items
            print(f"Item filtering: {len(valid_items)}/{n_items} items (custom filter)")
        else:
            valid_items = np.arange(n_items)
            print(f"No item filtering: evaluating on all {n_items} items")
        
        # Collect user-item interactions
        user_items = {}  # user_id -> list of (item_id, rating)
        
        for user_ids, item_ids, ratings in dataloader:
            for u, i, r in zip(user_ids.numpy(), item_ids.numpy(), ratings.numpy()):
                if u not in user_items:
                    user_items[u] = []
                user_items[u].append((i, r))
        
        ndcg_scores = []
        recall_scores = []
        precision_scores = []
        
        with torch.no_grad():
            for user_id, items in tqdm(user_items.items(), desc='Computing ranking metrics'):
                # Only rank valid items
                item_ids_tensor = torch.tensor(valid_items, dtype=torch.long).to(self.device)
                user_ids_tensor = torch.full((len(valid_items),), user_id, 
                                            dtype=torch.long).to(self.device)
                
                # Get predictions
                if self.is_bpr:
                    predictions = self.model.predict(user_ids_tensor, item_ids_tensor).cpu().numpy()
                else:
                    predictions = self.model(user_ids_tensor, item_ids_tensor).cpu().numpy()
                
                # Get top-k items
                top_k_indices = np.argsort(predictions)[-self.k:][::-1]
                top_k_items = valid_items[top_k_indices]
                
                # Get ground truth items (items user actually interacted with)
                ground_truth_items = set([i for i, r in items if r > 0])
                
                if len(ground_truth_items) == 0:
                    continue
                
                # Compute metrics
                hits = [1 if item in ground_truth_items else 0 for item in top_k_items]
                
                # NDCG@k
                dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
                idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(ground_truth_items), self.k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
                
                # Recall@k
                recall = sum(hits) / len(ground_truth_items)
                recall_scores.append(recall)
                
                # Precision@k
                precision = sum(hits) / self.k
                precision_scores.append(precision)
        
        return {
            f'ndcg@{self.k}': np.mean(ndcg_scores),
            f'recall@{self.k}': np.mean(recall_scores),
            f'precision@{self.k}': np.mean(precision_scores)
        }
    
    def evaluate(self, dataloader: DataLoader, 
                 compute_ranking: bool = False,
                 n_users: Optional[int] = None,
                 n_items: Optional[int] = None,
                 item_counts: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            dataloader: DataLoader with test data
            compute_ranking: Whether to compute ranking metrics (slower)
            n_users: Total number of users (required if compute_ranking=True)
            n_items: Total number of items (required if compute_ranking=True)
        
        Returns:
            Dictionary with all metrics
        """
        print("\nEvaluating model...")
        
        # Get predictions
        predictions, targets = self.get_predictions(dataloader)
        
        # Compute rating prediction metrics
        metrics = {
            'rmse': self.compute_rmse(predictions, targets),
            'mae': self.compute_mae(predictions, targets)
        }
        
        # Compute ranking metrics if requested
        if compute_ranking:
            if n_users is None or n_items is None:
                raise ValueError("n_users and n_items must be provided for ranking metrics")
            ranking_metrics = self.compute_ranking_metrics(dataloader, n_users, n_items, item_counts)
            metrics.update(ranking_metrics)
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 40)
        for metric_name, value in metrics.items():
            print(f"{metric_name:20s}: {value:.4f}")
        print("-" * 40)
        
        return metrics
    
    def get_user_recommendations(self, user_id: int, n_items: int, 
                                 top_k: int = 10, 
                                 exclude_items: Optional[List[int]] = None) -> np.ndarray:
        """
        Get top-k item recommendations for a user.
        
        Args:
            user_id: User ID
            n_items: Total number of items
            top_k: Number of recommendations to return
            exclude_items: List of items to exclude from recommendations
        
        Returns:
            Array of top-k item IDs
        """
        self.model.eval()
        
        with torch.no_grad():
            # Predict for all items
            item_ids = torch.arange(n_items).to(self.device)
            user_ids = torch.full((n_items,), user_id, dtype=torch.long).to(self.device)
            
            predictions = self.model(user_ids, item_ids).cpu().numpy()
            
            # Exclude items if specified
            if exclude_items is not None:
                predictions[exclude_items] = -np.inf
            
            # Get top-k
            top_k_items = np.argsort(predictions)[-top_k:][::-1]
        
        return top_k_items


def evaluate_multiple_metrics(model: nn.Module, dataloader: DataLoader,
                              device: str = 'cpu', k_values: List[int] = [5, 10, 20],
                              compute_ranking: bool = False,
                              n_users: Optional[int] = None,
                              n_items: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model with multiple k values for ranking metrics.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device for evaluation
        k_values: List of k values for ranking metrics
        compute_ranking: Whether to compute ranking metrics
        n_users: Total number of users
        n_items: Total number of items
    
    Returns:
        Dictionary mapping k values to metric dictionaries
    """
    results = {}
    
    # Compute basic metrics once
    evaluator = RecommendationEvaluator(model, device, k=k_values[0])
    predictions, targets = evaluator.get_predictions(dataloader)
    
    base_metrics = {
        'rmse': evaluator.compute_rmse(predictions, targets),
        'mae': evaluator.compute_mae(predictions, targets)
    }
    
    # Compute ranking metrics for each k
    if compute_ranking:
        for k in k_values:
            evaluator.k = k
            if n_users is not None and n_items is not None:
                ranking_metrics = evaluator.compute_ranking_metrics(dataloader, n_users, n_items)
                results[f'k={k}'] = {**base_metrics, **ranking_metrics}
            else:
                results[f'k={k}'] = base_metrics
    else:
        results['metrics'] = base_metrics
    
    return results

