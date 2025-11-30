"""
Matrix Factorization and other recommendation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MatrixFactorization(nn.Module):
    """
    Standard Matrix Factorization model for collaborative filtering.
    
    Learns latent representations for users and items by factorizing
    the user-item interaction matrix.
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 dropout: float = 0.0):
        """
        Initialize Matrix Factorization model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of user and item embeddings
            dropout: Dropout rate for regularization
        """
        super(MatrixFactorization, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict ratings.
        
        Args:
            user_ids: Tensor of user indices [batch_size]
            item_ids: Tensor of item indices [batch_size]
        
        Returns:
            Predicted ratings [batch_size]
        """
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embeddings(item_ids)  # [batch_size, embedding_dim]
        
        # Apply dropout
        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)
        
        # Compute dot product
        dot_product = (user_emb * item_emb).sum(dim=1)  # [batch_size]
        
        # Add biases
        user_b = self.user_bias(user_ids).squeeze()  # [batch_size]
        item_b = self.item_bias(item_ids).squeeze()  # [batch_size]
        
        # Final prediction
        prediction = dot_product + user_b + item_b + self.global_bias
        
        return prediction
    
    def get_user_embeddings(self) -> torch.Tensor:
        """
        Get all user embeddings.
        
        Returns:
            User embeddings [n_users, embedding_dim]
        """
        return self.user_embeddings.weight.data
    
    def get_item_embeddings(self) -> torch.Tensor:
        """
        Get all item embeddings.
        
        Returns:
            Item embeddings [n_items, embedding_dim]
        """
        return self.item_embeddings.weight.data
    
    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """
        Get embedding for a specific user.
        
        Args:
            user_id: User index
        
        Returns:
            User embedding [embedding_dim]
        """
        return self.user_embeddings.weight[user_id].data


class NeuralMatrixFactorization(nn.Module):
    """
    Neural Matrix Factorization (NeuMF) model.
    
    Combines Matrix Factorization with Multi-Layer Perceptron
    for more expressive power.
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_layers: Tuple[int, ...] = (128, 64, 32), dropout: float = 0.2):
        """
        Initialize Neural Matrix Factorization model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of embeddings
            hidden_layers: Tuple of hidden layer sizes for MLP
            dropout: Dropout rate
        """
        super(NeuralMatrixFactorization, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # GMF (Generalized Matrix Factorization) embeddings
        self.gmf_user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # MLP embeddings
        self.mlp_user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.output_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
        
        Returns:
            Predicted ratings
        """
        # GMF part
        gmf_user_emb = self.gmf_user_embeddings(user_ids)
        gmf_item_emb = self.gmf_item_embeddings(item_ids)
        gmf_output = gmf_user_emb * gmf_item_emb
        
        # MLP part
        mlp_user_emb = self.mlp_user_embeddings(user_ids)
        mlp_item_emb = self.mlp_item_embeddings(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate and predict
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.output_layer(combined).squeeze()
        
        return prediction
    
    def get_user_embeddings(self) -> torch.Tensor:
        """Get user embeddings (GMF embeddings)."""
        return self.gmf_user_embeddings.weight.data
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embeddings (GMF embeddings)."""
        return self.gmf_item_embeddings.weight.data


class BPRMatrixFactorization(nn.Module):
    """
    Matrix Factorization with BPR (Bayesian Personalized Ranking) loss.
    
    Designed for implicit feedback and ranking tasks.
    Learns by comparing positive items against negative samples.
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 dropout: float = 0.0):
        """
        Initialize BPR Matrix Factorization model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of user and item embeddings
            dropout: Dropout rate for regularization
        """
        super(BPRMatrixFactorization, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings (no biases for BPR - cleaner)
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with Xavier initialization."""
        nn.init.xavier_normal_(self.user_embeddings.weight)
        nn.init.xavier_normal_(self.item_embeddings.weight)
    
    def forward(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor,
                neg_item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BPR loss computation.
        
        Args:
            user_ids: Tensor of user indices [batch_size]
            pos_item_ids: Tensor of positive item indices [batch_size]
            neg_item_ids: Tensor of negative item indices [batch_size, n_negatives]
        
        Returns:
            BPR loss value
        """
        # Get user embeddings
        user_emb = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        user_emb = self.dropout(user_emb)
        
        # Get positive item embeddings
        pos_item_emb = self.item_embeddings(pos_item_ids)  # [batch_size, embedding_dim]
        pos_item_emb = self.dropout(pos_item_emb)
        
        # Get negative item embeddings
        neg_item_emb = self.item_embeddings(neg_item_ids)  # [batch_size, n_neg, embedding_dim]
        neg_item_emb = self.dropout(neg_item_emb)
        
        # Compute scores
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # [batch_size]
        
        # For negative items: [batch_size, n_neg, emb_dim] * [batch_size, 1, emb_dim]
        user_emb_expanded = user_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        neg_scores = (user_emb_expanded * neg_item_emb).sum(dim=2)  # [batch_size, n_neg]
        
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        # For multiple negatives, average over them
        pos_scores_expanded = pos_scores.unsqueeze(1)  # [batch_size, 1]
        diff = pos_scores_expanded - neg_scores  # [batch_size, n_neg]
        
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        
        return loss
    
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for user-item pairs (for evaluation).
        
        Args:
            user_ids: Tensor of user indices [batch_size]
            item_ids: Tensor of item indices [batch_size]
        
        Returns:
            Predicted scores [batch_size]
        """
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        scores = (user_emb * item_emb).sum(dim=1)
        return scores
    
    def get_user_embeddings(self) -> torch.Tensor:
        """Get all user embeddings."""
        return self.user_embeddings.weight.data
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get all item embeddings."""
        return self.item_embeddings.weight.data
    
    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """Get embedding for a specific user."""
        return self.user_embeddings.weight[user_id].data


def get_model(model_name: str, n_users: int, n_items: int, **kwargs) -> nn.Module:
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of the model ('MatrixFactorization', 'NeuralMF', or 'BPR')
        n_users: Number of users
        n_items: Number of items
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance
    """
    models = {
        'MatrixFactorization': MatrixFactorization,
        'NeuralMF': NeuralMatrixFactorization,
        'BPR': BPRMatrixFactorization,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](n_users, n_items, **kwargs)

