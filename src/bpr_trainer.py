"""
BPR-specific trainer for recommendation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path


class BPRTrainer:
    """
    Trainer for BPR (Bayesian Personalized Ranking) models.
    Handles training loop with BPR loss and validation.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 device: str = 'cpu', early_stopping_patience: int = 5,
                 verbose: bool = True):
        """
        Initialize BPR Trainer.
        
        Args:
            model: BPR model to train
            optimizer: Optimizer for training
            device: Device to train on ('cpu' or 'cuda')
            early_stopping_patience: Number of epochs to wait before early stopping
            verbose: Whether to print progress
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch with BPR loss.
        
        Args:
            train_loader: BPR DataLoader with negative sampling
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', disable=not self.verbose)
        for batch in pbar:
            user_ids, pos_item_ids, neg_item_ids = batch
            
            # Move to device
            user_ids = user_ids.to(self.device)
            pos_item_ids = pos_item_ids.to(self.device)
            neg_item_ids = neg_item_ids.to(self.device)
            
            # Forward pass (model computes BPR loss internally)
            loss = self.model(user_ids, pos_item_ids, neg_item_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': total_loss / n_batches})
        
        return total_loss / n_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model (using MSE on actual ratings for monitoring).
        
        Args:
            val_loader: Regular DataLoader for validation
        
        Returns:
            Average validation loss (MSE for monitoring)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', disable=not self.verbose)
            for user_ids, item_ids, ratings in pbar:
                # Move to device
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                # Forward pass (use predict method for evaluation)
                predictions = self.model.predict(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                # Update metrics
                total_loss += loss.item()
                n_batches += 1
                
                pbar.set_postfix({'loss': total_loss / n_batches})
        
        return total_loss / n_batches
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            num_epochs: int) -> Dict[str, list]:
        """
        Train model for multiple epochs with early stopping.
        
        Args:
            train_loader: BPR DataLoader for training
            val_loader: Regular DataLoader for validation
            num_epochs: Maximum number of epochs to train
        
        Returns:
            Dictionary containing training history
        """
        if self.verbose:
            print(f"\nTraining BPR Model for up to {num_epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            if self.verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate (using MSE for monitoring convergence)
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            if self.verbose:
                print(f"Train Loss (BPR): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                # Save best model
                self.best_model_state = {
                    key: value.cpu().clone() for key, value in self.model.state_dict().items()
                }
                if self.verbose:
                    print(f"✓ New best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.verbose:
                    print(f"No improvement for {self.patience_counter} epoch(s)")
                
                if self.patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        print(f"Best model was at epoch {self.best_epoch + 1} with val_loss: {self.best_val_loss:.4f}")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print("\nRestored best model weights")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
    
    def save_model(self, path: str):
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, path)
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        if self.verbose:
            print(f"Model loaded from {path}")

