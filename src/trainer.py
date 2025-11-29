"""
Training utilities for recommendation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Tuple
from tqdm import tqdm
import numpy as np
from pathlib import Path


class Trainer:
    """
    Trainer class for recommendation models.
    Handles training loop, validation, and early stopping.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 criterion: nn.Module, device: str = 'cpu',
                 early_stopping_patience: int = 5, verbose: bool = True):
        """
        Initialize Trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on ('cpu' or 'cuda')
            early_stopping_patience: Number of epochs to wait before early stopping
            verbose: Whether to print progress
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
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
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', disable=not self.verbose)
        for user_ids, item_ids, ratings in pbar:
            # Move to device
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)
            
            # Forward pass
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
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
        Validate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', disable=not self.verbose)
            for user_ids, item_ids, ratings in pbar:
                # Move to device
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                # Forward pass
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                
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
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs to train
        
        Returns:
            Dictionary containing training history
        """
        if self.verbose:
            print(f"\nTraining for up to {num_epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            if self.verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            if self.verbose:
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
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


class CrossValidator:
    """
    K-Fold Cross-Validation for recommendation models.
    """
    
    def __init__(self, model_class: type, model_kwargs: Dict,
                 optimizer_class: type, optimizer_kwargs: Dict,
                 criterion: nn.Module, device: str = 'cpu',
                 num_epochs: int = 50, early_stopping_patience: int = 5,
                 verbose: bool = True):
        """
        Initialize CrossValidator.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            optimizer_class: Optimizer class
            optimizer_kwargs: Keyword arguments for optimizer
            criterion: Loss function
            device: Device to train on
            num_epochs: Number of epochs per fold
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        
        self.fold_results = []
    
    def run(self, fold_dataloaders: list) -> Dict[str, float]:
        """
        Run cross-validation.
        
        Args:
            fold_dataloaders: List of (train_loader, val_loader) tuples for each fold
        
        Returns:
            Dictionary with mean and std of validation losses
        """
        n_folds = len(fold_dataloaders)
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running {n_folds}-Fold Cross-Validation")
            print(f"{'='*60}")
        
        for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Fold {fold_idx + 1}/{n_folds}")
                print(f"{'='*60}")
            
            # Create new model and trainer for each fold
            model = self.model_class(**self.model_kwargs)
            optimizer = self.optimizer_class(model.parameters(), **self.optimizer_kwargs)
            trainer = Trainer(
                model, optimizer, self.criterion, self.device,
                self.early_stopping_patience, self.verbose
            )
            
            # Train
            history = trainer.fit(train_loader, val_loader, self.num_epochs)
            
            # Store results
            self.fold_results.append({
                'fold': fold_idx,
                'best_val_loss': history['best_val_loss'],
                'best_epoch': history['best_epoch'],
                'train_losses': history['train_losses'],
                'val_losses': history['val_losses']
            })
        
        # Compute statistics
        val_losses = [result['best_val_loss'] for result in self.fold_results]
        mean_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Cross-Validation Results")
            print(f"{'='*60}")
            print(f"Mean Val Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
            print(f"Fold Val Losses: {[f'{loss:.4f}' for loss in val_losses]}")
        
        return {
            'mean_val_loss': mean_val_loss,
            'std_val_loss': std_val_loss,
            'fold_results': self.fold_results
        }

