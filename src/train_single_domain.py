"""
Main experiment runner for single-domain recommendation.

This script trains a recommendation model on a single domain,
performs cross-validation, and evaluates the model.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from data_loader import RecommendationDataLoader
from models import get_model
from trainer import Trainer, CrossValidator
from evaluator import RecommendationEvaluator


def train_single_domain(domain: str = 'books', config_path: str = 'config.yaml',
                       use_cross_validation: bool = True):
    """
    Train a recommendation model on a single domain.
    
    Args:
        domain: Domain to train on ('books' or 'movies')
        config_path: Path to configuration file
        use_cross_validation: Whether to use cross-validation
    """
    # Load configuration
    config = Config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('training.device') == 'cuda' else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Select data path
    if domain == 'books':
        data_path = config.get('data.books_path')
    elif domain == 'movies':
        data_path = config.get('data.movies_path')
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    print(f"\n{'='*60}")
    print(f"Training Recommendation Model for {domain.upper()} domain")
    print(f"{'='*60}")
    
    # Load data
    data_loader = RecommendationDataLoader(
        data_path=data_path,
        test_ratio=config.get('data.test_ratio'),
        val_ratio=config.get('data.val_ratio'),
        random_seed=config.get('data.random_seed')
    )
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.get('logging.output_dir')) / f"{domain}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Model parameters
    model_kwargs = {
        'n_users': data_loader.n_users,
        'n_items': data_loader.n_items,
        'embedding_dim': config.get('model.embedding_dim'),
        'dropout': config.get('model.dropout', 0.0)
    }
    
    if use_cross_validation and config.get('cross_validation.enabled'):
        print(f"\n{'='*60}")
        print("Running Cross-Validation")
        print(f"{'='*60}")
        
        # Get K-fold splits
        n_folds = config.get('cross_validation.n_folds')
        fold_splits = data_loader.get_kfold_splits(n_folds=n_folds)
        
        # Create dataloaders for each fold
        fold_dataloaders = []
        for train_df, val_df in fold_splits:
            loaders = data_loader.create_dataloaders(
                train_df, val_df, test_df=None,
                batch_size=config.get('training.batch_size'),
                num_workers=0  # Use 0 for compatibility
            )
            fold_dataloaders.append((loaders['train'], loaders['val']))
        
        # Get model class
        model_name = config.get('model.name')
        if model_name == 'MatrixFactorization':
            from models import MatrixFactorization
            model_class = MatrixFactorization
        elif model_name == 'NeuralMF':
            from models import NeuralMatrixFactorization
            model_class = NeuralMatrixFactorization
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Run cross-validation
        cv = CrossValidator(
            model_class=model_class,
            model_kwargs=model_kwargs,
            optimizer_class=optim.Adam,
            optimizer_kwargs={'lr': config.get('model.learning_rate'), 
                            'weight_decay': config.get('model.weight_decay')},
            criterion=nn.MSELoss(),
            device=str(device),
            num_epochs=config.get('training.num_epochs'),
            early_stopping_patience=config.get('training.early_stopping_patience'),
            verbose=config.get('logging.verbose')
        )
        
        cv_results = cv.run(fold_dataloaders)
        
        # Save CV results
        cv_results_path = output_dir / 'cv_results.json'
        with open(cv_results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            cv_results_json = {
                'mean_val_loss': float(cv_results['mean_val_loss']),
                'std_val_loss': float(cv_results['std_val_loss']),
                'fold_results': [
                    {
                        'fold': r['fold'],
                        'best_val_loss': float(r['best_val_loss']),
                        'best_epoch': int(r['best_epoch'])
                    }
                    for r in cv_results['fold_results']
                ]
            }
            json.dump(cv_results_json, f, indent=2)
        print(f"\nCross-validation results saved to {cv_results_path}")
    
    # Train final model on full train set
    print(f"\n{'='*60}")
    print("Training Final Model")
    print(f"{'='*60}")
    
    train_df, val_df, test_df = data_loader.get_train_val_test_split()
    dataloaders = data_loader.create_dataloaders(
        train_df, val_df, test_df,
        batch_size=config.get('training.batch_size'),
        num_workers=0
    )
    
    # Create model
    model = get_model(
        config.get('model.name'),
        data_loader.n_users,
        data_loader.n_items,
        embedding_dim=config.get('model.embedding_dim'),
        dropout=config.get('model.dropout', 0.0)
    )
    
    # Create optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('model.learning_rate'),
        weight_decay=config.get('model.weight_decay')
    )
    criterion = nn.MSELoss()
    
    # Train
    trainer = Trainer(
        model, optimizer, criterion, str(device),
        early_stopping_patience=config.get('training.early_stopping_patience'),
        verbose=config.get('logging.verbose')
    )
    
    history = trainer.fit(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=config.get('training.num_epochs')
    )
    
    # Save model
    if config.get('logging.save_model'):
        model_path = output_dir / 'model.pt'
        trainer.save_model(str(model_path))
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}")
    
    evaluator = RecommendationEvaluator(model, str(device), k=10)
    test_metrics = evaluator.evaluate(
        dataloaders['test'],
        compute_ranking=False  # Set to True for ranking metrics (slower)
    )
    
    # Save test metrics
    test_metrics_path = output_dir / 'test_metrics.json'
    with open(test_metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    print(f"\nTest metrics saved to {test_metrics_path}")
    
    # Save user embeddings
    if config.get('logging.save_embeddings'):
        user_embeddings = model.get_user_embeddings().cpu().numpy()
        item_embeddings = model.get_item_embeddings().cpu().numpy()
        
        embeddings_path = output_dir / 'embeddings.npz'
        np.savez(
            embeddings_path,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            user2idx=data_loader.user2idx,
            item2idx=data_loader.item2idx
        )
        print(f"\nEmbeddings saved to {embeddings_path}")
        print(f"  User embeddings shape: {user_embeddings.shape}")
        print(f"  Item embeddings shape: {item_embeddings.shape}")
    
    # Save training history
    history_path = output_dir / 'training_history.npz'
    np.savez(
        history_path,
        train_losses=history['train_losses'],
        val_losses=history['val_losses']
    )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'model': model,
        'data_loader': data_loader,
        'test_metrics': test_metrics,
        'output_dir': output_dir
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train recommendation model on single domain')
    parser.add_argument('--domain', type=str, default='books', choices=['books', 'movies'],
                       help='Domain to train on')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-cv', action='store_true',
                       help='Disable cross-validation')
    
    args = parser.parse_args()
    
    results = train_single_domain(
        domain=args.domain,
        config_path=args.config,
        use_cross_validation=not args.no_cv
    )
    
    print("\n✓ Experiment completed successfully!")


if __name__ == '__main__':
    main()

