"""
Evaluate model with comprehensive ranking metrics for recommendations.
"""

import sys
import torch
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_loader import RecommendationDataLoader
from models import get_model
from evaluator import RecommendationEvaluator
from config import Config


def evaluate_with_ranking_metrics(domain: str = 'books', k_values: list = [5, 10, 20]):
    """
    Evaluate trained model with ranking metrics.
    
    Args:
        domain: Domain to evaluate ('books' or 'movies')
        k_values: List of k values for Top-K metrics
    """
    config = Config('config.yaml')
    
    # Find latest model
    output_dir = Path("outputs")
    domain_dirs = sorted(output_dir.glob(f"{domain}_*"))
    
    if not domain_dirs:
        print(f"No trained models found for {domain} domain.")
        print(f"Please train first: python src/train_single_domain.py --domain {domain}")
        return
    
    latest_dir = domain_dirs[-1]
    print(f"Evaluating model: {latest_dir}")
    
    # Load data
    if domain == 'books':
        data_path = config.get('data.books_path')
    else:
        data_path = config.get('data.movies_path')
    
    data_loader = RecommendationDataLoader(
        data_path=data_path,
        test_ratio=config.get('data.test_ratio'),
        val_ratio=config.get('data.val_ratio'),
        random_seed=config.get('data.random_seed')
    )
    
    # Get test data
    train_df, val_df, test_df = data_loader.get_train_val_test_split()
    dataloaders = data_loader.create_dataloaders(
        train_df, val_df, test_df,
        batch_size=config.get('training.batch_size'),
        num_workers=0
    )
    
    # Load model
    device = torch.device('cpu')
    model_path = latest_dir / 'model.pt'
    
    model = get_model(
        config.get('model.name'),
        n_users=data_loader.n_users,
        n_items=data_loader.n_items,
        embedding_dim=config.get('model.embedding_dim'),
        dropout=config.get('model.dropout', 0.0)
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n" + "="*60)
    print("RECOMMENDATION PERFORMANCE EVALUATION")
    print("="*60)
    
    # Evaluate with different k values
    all_results = {}
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Evaluating with K = {k}")
        print(f"{'='*60}")
        
        evaluator = RecommendationEvaluator(model, str(device), k=k)
        
        # Compute all metrics including ranking
        metrics = evaluator.evaluate(
            dataloaders['test'],
            compute_ranking=True,
            n_users=data_loader.n_users,
            n_items=data_loader.n_items
        )
        
        all_results[f'k={k}'] = metrics
    
    # Save comprehensive results
    results_path = latest_dir / 'ranking_metrics.json'
    with open(results_path, 'w') as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items()} 
                  for k, v in all_results.items()}, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_path}")
    
    # Print comparison table
    print(f"\n{'Metric':<20}", end='')
    for k in k_values:
        print(f"K={k:>2}", end='  ')
    print()
    print("-" * 60)
    
    metric_names = ['rmse', 'mae', 'ndcg', 'recall', 'precision']
    for metric in metric_names:
        print(f"{metric:<20}", end='')
        for k in k_values:
            key = f'k={k}'
            # Find the metric with @k suffix
            metric_key = None
            for mk in all_results[key].keys():
                if metric in mk.lower():
                    metric_key = mk
                    break
            
            if metric_key:
                value = all_results[key][metric_key]
                print(f"{value:>5.4f}", end='  ')
            else:
                # For metrics without @k (like rmse, mae)
                if metric in all_results[key]:
                    value = all_results[key][metric]
                    print(f"{value:>5.4f}", end='  ')
                else:
                    print(f"{'N/A':>5}", end='  ')
        print()
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
    RMSE/MAE: Lower is better (rating prediction error)
    NDCG@K:   Higher is better (0-1, ranking quality)
    Recall@K: Higher is better (0-1, coverage of relevant items)
    Precision@K: Higher is better (0-1, accuracy of recommendations)
    
    Typical good values for recommendation systems:
    - NDCG@10:   > 0.20 is decent, > 0.30 is good
    - Recall@10: > 0.10 is decent, > 0.20 is good
    - Precision@10: > 0.05 is decent, > 0.10 is good
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model with ranking metrics')
    parser.add_argument('--domain', type=str, default='books', choices=['books', 'movies'])
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20],
                       help='K values for Top-K metrics')
    
    args = parser.parse_args()
    
    evaluate_with_ranking_metrics(args.domain, args.k)

