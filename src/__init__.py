"""Package initialization."""

__version__ = "1.0.0"

from .config import Config
from .data_loader import RecommendationDataLoader, InteractionDataset
from .models import MatrixFactorization, NeuralMatrixFactorization, get_model
from .trainer import Trainer, CrossValidator
from .evaluator import RecommendationEvaluator
from .utils import (
    load_embeddings,
    get_user_embedding,
    get_overlapping_users,
    extract_overlapping_embeddings,
    load_metrics,
    compare_results
)

# New OT imports
from .ot_wasserstein import OTConfig, OTWassersteinMapper, compute_ot_mapping_from_files

__all__ = [
    'Config',
    'RecommendationDataLoader',
    'InteractionDataset',
    'MatrixFactorization',
    'NeuralMatrixFactorization',
    'get_model',
    'Trainer',
    'CrossValidator',
    'RecommendationEvaluator',
    'load_embeddings',
    'get_user_embedding',
    'get_overlapping_users',
    'extract_overlapping_embeddings',
    'load_metrics',
    'compare_results',
    # OT
    "OTConfig",
    "OTWassersteinMapper",
    "compute_ot_mapping_from_files",
]

