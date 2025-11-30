"""
OT-based cross-domain user mapping.

This module implements:
  - A user distance metric between Books and Movies user embeddings
  - A Wasserstein (optimal transport) coupling between user distributions
  - A barycentric projection map f: books_emb -> movies_emb
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import ot  # POT: pip install POT

from utils import extract_overlapping_embeddings

# ---------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------

UserDistance = Literal["euclidean", "cosine", "mahalanobis"]


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """
    L2-normalize vectors along a given axis.

    Args:
        x: Array of shape (n, d) or similar.
        axis: Axis along which to compute norms.
        eps: Small constant to avoid division by zero.

    Returns:
        L2-normalized array with same shape as x.
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / norm


def _compute_cost_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    metric: UserDistance = "euclidean",
) -> np.ndarray:
    """
    Compute the pairwise cost matrix between two sets of user embeddings.

    This implements the user distance metric we decided to use:
      - "euclidean": squared L2 distance on L2-normalized embeddings
      - "cosine": cosine distance = 1 - cosine similarity
      - "mahalanobis": squared L2 distance in a whitened space

    Args:
        X: Array of shape (n_src, d) - Books user embeddings.
        Y: Array of shape (n_tgt, d) - Movies user embeddings.
        metric: Type of user distance to use.

    Returns:
        Cost matrix C of shape (n_src, n_tgt) where C[i, j] is
        the cost between user i in Books and user j in Movies.
    """
    if metric == "euclidean":
        # Normalize first to remove scale mismatch between domains
        Xn = _l2_normalize(X)
        Yn = _l2_normalize(Y)
        M = ot.dist(Xn, Yn, metric="euclidean")  # (n_src, n_tgt)
        return M ** 2

    if metric == "cosine":
        Xn = _l2_normalize(X)
        Yn = _l2_normalize(Y)
        # Cosine similarity, then convert to a distance / cost
        sim = Xn @ Yn.T  # (n_src, n_tgt)
        return 1.0 - sim  # cost = 1 - cos

    if metric == "mahalanobis":
        # Fit a global covariance over concatenated data, then whiten
        Z = np.vstack([X, Y])
        mean = Z.mean(axis=0, keepdims=True)
        Zc = Z - mean
        cov = (Zc.T @ Zc) / max(Zc.shape[0] - 1, 1)
        # Regularize to ensure positive-definiteness
        reg = 1e-4 * np.eye(cov.shape[0], dtype=cov.dtype)
        cov = cov + reg
        L = np.linalg.cholesky(cov)  # cov = L L^T
        Linv = np.linalg.inv(L)

        Xw = (Linv @ (X - mean).T).T
        Yw = (Linv @ (Y - mean).T).T

        M = ot.dist(Xw, Yw, metric="euclidean")
        return M ** 2

    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------
# OT configuration and mapper
# ---------------------------------------------------------------------


@dataclass
class OTConfig:
    """
    Hyperparameters for the Wasserstein user mapping.

    Attributes:
        metric: User distance metric for the ground cost.
        reg: Entropic regularization parameter (Sinkhorn).
        sinkhorn: If True, use Sinkhorn; otherwise, use exact EMD.
    """

    metric: UserDistance = "euclidean"
    reg: float = 0.05
    sinkhorn: bool = True


class OTWassersteinMapper:
    """
    Fit an optimal transport plan between Books and Movies user embeddings
    and provide a barycentric projection mapping f: books_emb -> movies_emb.
    """

    def __init__(self, config: OTConfig):
        self.config = config

        # Learned quantities
        self.books_emb_: Optional[np.ndarray] = None
        self.movies_emb_: Optional[np.ndarray] = None
        self.cost_matrix_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None  # transport plan (n_books, n_movies)

    # ----------------------------- core fitting -----------------------------

    def fit(self, books_emb: np.ndarray, movies_emb: np.ndarray) -> "OTWassersteinMapper":
        """
        Fit the OT coupling between Books and Movies user distributions.

        Args:
            books_emb: (n_books, d) embeddings for overlapping users in Books.
            movies_emb: (n_movies, d) embeddings for same overlapping users in Movies.

        Returns:
            self
        """
        books_emb = np.asarray(books_emb, dtype=np.float64)
        movies_emb = np.asarray(movies_emb, dtype=np.float64)

        if books_emb.shape[0] == 0 or movies_emb.shape[0] == 0:
            raise ValueError("Empty embeddings passed to OTWassersteinMapper.fit().")

        n_src, n_tgt = books_emb.shape[0], movies_emb.shape[0]

        self.books_emb_ = books_emb
        self.movies_emb_ = movies_emb

        # Uniform weights over users in each domain
        a = np.ones(n_src, dtype=np.float64) / n_src
        b = np.ones(n_tgt, dtype=np.float64) / n_tgt

        # Ground cost between users
        C = _compute_cost_matrix(books_emb, movies_emb, metric=self.config.metric)
        self.cost_matrix_ = C

        # Solve OT
        if self.config.sinkhorn and self.config.reg > 0:
            gamma = ot.sinkhorn(a, b, C, reg=self.config.reg)
        else:
            gamma = ot.emd(a, b, C)

        self.gamma_ = gamma
        return self

    # ----------------------------- mapping ----------------------------------

    def barycentric_projection(self) -> np.ndarray:
        """
        Compute barycentric projection of Books embeddings into Movies space.

        For each Books user i, we define:

            f(b_i) = sum_j gamma[i, j] * movies_emb[j] / sum_j gamma[i, j]

        Returns:
            mapped: (n_books, d) array of predicted Movies-domain embeddings.
        """
        if self.gamma_ is None or self.movies_emb_ is None:
            raise RuntimeError("Call fit() before barycentric_projection().")

        weights = self.gamma_.sum(axis=1, keepdims=True) + 1e-8
        mapped = self.gamma_ @ self.movies_emb_ / weights
        return mapped

    # ----------------------------- evaluation -------------------------------

    def evaluate_on_overlap(self) -> Dict[str, float]:
        """
        Evaluate mapping on overlapping users using embedding-level metrics.

        We assume the Books and Movies embeddings are ordered so that
        books_emb[i] and movies_emb[i] correspond to the same user ID
        (as produced by utils.extract_overlapping_embeddings).

        Metrics:
            - mse: MSE between mapped Movies embedding and true Movies embedding
            - mae: MAE between mapped and true
            - mean_cosine: mean cosine similarity between mapped and true

        Returns:
            Dictionary of metrics.
        """
        if self.movies_emb_ is None:
            raise RuntimeError("Call fit() before evaluate_on_overlap().")

        mapped = self.barycentric_projection()
        target = self.movies_emb_

        mse = float(np.mean((mapped - target) ** 2))
        mae = float(np.mean(np.abs(mapped - target)))

        mapped_n = _l2_normalize(mapped)
        target_n = _l2_normalize(target)
        cos = np.sum(mapped_n * target_n, axis=1)
        mean_cos = float(np.mean(cos))

        return {
            "mse": mse,
            "mae": mae,
            "mean_cosine": mean_cos,
        }


# ---------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------


def compute_ot_mapping_from_files(
    books_embeddings_path: str,
    movies_embeddings_path: str,
    metric: UserDistance = "euclidean",
    reg: float = 0.05,
    sinkhorn: bool = True,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    High-level wrapper:
        1. Load overlapping embeddings from both domains
        2. Fit OT mapper
        3. Compute barycentric mapping
        4. Optionally save mapping artifacts

    Args:
        books_embeddings_path: Path to Books embeddings.npz.
        movies_embeddings_path: Path to Movies embeddings.npz.
        metric: User distance metric ("euclidean", "cosine", "mahalanobis").
        reg: Entropic regularization parameter for Sinkhorn.
        sinkhorn: If True, use Sinkhorn; otherwise, exact EMD.
        save_dir: If provided, directory to save mapping and metrics.

    Returns:
        Dictionary with:
            - "mapped_movies_embeddings": mapped Movies embeddings for each Books user
            - "metrics": mapping evaluation metrics
            - "overlapping_user_ids": list of overlapping user IDs
            - "books_embeddings": overlapping Books embeddings
            - "movies_embeddings": overlapping Movies embeddings
            - "ot_config": dict of config parameters
    """
    # Extract overlapping user embeddings (uses your teammate's utils)
    result = extract_overlapping_embeddings(
        books_embeddings_path,
        movies_embeddings_path,
        output_path=None,  # don't overwrite their npz unless you want to
    )

    books_emb = result["books_embeddings"]
    movies_emb = result["movies_embeddings"]
    user_ids = result["overlapping_user_ids"]

    config = OTConfig(metric=metric, reg=reg, sinkhorn=sinkhorn)
    mapper = OTWassersteinMapper(config).fit(books_emb, movies_emb)

    mapped = mapper.barycentric_projection()
    metrics = mapper.evaluate_on_overlap()

    output: Dict = {
        "mapped_movies_embeddings": mapped,
        "metrics": metrics,
        "overlapping_user_ids": user_ids,
        "books_embeddings": books_emb,
        "movies_embeddings": movies_emb,
        "ot_config": {
            "metric": metric,
            "reg": reg,
            "sinkhorn": sinkhorn,
        },
    }

    # Optional: save everything for later analysis
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save numeric artifacts
        np.savez(
            save_path / "ot_mapping.npz",
            overlapping_user_ids=np.array(user_ids, dtype=object),
            mapped_embeddings=mapped,
            books_embeddings=books_emb,
            movies_embeddings=movies_emb,
            cost_matrix=mapper.cost_matrix_,
            transport_plan=mapper.gamma_,
        )

        # Save metrics
        import json

        with open(save_path / "ot_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[OT] Saved mapping artifacts under: {save_path}")

    return output