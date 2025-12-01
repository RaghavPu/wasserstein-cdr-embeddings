"""
OT over user preference distributions (PMD-style)
=================================================

This module implements the full theoretical pipeline discussed:

1. For each domain (Books, Movies), load:
   - item_embeddings, user2idx, item2idx
   - raw interactions (user_id, item_id, rating)

2. For each user u in each domain D, build a discrete preference
   distribution over item embeddings:

      p_u^D = sum_{i in I_u^D} w_{ui}^D delta_{e_i^D}

   where weights w_{ui}^D are softmax(beta * rating_{ui}).

3. Define cross-domain user distance as 1-Wasserstein between
   p_u^A and p_v^B with ground cost c(i,j) = ||e_i^A - e_j^B||_2.

4. Build a user–user distance matrix and use it as the ground cost
   of an outer OT problem between user populations. This yields a
   Wasserstein-of-Wassersteins between domains.

5. Optionally, obtain:
   - per-user item-level OT plans (Monge maps),
   - user-population OT coupling Pi,
   - simple cross-domain recommendation via Pi + target ratings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import ot  # POT: pip install POT
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# Dataclass to represent a user's preference distribution
# ---------------------------------------------------------------------


@dataclass
class UserDistribution:
    """
    Discrete user preference distribution over item embeddings.

    Attributes
    ----------
    user_id : str
        Original user ID string from the dataset.
    user_idx : int
        Integer index of the user in user_embeddings (from user2idx).
    weights : np.ndarray
        Probability vector over the user's items, shape (m,).
    item_indices : np.ndarray
        Item indices (into item_embeddings) for this user, shape (m,).
    item_embeddings : np.ndarray
        Embeddings of those items, shape (m, embedding_dim).
    """

    user_id: str
    user_idx: int
    weights: np.ndarray
    item_indices: np.ndarray
    item_embeddings: np.ndarray


# ---------------------------------------------------------------------
# Embeddings & interactions loading
# ---------------------------------------------------------------------


def load_embeddings(
    embeddings_path: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Load embeddings from embeddings.npz for a single domain.

    Expected format:

        user_embeddings : (n_users, d)
        item_embeddings : (n_items, d)
        user2idx        : dict[raw_user_id -> int]
        item2idx        : dict[raw_item_id -> int]

    Parameters
    ----------
    embeddings_path : str
        Path to embeddings.npz for the domain.

    Returns
    -------
    user_embeddings, item_embeddings, user2idx, item2idx
    """
    data = np.load(embeddings_path, allow_pickle=True)
    user_embeddings = data["user_embeddings"]
    item_embeddings = data["item_embeddings"]
    user2idx = data["user2idx"].item()
    item2idx = data["item2idx"].item()
    return user_embeddings, item_embeddings, user2idx, item2idx


def load_interactions(
    interactions_path: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    sep: str = "\t",
) -> pd.DataFrame:
    """
    Load raw interaction data for a single domain.

    Assumes a TSV/CSV file with at least [user_col, item_col, rating_col].
    You can adapt user_col/item_col/rating_col/sep in the CLI to match your
    actual Amazon-KG .inter format.

    Parameters
    ----------
    interactions_path : str
        Path to interactions file for the domain.
    user_col, item_col, rating_col : str
        Column names for user IDs, item IDs, and ratings.
    sep : str
        Field separator (default '\\t' for TSV).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with exactly [user_col, item_col, rating_col].
    """
    p = Path(interactions_path)
    if not p.exists():
        raise FileNotFoundError(f"Interactions file not found: {interactions_path}")

    df = pd.read_csv(p, sep=sep)
    missing = {user_col, item_col, rating_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns {missing} in {interactions_path}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[[user_col, item_col, rating_col]].copy()


# ---------------------------------------------------------------------
# Build user distributions p_u = sum_i w_{ui} delta_{e_i}
# ---------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over a 1D numpy array.
    """
    x = x.astype(np.float64)
    x = x - x.max()
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0:
        return np.ones_like(ex) / len(ex)
    return ex / s


def build_user_distributions(
    interactions: pd.DataFrame,
    item_embeddings: np.ndarray,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    beta: float = 1.0,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
) -> Dict[str, UserDistribution]:
    """
    Construct discrete preference distributions p_u over item embeddings
    for each user in a domain.

    For each user u:
      - Aggregate their interactions by item (e.g., mean rating per item).
      - Compute weights via softmax(beta * rating).
      - Store item embeddings for those items.

    Parameters
    ----------
    interactions : pd.DataFrame
        Columns [user_col, item_col, rating_col].
    item_embeddings : np.ndarray
        Domain-specific item embeddings of shape (n_items, d).
    user2idx, item2idx : dict
        Raw ID -> embedding index mappings.
    beta : float
        Softmax temperature; larger beta focuses mass on higher ratings.
    user_col, item_col, rating_col : str
        Column names in `interactions`.

    Returns
    -------
    dists : Dict[str, UserDistribution]
        Mapping raw_user_id -> UserDistribution.
    """
    dists: Dict[str, UserDistribution] = {}

    grouped_users = interactions.groupby(user_col)

    for user_id, grp in tqdm(grouped_users, desc="Building user distributions"):
        if user_id not in user2idx:
            # User not present in embeddings (should be rare)
            continue

        # Aggregate ratings per item (mean rating)
        agg = grp.groupby(item_col)[rating_col].mean()

        item_ids = agg.index.to_numpy()
        ratings = agg.to_numpy(dtype=np.float64)

        item_indices: List[int] = []
        filtered_ratings: List[float] = []

        for it, r in zip(item_ids, ratings):
            it_str = str(it)
            if it_str not in item2idx:
                # Item not seen in the trained model; skip
                continue
            item_indices.append(item2idx[it_str])
            filtered_ratings.append(float(r))

        if not item_indices:
            # No usable items (all missing in item2idx)
            continue

        item_indices_arr = np.array(item_indices, dtype=np.int64)
        ratings_arr = np.array(filtered_ratings, dtype=np.float64)

        # Softmax over ratings (PMD-style)
        weights = _softmax(beta * ratings_arr)
        emb = item_embeddings[item_indices_arr]  # (m, d)

        dists[user_id] = UserDistribution(
            user_id=user_id,
            user_idx=user2idx[user_id],
            weights=weights,
            item_indices=item_indices_arr,
            item_embeddings=emb,
        )

    return dists


# ---------------------------------------------------------------------
# OT primitives: item-level cost and user-level Wasserstein distances
# ---------------------------------------------------------------------


def pairwise_cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean cost matrix C[i,j] = ||X[i] - Y[j]||_2.

    Parameters
    ----------
    X : (n, d)
    Y : (m, d)

    Returns
    -------
    C : (n, m)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Item embeddings must share the same dimension for cross-domain OT. "
            f"Got {X.shape[1]} (X) vs {Y.shape[1]} (Y). "
            f"Ensure embedding_dim is the same for both domains."
        )

    diff = X[:, None, :] - Y[None, :, :]
    return np.linalg.norm(diff, ord=2, axis=-1)


def wasserstein_user_distance(
    dist_u: UserDistribution,
    dist_v: UserDistribution,
    reg: float = 0.05,
) -> Tuple[float, np.ndarray]:
    """
    Compute entropic-regularized 1-Wasserstein distance between
    two user preference distributions p_u and p_v.

    Parameters
    ----------
    dist_u, dist_v : UserDistribution
        User distributions for two users (possibly from different domains).
    reg : float
        Entropic regularization parameter for Sinkhorn. Set >0 for speed.

    Returns
    -------
    wdist : float
        Wasserstein distance between p_u and p_v.
    gamma : np.ndarray
        Optimal transport plan between their item supports, shape (n_u, n_v).
    """
    a = dist_u.weights.astype(np.float64)
    b = dist_v.weights.astype(np.float64)
    a = a / a.sum()
    b = b / b.sum()

    C = pairwise_cost_matrix(dist_u.item_embeddings, dist_v.item_embeddings)  # (n_u, n_v)

    gamma = ot.sinkhorn(a, b, C, reg)
    wdist = float((gamma * C).sum())
    return wdist, gamma


# ---------------------------------------------------------------------
# User–user distance matrix and population-level OT
# ---------------------------------------------------------------------


def overlapping_user_ids(
    dists_a: Dict[str, UserDistribution],
    dists_b: Dict[str, UserDistribution],
) -> List[str]:
    """
    Return sorted list of user_ids present in both domain distributions.
    """
    return sorted(set(dists_a.keys()) & set(dists_b.keys()))


def cross_domain_user_distance_matrix(
    dists_a: Dict[str, UserDistribution],
    dists_b: Dict[str, UserDistribution],
    users_a: Optional[Iterable[str]] = None,
    users_b: Optional[Iterable[str]] = None,
    reg: float = 0.05,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute matrix D[u, v] = W1(p_u^A, p_v^B) for subsets of users.

    WARNING: Complexity is O(|users_a| * |users_b| * OT_cost), so
             you must usually subsample users for large datasets.

    Parameters
    ----------
    dists_a, dists_b : dict
        Mapping user_id -> UserDistribution for each domain.
    users_a, users_b : iterable of str, optional
        User IDs to include (rows for domain A, cols for domain B).
        If None, use all keys in dists_*.
    reg : float
        Entropic regularization for inner OT.

    Returns
    -------
    D : np.ndarray
        User distance matrix of shape (len(users_a_list), len(users_b_list)).
    users_a_list, users_b_list : List[str]
        Exact user_id order for rows/columns of D.
    """
    if users_a is None:
        users_a_list = sorted(dists_a.keys())
    else:
        users_a_list = list(users_a)

    if users_b is None:
        users_b_list = sorted(dists_b.keys())
    else:
        users_b_list = list(users_b)

    U = len(users_a_list)
    V = len(users_b_list)
    D = np.zeros((U, V), dtype=np.float64)

    for i, ua in enumerate(tqdm(users_a_list, desc="User distances A→B")):
        dist_u = dists_a[ua]
        for j, vb in enumerate(users_b_list):
            dist_v = dists_b[vb]
            d_uv, _ = wasserstein_user_distance(dist_u, dist_v, reg=reg)
            D[i, j] = d_uv

    return D, users_a_list, users_b_list


def user_population_ot(
    D: np.ndarray,
    reg: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """
    Compute OT coupling between user populations in domain A and B
    using user-level Wasserstein distances as ground cost.

    Parameters
    ----------
    D : np.ndarray
        User distance matrix, D[u, v] = W1(p_u^A, p_v^B), shape (U, V).
    reg : float
        Entropic regularization for outer OT.

    Returns
    -------
    Pi : np.ndarray
        Optimal user-population coupling, shape (U, V).
    total_cost : float
        Wasserstein distance between the two empirical user distributions.
    """
    U, V = D.shape
    a = np.ones(U, dtype=np.float64) / U
    b = np.ones(V, dtype=np.float64) / V

    Pi = ot.sinkhorn(a, b, D, reg)
    total_cost = float((Pi * D).sum())
    return Pi, total_cost


# ---------------------------------------------------------------------
# Cross-domain recommendation via user-population OT (optional)
# ---------------------------------------------------------------------


def build_rating_matrix(
    interactions: pd.DataFrame,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
) -> np.ndarray:
    """
    Build a dense user–item rating matrix for a domain.

    NOTE: For large datasets this may be big; restrict to subsets if needed.

    Parameters
    ----------
    interactions : pd.DataFrame
    user2idx, item2idx : dict
        Raw ID -> index mappings.
    user_col, item_col, rating_col : str
        Column names for interactions.

    Returns
    -------
    R : np.ndarray
        Rating matrix of shape (n_users, n_items), zero for missing entries.
    """
    n_users = len(user2idx)
    n_items = len(item2idx)
    R = np.zeros((n_users, n_items), dtype=np.float32)

    for _, row in interactions.iterrows():
        u = str(row[user_col])
        i = str(row[item_col])
        if u not in user2idx or i not in item2idx:
            continue
        ui = user2idx[u]
        ii = item2idx[i]
        R[ui, ii] = float(row[rating_col])

    return R


def predict_cross_domain_scores_from_Pi(
    user_id_source: str,
    users_source: List[str],
    users_target: List[str],
    Pi: np.ndarray,
    R_target: np.ndarray,
    user2idx_target: Dict[str, int],
    topk: int = 10,
    exclude_items: Optional[Iterable[int]] = None,
) -> List[Tuple[int, float]]:
    """
    Simple OT-based user-based CF:

    Given:
      - a user_id in source domain (e.g., Books),
      - the user-population coupling Pi between Books and Movies,
      - the rating matrix R_target in the target domain (Movies),

    we define scores over target items as:

        score[j] = sum_v Pi[u_idx_source, v_idx] * R_target[v_idx, j]

    Parameters
    ----------
    user_id_source : str
        Raw user_id in source domain (must be in users_source).
    users_source, users_target : list of str
        User id lists corresponding to rows/columns of Pi.
    Pi : np.ndarray
        User-population coupling, shape (len(users_source), len(users_target)).
    R_target : np.ndarray
        Dense rating matrix in target domain, shape (n_users_target_total, n_items_target).
    user2idx_target : dict
        Mapping raw target user_ids -> row indices in R_target.
    topk : int
        How many items to return.
    exclude_items : iterable of int, optional
        Item indices in target domain to mask out (already seen).

    Returns
    -------
    ranked_items : List[Tuple[int, float]]
        List of (item_idx, predicted_score) sorted by decreasing score.
    """
    if user_id_source not in users_source:
        raise ValueError(f"user_id_source={user_id_source} not in users_source.")

    # Map users_target list to indices in R_target
    target_indices = [user2idx_target[u] for u in users_target]

    # Row in Pi corresponding to user_id_source
    row_idx = users_source.index(user_id_source)
    weights = Pi[row_idx]  # (len(users_target),)

    R_sub = R_target[target_indices]  # (len(users_target), n_items)
    scores = weights @ R_sub  # (n_items,)

    if exclude_items is not None:
        mask = np.zeros_like(scores, dtype=bool)
        for idx in exclude_items:
            if 0 <= idx < len(scores):
                mask[idx] = True
        scores = scores.copy()
        scores[mask] = -np.inf

    k = min(topk, len(scores))
    top_idxs = np.argpartition(-scores, k - 1)[:k]
    top_idxs = top_idxs[np.argsort(-scores[top_idxs])]

    return [(int(i), float(scores[i])) for i in top_idxs]