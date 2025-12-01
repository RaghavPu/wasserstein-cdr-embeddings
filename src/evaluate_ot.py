"""
Evaluate OT-based cross-domain recommendation vs random and popularity baselines,
and also check embedding-level OT alignment quality via cosine similarity.

Assumes you have already:

1. Trained single-domain models:
     python src/train_single_domain.py --domain books
     python src/train_single_domain.py --domain movies

2. Run the distribution-based OT pipeline:
     python src/run_ot_distributions.py \
         --books-interactions  data/Amazon-Dense-Subset/Amazon-Books-Dense.inter \
         --movies-interactions data/Amazon-Dense-Subset/Amazon-Movies-Dense.inter \
         --output outputs/ot_distributions.npz

This script then:

  - Loads OT artifacts (Pi, books_users, movies_users)
  - Loads Movies embeddings & interactions
  - For overlapping users (Books+Movies, and present in Pi) compares:

      (a) OT-based cross-domain CF:
          scores_OT[j] = sum_v Pi[u, v] * R_movies[v, j]

      (b) Random baseline:
          scores_RND[j] ~ Unif(0, 1)

      (c) Popularity baseline:
          scores_POP[j] = global popularity(item j)  (same for all users)

    and computes Recall@K / Precision@K for each.

  - Additionally runs embedding-level OT (using ot_wasserstein.py) and
    prints MSE, MAE, and mean cosine similarity between mapped Movies
    embeddings and true Movies embeddings for overlapping users.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Make local src imports work when running as a script
sys.path.append(str(Path(__file__).parent))

from ot_distributions import (  # noqa: E402
    build_rating_matrix,
    load_embeddings,
    load_interactions,
)
from ot_wasserstein import compute_ot_mapping_from_files  # noqa: E402


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


def _find_latest_embeddings(domain: str) -> str:
    """
    Find the most recent embeddings.npz for a given domain (books or movies).
    """
    output_dir = Path("outputs")
    dirs = sorted(output_dir.glob(f"{domain}_*"))

    if not dirs:
        raise FileNotFoundError(
            f"No {domain} outputs found in {output_dir}. "
            f"Run `python src/train_single_domain.py --domain {domain}` first."
        )

    latest_dir = dirs[-1]
    path = latest_dir / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return str(path)


def _resolve_embeddings_input(path_or_pattern: str) -> str:
    """
    Resolve a user-specified embeddings location into a concrete .npz file.

    Accepts:
      - exact .npz file
      - a directory containing 'embeddings.npz'
      - a glob pattern that matches files/directories
    """
    p = Path(path_or_pattern)

    if p.is_dir():
        candidate = p / "embeddings.npz"
        if not candidate.exists():
            raise FileNotFoundError(
                f"{path_or_pattern} is a directory but no 'embeddings.npz' inside."
            )
        return str(candidate)

    if p.exists():
        return str(p)

    matches = sorted(Path().glob(path_or_pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files or directories match pattern: '{path_or_pattern}'"
        )
    last = matches[-1]
    if last.is_dir():
        candidate = last / "embeddings.npz"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Pattern '{path_or_pattern}' matched directory '{last}', "
                f"but no 'embeddings.npz' inside."
            )
        return str(candidate)
    return str(last)


def _recall_precision_at_k(
    scores: np.ndarray,
    positive_indices: Iterable[int],
    k: int,
) -> Tuple[float, float]:
    """
    Compute Recall@K and Precision@K given scores over all items
    and a set of positive item indices.

    Args:
        scores: array of shape (n_items,)
        positive_indices: iterable of item indices considered positives
        k: top-k cutoff

    Returns:
        (recall_at_k, precision_at_k)
    """
    positives = set(int(i) for i in positive_indices)
    if not positives:
        return 0.0, 0.0

    k = min(k, scores.shape[0])
    # indices of top-k scores
    topk_idx = np.argpartition(-scores, k - 1)[:k]
    topk_idx = topk_idx[np.argsort(-scores[topk_idx])]

    hits = len(positives.intersection(topk_idx.tolist()))
    recall = hits / float(len(positives))
    precision = hits / float(k)
    return recall, precision


# ---------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OT-based cross-domain recommendation vs random and "
            "popularity baselines, and check embedding-level OT alignment."
        )
    )

    # Embeddings
    parser.add_argument(
        "--books-embeddings",
        type=str,
        default=None,
        help="Books embeddings location (npz / dir / glob). If omitted, uses latest outputs/books_*/embeddings.npz.",
    )
    parser.add_argument(
        "--movies-embeddings",
        type=str,
        default=None,
        help="Movies embeddings location (npz / dir / glob). If omitted, uses latest outputs/movies_*/embeddings.npz.",
    )

    # Interactions
    parser.add_argument(
        "--books-interactions",
        type=str,
        required=True,
        help="Path to Books interactions file (.inter / .tsv / .csv).",
    )
    parser.add_argument(
        "--movies-interactions",
        type=str,
        required=True,
        help="Path to Movies interactions file (.inter / .tsv / .csv).",
    )

    # Column names & separator
    parser.add_argument("--user-col", type=str, default="user_id", help="User ID column name.")
    parser.add_argument("--item-col", type=str, default="item_id", help="Item ID column name.")
    parser.add_argument("--rating-col", type=str, default="rating", help="Rating column name.")
    parser.add_argument("--sep", type=str, default="\t", help="Field separator (default: '\\t').")

    # OT distribution results
    parser.add_argument(
        "--ot-results",
        type=str,
        default="outputs/ot_distributions.npz",
        help="Path to OT results file produced by run_ot_distributions.py.",
    )

    # Evaluation settings
    parser.add_argument(
        "--Ks",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="List of K values for Recall@K / Precision@K.",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=3,
        help="Minimum number of Movies positives required to include a user in evaluation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the random baseline.",
    )

    # Embedding-level OT diagnostics
    parser.add_argument(
        "--embedding-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "mahalanobis"],
        help="Metric for embedding-level OT (ot_wasserstein).",
    )
    parser.add_argument(
        "--embedding-reg",
        type=float,
        default=0.05,
        help="Entropic regularization for embedding-level Sinkhorn.",
    )
    parser.add_argument(
        "--embedding-no-sinkhorn",
        action="store_true",
        help="If set, use exact EMD for embedding-level OT instead of Sinkhorn.",
    )

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    # Resolve embeddings paths
    if args.books_embeddings is None:
        books_emb_path = _find_latest_embeddings("books")
    else:
        books_emb_path = _resolve_embeddings_input(args.books_embeddings)

    if args.movies_embeddings is None:
        movies_emb_path = _find_latest_embeddings("movies")
    else:
        movies_emb_path = _resolve_embeddings_input(args.movies_embeddings)

    print("=" * 70)
    print("Evaluate OT vs random/popularity + embedding-level OT diagnostics")
    print("=" * 70)
    print(f"Books embeddings:   {books_emb_path}")
    print(f"Movies embeddings:  {movies_emb_path}")
    print(f"Books interactions: {args.books_interactions}")
    print(f"Movies interactions:{args.movies_interactions}")
    print(f"OT results:         {args.ot_results}")
    print(f"Ks:                 {args.Ks}")
    print(f"min_positives:      {args.min_positives}")
    print(f"random_seed:        {args.random_seed}")

    # ------------------------------------------------------------------
    # Load OT artifacts from distribution-based pipeline
    # ------------------------------------------------------------------
    ot_data = np.load(args.ot_results, allow_pickle=True)
    Pi = ot_data["Pi"]                             # (U, V)
    books_users = ot_data["books_users"].tolist() # list[str], len = U
    movies_users = ot_data["movies_users"].tolist()  # list[str], len = V

    # Map user IDs to indices in Pi
    books_idx_in_Pi: Dict[str, int] = {u: i for i, u in enumerate(books_users)}
    movies_idx_in_Pi: Dict[str, int] = {u: j for j, u in enumerate(movies_users)}

    # ------------------------------------------------------------------
    # Load embeddings & interactions
    # ------------------------------------------------------------------
    (
        books_user_emb,
        books_item_emb,
        books_user2idx,
        books_item2idx,
    ) = load_embeddings(books_emb_path)

    (
        movies_user_emb,
        movies_item_emb,
        movies_user2idx,
        movies_item2idx,
    ) = load_embeddings(movies_emb_path)

    books_inter = load_interactions(
        args.books_interactions,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        sep=args.sep,
    )
    movies_inter = load_interactions(
        args.movies_interactions,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        sep=args.sep,
    )

    # Build Movies rating matrix R_movies (n_movies_users_total x n_items_movies)
    print("\n[Eval] Building Movies rating matrix (dense)...")
    R_movies = build_rating_matrix(
        movies_inter,
        user2idx=movies_user2idx,
        item2idx=movies_item2idx,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
    )

    # ------------------------------------------------------------------
    # Determine evaluation users (overlap in interactions + present in Pi)
    # ------------------------------------------------------------------
    books_users_set = set(books_inter[args.user_col].astype(str).unique())
    movies_users_set = set(movies_inter[args.user_col].astype(str).unique())

    interaction_overlap = books_users_set & movies_users_set

    eval_users: List[str] = []
    for u in interaction_overlap:
        if u not in books_idx_in_Pi:
            continue
        if u not in movies_user2idx:
            continue

        # Check that user has enough Movies positives
        user_movies_items = movies_inter.loc[
            movies_inter[args.user_col].astype(str) == u, args.item_col
        ]
        pos_indices = [
            movies_item2idx[str(it)]
            for it in user_movies_items
            if str(it) in movies_item2idx
        ]
        if len(set(pos_indices)) < args.min_positives:
            continue

        eval_users.append(u)

    if not eval_users:
        print("\n[Eval] No users meet overlap + in Pi + min_positives criteria. Nothing to evaluate.")
    else:
        print(f"\n[Eval] Number of eval users: {len(eval_users)}")

    # ------------------------------------------------------------------
    # Prepare OT scores, random baseline, popularity baseline
    # ------------------------------------------------------------------
    # Filter Movies users in Pi to those that exist in movies_user2idx
    movies_users_valid = [u for u in movies_users if u in movies_user2idx]
    valid_cols_mask = np.array([u in movies_user2idx for u in movies_users], dtype=bool)
    Pi_valid = Pi[:, valid_cols_mask]  # (U, V_valid)

    # Mapping from movies_users_valid -> index in R_movies
    movies_users_indices_in_R = [movies_user2idx[u] for u in movies_users_valid]
    R_movies_sub = R_movies[movies_users_indices_in_R]  # (V_valid, n_items_movies)

    # Popularity scores: global popularity per item (same for every user)
    popularity_counts = np.zeros(movies_item_emb.shape[0], dtype=np.float64)
    for _, row in movies_inter.iterrows():
        it = str(row[args.item_col])
        if it in movies_item2idx:
            ii = movies_item2idx[it]
            popularity_counts[ii] += 1.0

    # Set Ks
    Ks = sorted(set(k for k in args.Ks if k > 0))
    if not Ks:
        print("[Eval] No positive K values provided.")
        Ks = []

    # metrics[model_name][K] -> list over users
    metrics = {
        "ot_from_books": {k: {"recall": [], "precision": []} for k in Ks},
        "random": {k: {"recall": [], "precision": []} for k in Ks},
        "popularity": {k: {"recall": [], "precision": []} for k in Ks},
    }

    # ------------------------------------------------------------------
    # Per-user evaluation
    # ------------------------------------------------------------------
    if eval_users and Ks:
        print("\n[Eval] Evaluating OT vs random & popularity...")
        n_items_movies = movies_item_emb.shape[0]

        for u in eval_users:
            # Ground-truth Movies positives (item indices)
            user_movies_items = movies_inter.loc[
                movies_inter[args.user_col].astype(str) == u, args.item_col
            ]
            pos_indices = [
                movies_item2idx[str(it)]
                for it in user_movies_items
                if str(it) in movies_item2idx
            ]
            pos_indices = sorted(set(pos_indices))
            if len(pos_indices) < args.min_positives:
                continue  # sanity check

            # ---------- OT scores ----------
            row_pi = books_idx_in_Pi[u]
            weights = Pi_valid[row_pi]  # (V_valid,)
            ot_scores = weights @ R_movies_sub  # (n_items_movies,)

            # ---------- Random scores ----------
            random_scores = np.random.rand(n_items_movies)

            # ---------- Popularity scores ----------
            pop_scores = popularity_counts.copy()

            # ---------- Metrics per K ----------
            for k in Ks:
                rec_ot, prec_ot = _recall_precision_at_k(ot_scores, pos_indices, k)
                rec_rnd, prec_rnd = _recall_precision_at_k(random_scores, pos_indices, k)
                rec_pop, prec_pop = _recall_precision_at_k(pop_scores, pos_indices, k)

                metrics["ot_from_books"][k]["recall"].append(rec_ot)
                metrics["ot_from_books"][k]["precision"].append(prec_ot)

                metrics["random"][k]["recall"].append(rec_rnd)
                metrics["random"][k]["precision"].append(prec_rnd)

                metrics["popularity"][k]["recall"].append(rec_pop)
                metrics["popularity"][k]["precision"].append(prec_pop)

        # ------------------------------------------------------------------
        # Report aggregated results
        # ------------------------------------------------------------------
        print("\n[Eval] Recommendation results (averaged over eval users):")
        n_users_eval = len(eval_users)
        print(f"  #users evaluated: {n_users_eval}\n")

        for k in Ks:
            print(f"--- K = {k} ---")
            for model_name in ["random", "popularity", "ot_from_books"]:
                recs = metrics[model_name][k]["recall"]
                precs = metrics[model_name][k]["precision"]

                if recs:
                    mean_rec = float(np.mean(recs))
                    mean_prec = float(np.mean(precs))
                else:
                    mean_rec = float("nan")
                    mean_prec = float("nan")

                if model_name == "random":
                    label = "Random"
                elif model_name == "popularity":
                    label = "Popularity"
                else:
                    label = "OT (Books→Movies)"

                print(f"{label:15s} | Recall@{k}: {mean_rec:.4f}, Precision@{k}: {mean_prec:.4f}")
            print()
    else:
        print("\n[Eval] Skipping recommendation comparison (no eval users or no K).")

    # ------------------------------------------------------------------
    # Embedding-level OT diagnostics
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Embedding-level OT diagnostics (user embeddings)")
    print("=" * 70)
    print(
        f"Using metric={args.embedding_metric}, "
        f"reg={args.embedding_reg}, "
        f"sinkhorn={not args.embedding_no_sinkhorn}"
    )

    try:
        embedding_ot_result = compute_ot_mapping_from_files(
            books_embeddings_path=books_emb_path,
            movies_embeddings_path=movies_emb_path,
            metric=args.embedding_metric,
            reg=args.embedding_reg,
            sinkhorn=not args.embedding_no_sinkhorn,
            save_dir=None,  # don't write mapping files here
        )
        emb_metrics = embedding_ot_result["metrics"]

        print("\n[Embed OT] Alignment metrics on overlapping users:")
        print(f"  MSE        : {emb_metrics.get('mse', float('nan')):.6f}")
        print(f"  MAE        : {emb_metrics.get('mae', float('nan')):.6f}")
        print(f"  Mean cosine: {emb_metrics.get('mean_cosine', float('nan')):.6f}")
        print(
            "  (Mean cosine close to 1 and much higher than a random map "
            "indicates good embedding-space alignment.)"
        )
    except Exception as e:
        print("\n[Embed OT] Failed to run embedding-level OT diagnostics:")
        print(f"  Error: {e}")
        print(
            "  Make sure Books and Movies user embeddings have the same dimension "
            "and ot_wasserstein.py is correctly configured."
        )

    print("\n[Eval] Done.")


if __name__ == "__main__":
    main()