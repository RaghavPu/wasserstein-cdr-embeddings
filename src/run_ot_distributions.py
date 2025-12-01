"""
Run OT on user preference distributions (PMD-style).

Usage example (adjust paths/columns as needed):

  python src/run_ot_distributions.py \
      --books-embeddings "outputs/books_*/embeddings.npz" \
      --movies-embeddings "outputs/movies_*/embeddings.npz" \
      --books-interactions "data/books.inter" \
      --movies-interactions "data/movies.inter" \
      --user-col "user_id" --item-col "item_id" --rating-col "rating"

You can also omit the paths to embeddings, in which case the script
will take the latest outputs/books_* and outputs/movies_* directories.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

# Make local src imports work when running as a script
sys.path.append(str(Path(__file__).parent))

from ot_distributions import (  # noqa: E402
    build_rating_matrix,
    build_user_distributions,
    cross_domain_user_distance_matrix,
    load_embeddings,
    load_interactions,
    overlapping_user_ids,
    user_population_ot,
    wasserstein_user_distance,
)


# ---------------------------------------------------------------------
# Helpers to resolve paths
# ---------------------------------------------------------------------


def _find_latest_embeddings(domain: str) -> str:
    """
    Find the most recent embeddings.npz for a given domain (books or movies).
    Looks for directories like outputs/books_YYYYMMDD_HHMMSS.
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


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OT-based cross-domain analysis on user *distributions* (PMD-style)."
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
        help="Path to Books interactions file (.inter, .tsv, .csv).",
    )
    parser.add_argument(
        "--movies-interactions",
        type=str,
        required=True,
        help="Path to Movies interactions file (.inter, .tsv, .csv).",
    )

    # Column names / separator
    parser.add_argument("--user-col", type=str, default="user_id", help="User ID column name.")
    parser.add_argument("--item-col", type=str, default="item_id", help="Item ID column name.")
    parser.add_argument("--rating-col", type=str, default="rating", help="Rating column name.")
    parser.add_argument("--sep", type=str, default="\t", help="Field separator (default: '\\t').")

    # Distribution & OT hyperparameters
    parser.add_argument("--beta", type=float, default=1.0, help="Softmax temperature for user weights.")
    parser.add_argument("--reg-inner", type=float, default=0.05, help="Entropic regularization for inner W1 (user–user).")
    parser.add_argument("--reg-outer", type=float, default=0.1, help="Entropic regularization for outer OT (user populations).")

    # Subsampling
    parser.add_argument(
        "--num-users-books",
        type=int,
        default=300,
        help="Number of Books users to include in the cross-domain user distance matrix.",
    )
    parser.add_argument(
        "--num-users-movies",
        type=int,
        default=300,
        help="Number of Movies users to include in the cross-domain user distance matrix.",
    )
    parser.add_argument(
        "--max-overlap-eval",
        type=int,
        default=200,
        help="Max # of overlapping users for per-user W1(Books_u, Movies_u) diagnostics.",
    )

    # Demo recommendation
    parser.add_argument(
        "--demo-user",
        type=str,
        default=None,
        help="Raw Books user_id for which to run cross-domain recommendation demo.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K items to show in recommendation demo.",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ot_distributions.npz",
        help="Path to save OT artifacts (D, Pi, user lists, metrics).",
    )

    args = parser.parse_args()

    # Resolve embeddings
    if args.books_embeddings is None:
        books_emb_path = _find_latest_embeddings("books")
    else:
        books_emb_path = _resolve_embeddings_input(args.books_embeddings)

    if args.movies_embeddings is None:
        movies_emb_path = _find_latest_embeddings("movies")
    else:
        movies_emb_path = _resolve_embeddings_input(args.movies_embeddings)

    print("=" * 70)
    print("OT on User Distributions (Books ↔ Movies)")
    print("=" * 70)
    print(f"Books embeddings:   {books_emb_path}")
    print(f"Movies embeddings:  {movies_emb_path}")
    print(f"Books interactions: {args.books_interactions}")
    print(f"Movies interactions:{args.movies_interactions}")
    print(f"beta={args.beta}, reg_inner={args.reg_inner}, reg_outer={args.reg_outer}")

    # Load embeddings
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

    # Load interactions
    books_interactions = load_interactions(
        args.books_interactions,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        sep=args.sep,
    )
    movies_interactions = load_interactions(
        args.movies_interactions,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        sep=args.sep,
    )

    # Build user distributions
    print("\n[OT] Building Books user distributions...")
    books_dists = build_user_distributions(
        books_interactions,
        item_embeddings=books_item_emb,
        user2idx=books_user2idx,
        item2idx=books_item2idx,
        beta=args.beta,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
    )
    print(f"[OT] Books users with distributions: {len(books_dists)}")

    print("\n[OT] Building Movies user distributions...")
    movies_dists = build_user_distributions(
        movies_interactions,
        item_embeddings=movies_item_emb,
        user2idx=movies_user2idx,
        item2idx=movies_item2idx,
        beta=args.beta,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
    )
    print(f"[OT] Movies users with distributions: {len(movies_dists)}")

    # Overlapping users (for diagnostics)
    overlap = overlapping_user_ids(books_dists, movies_dists)
    print(f"\n[OT] Overlapping users with distributions: {len(overlap)}")

    if overlap:
        # Per-user W1(Books_u, Movies_u) for a subset
        print(f"[OT] Computing per-user W1(p_u^Books, p_u^Movies) "
              f"for up to {args.max_overlap_eval} overlapping users...")
        subset = overlap[: args.max_overlap_eval]
        vals = []
        for uid in subset:
            d_uv, _ = wasserstein_user_distance(
                books_dists[uid],
                movies_dists[uid],
                reg=args.reg_inner,
            )
            vals.append(d_uv)
        vals = np.array(vals, dtype=np.float64)
        print(f"[OT] Per-user W1 stats on {len(vals)} users:")
        print(f"     mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")

    # Subsample users for full cross-domain user distance matrix
    books_users_all = list(books_dists.keys())
    movies_users_all = list(movies_dists.keys())

    books_users_for_D = books_users_all[: args.num_users_books]
    movies_users_for_D = movies_users_all[: args.num_users_movies]

    print(f"\n[OT] Computing cross-domain user distance matrix D "
          f"for {len(books_users_for_D)} Books users × {len(movies_users_for_D)} Movies users...")
    D, books_users_for_D, movies_users_for_D = cross_domain_user_distance_matrix(
        books_dists,
        movies_dists,
        users_a=books_users_for_D,
        users_b=movies_users_for_D,
        reg=args.reg_inner,
    )
    print(f"[OT] D shape={D.shape}, mean={D.mean():.4f}, std={D.std():.4f}")

    # Outer OT between user populations
    print(f"\n[OT] Computing user-population OT coupling Pi (reg_outer={args.reg_outer})...")
    from ot_distributions import user_population_ot  # local import for clarity
    Pi, W_domains = user_population_ot(D, reg=args.reg_outer)
    print(f"[OT] Domain-level Wasserstein distance W(Books, Movies) = {W_domains:.4f}")

    # Optional demo: cross-domain recommendation
    if args.demo_user is not None:
        demo_user_id = args.demo_user
        if demo_user_id not in books_users_for_D:
            print(f"\n[OT] demo_user={demo_user_id} not in Books subset; skipping recommendation demo.")
        else:
            print(f"\n[OT] Running cross-domain recommendation demo for Books user: {demo_user_id}")
            from ot_distributions import predict_cross_domain_scores_from_Pi, build_rating_matrix

            # Build Movies rating matrix
            print("[OT] Building Movies rating matrix...")
            R_movies = build_rating_matrix(
                movies_interactions,
                user2idx=movies_user2idx,
                item2idx=movies_item2idx,
                user_col=args.user_col,
                item_col=args.item_col,
                rating_col=args.rating_col,
            )

            ranked_items = predict_cross_domain_scores_from_Pi(
                user_id_source=demo_user_id,
                users_source=books_users_for_D,
                users_target=movies_users_for_D,
                Pi=Pi,
                R_target=R_movies,
                user2idx_target=movies_user2idx,
                topk=args.topk,
            )
            print(f"[OT] Top-{args.topk} Movies item indices and scores for Books user {demo_user_id}:")
            for rank, (item_idx, score) in enumerate(ranked_items, start=1):
                print(f"  {rank:2d}. item_idx={item_idx:6d}, score={score:.4f}")

    # Save artifacts
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        D=D,
        Pi=Pi,
        books_users=np.array(books_users_for_D, dtype=object),
        movies_users=np.array(movies_users_for_D, dtype=object),
        W_domains=np.array([W_domains]),
    )
    print(f"\n[OT] Saved OT distribution artifacts to: {out_path}")


if __name__ == "__main__":
    main()