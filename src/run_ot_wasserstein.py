"""
Command-line entry point for running the OT-based Wasserstein mapping
between Books and Movies user embeddings.

Usage (from root):

    python src/run_ot_wasserstein.py
    python src/run_ot_wasserstein.py --metric cosine --reg 0.1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make local src imports work when running as a script
sys.path.append(str(Path(__file__).parent))

from ot_wasserstein import compute_ot_mapping_from_files  # noqa: E402


def _find_latest_embeddings(domain: str) -> str:
    """
    Find the most recent embeddings.npz for a given domain (books or movies).

    This mirrors the pattern used in demo.py, which looks for directories
    like outputs/books_YYYYMMDD_HHMMSS/.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Wasserstein OT mapping between Books and Movies users."
    )
    parser.add_argument(
        "--books-path",
        type=str,
        default=None,
        help="Path to Books embeddings.npz (default: latest outputs/books_*/embeddings.npz).",
    )
    parser.add_argument(
        "--movies-path",
        type=str,
        default=None,
        help="Path to Movies embeddings.npz (default: latest outputs/movies_*/embeddings.npz).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "mahalanobis"],
        help="User distance metric for the OT ground cost.",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=0.05,
        help="Entropic regularization parameter for Sinkhorn.",
    )
    parser.add_argument(
        "--no-sinkhorn",
        action="store_true",
        help="If set, use exact EMD instead of Sinkhorn.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ot",
        help="Base directory to save OT mapping artifacts.",
    )

    args = parser.parse_args()

    # Resolve embeddings paths
    books_path = args.books_path or _find_latest_embeddings("books")
    movies_path = args.movies_path or _find_latest_embeddings("movies")

    print("=" * 60)
    print("Wasserstein OT Mapping: Books → Movies")
    print("=" * 60)
    print(f"Books embeddings:  {books_path}")
    print(f"Movies embeddings: {movies_path}")
    print(f"Metric: {args.metric}, reg={args.reg}, sinkhorn={not args.no_sinkhorn}")

    # Output subdir name encodes configuration for easy comparison
    save_dir = (
        Path(args.output_dir)
        / f"books_to_movies_metric-{args.metric}_reg-{args.reg:.3f}"
    )

    result = compute_ot_mapping_from_files(
        books_embeddings_path=books_path,
        movies_embeddings_path=movies_path,
        metric=args.metric,
        reg=args.reg,
        sinkhorn=not args.no_sinkhorn,
        save_dir=str(save_dir),
    )

    metrics = result["metrics"]

    print("\n[OT] Mapping evaluation on overlapping users:")
    print(f"  MSE        : {metrics['mse']:.6f}")
    print(f"  MAE        : {metrics['mae']:.6f}")
    print(f"  Mean Cosine: {metrics['mean_cosine']:.6f}")

    print(f"\n[OT] Saved artifacts to: {save_dir}")
    print("    - ot_mapping.npz  (mapped embeddings, cost matrix, transport plan)")
    print("    - ot_metrics.json (MSE / MAE / cosine metrics)")


if __name__ == "__main__":
    main()