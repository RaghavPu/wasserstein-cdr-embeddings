"""
evaluate_ot_mapping.py
======================

Evaluate the Wasserstein OT user mapping (Books -> Movies) in embedding space.

Assumes:
  1. You trained both domains with train_single_domain.py (books, movies).
  2. You ran run_ot_wasserstein.py, which saved an OT mapping file at
     outputs/ot/books_to_movies_metric-<metric>_reg-<reg>/ot_mapping.npz

This script:
  - Loads that OT mapping npz (overlapping_user_ids, mapped_embeddings, etc.)
  - Loads the baseline Movies embeddings.npz (latest movies_* run by default)
  - Aligns overlapping users and computes:
      * MSE, MAE between OT-mapped Movies embeddings and baseline Movies embeddings
      * Cosine similarity stats
  - Optionally: compares Books vs Movies embeddings as a naive identity baseline
  - Saves metrics to a JSON file and prints a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import sys

# ---------------------------------------------------------------------------
# Make sure we can import from src when running `python src/evaluate_ot_mapping.py`
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import load_embeddings  # uses your teammate's implementation


# ---------------------------------------------------------------------------
# Helper: find latest movies_* run and latest OT mapping
# ---------------------------------------------------------------------------
def find_latest_run(domain: str, outputs_dir: Path) -> Path:
    """
    Find the most recent outputs/<domain>_* directory.
    """
    candidates = sorted(outputs_dir.glob(f"{domain}_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No runs found for domain '{domain}' under {outputs_dir!s}. "
            f"Expected something like '{outputs_dir}/{domain}_YYYYMMDD_HHMMSS'."
        )
    return candidates[-1]


def find_latest_mapping(outputs_dir: Path) -> Path:
    """
    Find the most recent OT mapping file under:
        outputs/ot/books_to_movies_metric-*/ot_mapping.npz
    """
    ot_root = outputs_dir / "ot"
    candidates = sorted(ot_root.glob("books_to_movies_metric-*/ot_mapping.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No OT mapping files found under {ot_root!s}. "
            "Make sure you've run `python src/run_ot_wasserstein.py`."
        )
    return candidates[-1]


# ---------------------------------------------------------------------------
# Helper: compute embedding-level metrics
# ---------------------------------------------------------------------------
def compute_embedding_metrics(
    true_emb: np.ndarray, pred_emb: np.ndarray
) -> Dict[str, float]:
    """
    Compute MSE / MAE / cosine similarity stats between two embedding matrices.

    Parameters
    ----------
    true_emb : (n, d)
        Ground truth embeddings (e.g., Movies user embeddings).
    pred_emb : (n, d)
        Predicted embeddings (e.g., OT-mapped Movies embeddings).

    Returns
    -------
    metrics : dict
    """
    true_emb = np.asarray(true_emb, dtype=np.float32)
    pred_emb = np.asarray(pred_emb, dtype=np.float32)

    if true_emb.shape != pred_emb.shape:
        raise ValueError(
            f"Shape mismatch: true_emb {true_emb.shape}, pred_emb {pred_emb.shape}"
        )

    diff = pred_emb - true_emb
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))

    true_norm = np.linalg.norm(true_emb, axis=1) + 1e-8
    pred_norm = np.linalg.norm(pred_emb, axis=1) + 1e-8
    cos = np.sum(true_emb * pred_emb, axis=1) / (true_norm * pred_norm)

    metrics = {
        "mse": mse,
        "mae": mae,
        "cosine_mean": float(np.mean(cos)),
        "cosine_std": float(np.std(cos)),
        "cosine_min": float(np.min(cos)),
        "cosine_max": float(np.max(cos)),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Wasserstein OT user mapping (Books -> Movies) in embedding space."
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default=None,
        help=(
            "Path to the OT mapping npz produced by run_ot_wasserstein.py. "
            "If omitted, will search under outputs/ot/**/ot_mapping.npz "
            "and use the latest one."
        ),
    )
    parser.add_argument(
        "--movies_run_dir",
        type=str,
        default=None,
        help=(
            "Path to a specific Movies run directory under outputs/, e.g. "
            "outputs/movies_20251128_235114. "
            "If omitted, the latest movies_* run is used."
        ),
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Base outputs directory (same as for training). Default: ./outputs",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=(
            "Optional path to save the metrics JSON. "
            "Default: outputs/ot_eval_books_to_movies.json"
        ),
    )

    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)

    # ----------------------------------------------------------------------
    # 1. Resolve mapping_path
    # ----------------------------------------------------------------------
    if args.mapping_path is None:
        mapping_path = find_latest_mapping(outputs_dir)
    else:
        mapping_path = Path(args.mapping_path)

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"OT mapping file not found: {mapping_path!s}\n"
            "Make sure you have already run `python src/run_ot_wasserstein.py`."
        )

    # ----------------------------------------------------------------------
    # 2. Load OT mapping npz
    # ----------------------------------------------------------------------
    mapping_data = np.load(mapping_path, allow_pickle=True)

    if "overlapping_user_ids" not in mapping_data:
        raise KeyError(
            f"'overlapping_user_ids' not found in {mapping_path!s}. "
            "Please check that run_ot_wasserstein.py saved this key."
        )

    overlapping_user_ids = mapping_data["overlapping_user_ids"]
    overlapping_user_ids: List[str] = list(overlapping_user_ids.tolist())

    # Handle mapped embeddings key name
    if "mapped_movies_embeddings" in mapping_data:
        mapped_movies_emb = mapping_data["mapped_movies_embeddings"]
    elif "mapped_embeddings" in mapping_data:
        mapped_movies_emb = mapping_data["mapped_embeddings"]
    else:
        raise KeyError(
            f"Neither 'mapped_movies_embeddings' nor 'mapped_embeddings' "
            f"found in {mapping_path!s}."
        )

    mapped_movies_emb = np.asarray(mapped_movies_emb, dtype=np.float32)

    # Optional: identity baseline arrays from mapping_data (if present)
    books_emb_opt = mapping_data["books_embeddings"] if "books_embeddings" in mapping_data else None
    movies_emb_opt = mapping_data["movies_embeddings"] if "movies_embeddings" in mapping_data else None

    # ----------------------------------------------------------------------
    # 3. Load baseline Movies embeddings (latest or specified run)
    # ----------------------------------------------------------------------
    if args.movies_run_dir is None:
        movies_run_dir = find_latest_run("movies", outputs_dir)
    else:
        movies_run_dir = Path(args.movies_run_dir)

    movies_embeddings_path = movies_run_dir / "embeddings.npz"
    if not movies_embeddings_path.exists():
        raise FileNotFoundError(
            f"Movies embeddings not found at {movies_embeddings_path!s}.\n"
            "It should have been created by train_single_domain.py."
        )

    # utils.load_embeddings returns a 4-tuple:
    #   (user_embeddings, item_embeddings, user2idx, item2idx)
    movies_user_emb, movies_item_emb, movies_user2idx, movies_item2idx = load_embeddings(
        str(movies_embeddings_path)
    )
    movies_user_emb = np.asarray(movies_user_emb, dtype=np.float32)

    # ----------------------------------------------------------------------
    # 4. Align overlapping users for evaluation
    # ----------------------------------------------------------------------
    true_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []
    missing_ids: List[str] = []

    for uid, mapped_vec in zip(overlapping_user_ids, mapped_movies_emb):
        if uid not in movies_user2idx:
            missing_ids.append(uid)
            continue
        idx = movies_user2idx[uid]
        true_list.append(movies_user_emb[idx])
        pred_list.append(mapped_vec)

    if not true_list:
        raise RuntimeError(
            "No overlapping users from OT mapping were found in Movies user2idx. "
            "Check that overlapping_user_ids is consistent with Movies embeddings."
        )

    true_arr = np.stack(true_list, axis=0)
    pred_arr = np.stack(pred_list, axis=0)

    # ----------------------------------------------------------------------
    # 5. Compute metrics: OT vs baseline Movies embeddings
    # ----------------------------------------------------------------------
    metrics: Dict[str, Any] = {
        "n_overlap_total_in_mapping": int(len(overlapping_user_ids)),
        "n_overlap_used_in_eval": int(true_arr.shape[0]),
        "n_overlap_missing_in_movies": int(len(missing_ids)),
        "embedding_metrics_ot_vs_movies": compute_embedding_metrics(true_arr, pred_arr),
    }

    # ----------------------------------------------------------------------
    # 6. Optional: Books vs Movies identity baseline (if arrays present)
    # ----------------------------------------------------------------------
    if books_emb_opt is not None and movies_emb_opt is not None:
        books_emb = np.asarray(books_emb_opt, dtype=np.float32)
        movies_emb = np.asarray(movies_emb_opt, dtype=np.float32)

        # Assume rows are in the same overlapping_user_ids order as mapped_emb
        if books_emb.shape[0] == movies_emb.shape[0]:
            # Restrict to same subset of users we used above (those not missing)
            valid_indices = [
                i for i, uid in enumerate(overlapping_user_ids) if uid not in missing_ids
            ]
            if valid_indices:
                books_arr = books_emb[valid_indices]
                movies_arr = movies_emb[valid_indices]
                metrics["embedding_metrics_books_vs_movies_identity"] = (
                    compute_embedding_metrics(movies_arr, books_arr)
                )

    # ----------------------------------------------------------------------
    # 7. Save metrics JSON and print summary
    # ----------------------------------------------------------------------
    if args.save_path is None:
        save_path = outputs_dir / "ot_eval_books_to_movies.json"
    else:
        save_path = Path(args.save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("============================================================")
    print("OT Mapping Evaluation (Books -> Movies, embedding space)")
    print("============================================================")
    print(f"OT mapping file : {mapping_path}")
    print(f"Movies run dir  : {movies_run_dir}")
    print(f"Overlap (mapping): {metrics['n_overlap_total_in_mapping']}")
    print(f"Overlap (used)  : {metrics['n_overlap_used_in_eval']}")
    if metrics["n_overlap_missing_in_movies"]:
        print(
            f"Overlap (missing in Movies): {metrics['n_overlap_missing_in_movies']}"
        )

    print("\nOT vs Movies user embeddings (main metric):")
    for k, v in metrics["embedding_metrics_ot_vs_movies"].items():
        print(f"  {k:>20}: {v:.6f}")

    if "embedding_metrics_books_vs_movies_identity" in metrics:
        print("\nBooks vs Movies user embeddings (identity / naive baseline):")
        for k, v in metrics["embedding_metrics_books_vs_movies_identity"].items():
            print(f"  {k:>20}: {v:.6f}")

    print(f"\nSaved JSON metrics to: {save_path}")


if __name__ == "__main__":
    main()