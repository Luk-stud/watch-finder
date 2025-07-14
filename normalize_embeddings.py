#!/usr/bin/env python3
"""
normalize_embeddings.py

Utility script to L2-normalize every embedding vector inside a
`precomputed_embeddings.pkl` file so each vector has unit length.

By default it looks for the pickle under `watch_finder_v2/backend/data/` and
writes a new file called `precomputed_embeddings_normalized.pkl` in the same
folder.  Use the `--inplace` flag to overwrite the original file instead, or
provide explicit `--input`/`--output` paths.

The pickle is expected to contain at least these keys:
    • `watch_data`          – metadata dict (untouched)
    • `final_embeddings`    – dict[int → np.ndarray]
    • `embedding_dim`       – int (untouched)

Example usages
--------------
# Dry-run, create a new normalized file next to the original
python normalize_embeddings.py

# Overwrite the original file in place (make a backup first!)
python normalize_embeddings.py --inplace

# Custom paths
python normalize_embeddings.py -i other_dir/my_embeddings.pkl -o fixed.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Any

import numpy as np

DEFAULT_INPUT = os.path.join(
    "watch_finder_v2", "backend", "data", "precomputed_embeddings.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize embedding vectors inside a precomputed_embeddings.pkl file"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the input pickle (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Where to save the normalized pickle. If omitted, a filename "
            "`*_normalized.pkl` is generated next to the input unless --inplace is set."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file instead of writing a new one",
    )
    return parser.parse_args()


def load_pickle(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input pickle not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_embeddings(data: Dict[str, Any]) -> None:
    """In-place L2 normalisation of all vectors in `data['final_embeddings']`."""
    if "final_embeddings" not in data:
        raise KeyError("`final_embeddings` key not found in pickle")

    final_embs = data["final_embeddings"]
    if not isinstance(final_embs, dict):
        raise TypeError("`final_embeddings` must be a dict of id → vector")

    norms_before = []
    zero_norm_ids = []

    # Perform normalisation in place
    for watch_id, vec in final_embs.items():
        vec_np = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec_np)
        norms_before.append(norm)
        if norm == 0:
            zero_norm_ids.append(watch_id)
            continue  # leave zero vector as-is
        final_embs[watch_id] = (vec_np / norm).astype(np.float32)

    norms_after = [np.linalg.norm(v) for v in final_embs.values() if np.linalg.norm(v) > 0]

    print(
        f"Processed {len(final_embs)} embeddings | "
        f"avg norm before: {np.mean(norms_before):.4f} | "
        f"avg norm after: {np.mean(norms_after):.4f}"
    )
    if zero_norm_ids:
        print(f"Warning: {len(zero_norm_ids)} vectors had zero norm and were left unchanged.")


if __name__ == "__main__":
    args = parse_args()

    output_path = (
        args.input if args.inplace else args.output or args.input.replace(".pkl", "_normalized.pkl")
    )

    if not args.inplace and os.path.abspath(output_path) == os.path.abspath(args.input):
        raise ValueError("Output path resolves to the same file as input — use --inplace instead.")

    print(f"Loading embeddings from {args.input} …")
    data_dict = load_pickle(args.input)

    normalize_embeddings(data_dict)

    print(f"Saving normalized embeddings to {output_path} …")
    save_pickle(data_dict, output_path)
    print("Done.") 