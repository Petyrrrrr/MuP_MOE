#!/usr/bin/env python3
"""Utility to materialize deterministic training/validation batches for the CCCC dataset.

This script slices the tokenized corpus (``train.bin``/``val.bin``) into fixed-size batches so
training can fetch them deterministically by iteration index. Each output file stores the raw
tokens for one "global" batch (480 sequences of length 1024 plus the next-token suffix) in a
flat ``.bin`` file, allowing the training loop to rebuild input/target tensors without extra
processing.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm


DEFAULT_DATASET_DIR = Path("data/cccc")
DEFAULT_BATCH_SIZE = 480
DEFAULT_BLOCK_SIZE = 1024
DEFAULT_TRAIN_BATCHES = 10_000
DEFAULT_VAL_BATCHES = 100
DTYPE = np.uint16


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    """Create (or clean) the output directory."""
    if path.exists():
        existing = list(path.iterdir())
        if existing:
            if not overwrite:
                raise FileExistsError(
                    f"Output directory {path} already contains files. Use --overwrite to replace them."
                )
            for item in existing:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write_manifest(
    dest: Path,
    *,
    total_batches: int,
    batch_size: int,
    block_size: int,
    tokens_per_sequence: int,
    tokens_per_batch: int,
) -> None:
    metadata = {
        "batch_size": batch_size,
        "block_size": block_size,
        "tokens_per_input_sample": block_size,
        "tokens_per_target_sample": block_size,
        "tokens_per_sequence": tokens_per_sequence,
        "tokens_per_batch": tokens_per_batch,
        "dtype": str(DTYPE.__name__),
        "num_batches": total_batches,
        "file_format": "raw_uint16",
        "layout": {
            "shape": [batch_size, tokens_per_sequence],
            "order": "C",
        },
    }
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)


def _compute_bounds(num_batches: int, batch_size: int, block_size: int) -> Tuple[int, int, int]:
    tokens_per_sequence = block_size + 1  # need one extra token to build the shifted targets
    tokens_per_batch = batch_size * tokens_per_sequence
    total_tokens_needed = tokens_per_batch * num_batches
    return tokens_per_sequence, tokens_per_batch, total_tokens_needed


def _consume_batches(
    data: np.memmap,
    dest_dir: Path,
    num_batches: int,
    *,
    tokens_per_sequence: int,
    tokens_per_batch: int,
    batch_size: int,
) -> None:
    """Emit ``num_batches`` raw ``.bin`` files to ``dest_dir`` from the token stream ``data``."""
    for batch_idx in tqdm(range(num_batches), desc=f"Writing {dest_dir.name}", unit="batch"):
        start = batch_idx * tokens_per_batch
        end = start + tokens_per_batch
        batch_tokens = np.asarray(data[start:end], dtype=DTYPE)
        batch_tokens = batch_tokens.reshape(batch_size, tokens_per_sequence)
        outfile = dest_dir / f"batch_{batch_idx:05d}.bin"
        batch_tokens.tofile(outfile)


def process_split(
    split: str,
    *,
    dataset_dir: Path,
    num_batches: int,
    batch_size: int,
    block_size: int,
    overwrite: bool,
) -> None:
    tokens_per_sequence, tokens_per_batch, total_tokens_needed = _compute_bounds(
        num_batches, batch_size, block_size
    )

    data_path = dataset_dir / f"{split}.bin"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected token file {data_path} not found")

    data = np.memmap(data_path, dtype=DTYPE, mode="r")
    if len(data) < total_tokens_needed:
        have_batches = len(data) // tokens_per_batch
        raise ValueError(
            f"{split}.bin does not contain enough tokens ({len(data)} available) for {num_batches} batches. "
            f"It can supply at most {have_batches} batches of size {batch_size}x{block_size}."
        )

    output_dir = dataset_dir / "split" / split
    _ensure_output_dir(output_dir, overwrite=overwrite)

    _consume_batches(
        data,
        output_dir,
        num_batches,
        tokens_per_sequence=tokens_per_sequence,
        tokens_per_batch=tokens_per_batch,
        batch_size=batch_size,
    )

    manifest_path = output_dir / "metadata.json"
    _write_manifest(
        manifest_path,
        total_batches=num_batches,
        batch_size=batch_size,
        block_size=block_size,
        tokens_per_sequence=tokens_per_sequence,
        tokens_per_batch=tokens_per_batch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to the tokenized dataset directory (holds train.bin/val.bin)",
    )
    parser.add_argument(
        "--train-batches",
        type=int,
        default=DEFAULT_TRAIN_BATCHES,
        help="Number of training batches to materialize",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=DEFAULT_VAL_BATCHES,
        help="Number of validation batches to materialize",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Sequences per global batch",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Sequence length (tokens per sample)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow clobbering existing split directories",
    )
    args = parser.parse_args()

    for name in ("batch_size", "block_size", "train_batches", "val_batches"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive (got {getattr(args, name)!r})")

    return args


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")

    process_split(
        "train",
        dataset_dir=dataset_dir,
        num_batches=args.train_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        overwrite=args.overwrite,
    )
    process_split(
        "val",
        dataset_dir=dataset_dir,
        num_batches=args.val_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
