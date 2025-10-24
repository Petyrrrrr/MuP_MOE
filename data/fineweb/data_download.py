#!/usr/bin/env python3
"""Download FineWeb train split with resume support and stop at 3M documents."""

import argparse
import os
import sys
import pickle
import signal
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from datasets import load_dataset

DEFAULT_LIMIT = 10_000_000
DEFAULT_CHECKPOINT_INTERVAL = 2_500_000
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_DATASET_CONFIG: Optional[str] = None
DEFAULT_SPLIT = "train"

DEFAULT_BIG_DIR = Path('/mnt/local')

# Global configuration populated via configure_big_dir(); defaults allow module import.
big_dir = DEFAULT_BIG_DIR
CHECKPOINT_DIR = big_dir / 'fineweb_checkpoint'
TRAIN_CHECKPOINT = CHECKPOINT_DIR / 'train.pkl'
PROGRESS_FILE = CHECKPOINT_DIR / 'progress.txt'
FINAL_OUTPUT = CHECKPOINT_DIR / 'train.pkl'

# Mutable state so helpers can access the resolved directories & counters.
train_data = []
current_stage = "starting"
train_limit = DEFAULT_LIMIT
checkpoint_interval = DEFAULT_CHECKPOINT_INTERVAL


def configure_big_dir(base_dir: Path) -> None:
    """Apply the big-dir override and ensure cache directories exist."""
    global big_dir, CHECKPOINT_DIR, TRAIN_CHECKPOINT, PROGRESS_FILE, FINAL_OUTPUT

    big_dir = base_dir
    CHECKPOINT_DIR = big_dir / 'fineweb_checkpoint'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_CHECKPOINT = CHECKPOINT_DIR / 'train_data.pkl'
    PROGRESS_FILE = CHECKPOINT_DIR / 'progress.txt'
    FINAL_OUTPUT = CHECKPOINT_DIR / 'train.pkl'

    hf_cache = big_dir / 'huggingface_cache'
    hf_cache.mkdir(parents=True, exist_ok=True)
    (hf_cache / 'hub').mkdir(parents=True, exist_ok=True)
    (hf_cache / 'datasets').mkdir(parents=True, exist_ok=True)

    os.environ['HF_HOME'] = str(hf_cache)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(hf_cache / 'hub')
    os.environ['HF_DATASETS_CACHE'] = str(hf_cache / 'datasets')
    os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'


try:
    configure_big_dir(DEFAULT_BIG_DIR)
except PermissionError:
    # Fallback when the default big-dir is not writable; will be overridden in main().
    print(f"[warn] Unable to access default big dir {DEFAULT_BIG_DIR}; awaiting --big-dir override.", file=sys.stderr)


def save_progress(stage_name: str) -> None:
    """Persist the latest stage so the script can resume gracefully."""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(stage_name)
    print(f"\n✓ Progress saved: {stage_name}")


def save_data_checkpoint() -> None:
    """Write the current train_data list to disk for resuming later."""
    if not train_data:
        return
    with open(TRAIN_CHECKPOINT, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"✓ Saved checkpoint with {len(train_data):,} documents")


def signal_handler(signum, frame) -> None:  # type: ignore[override]
    """Ensure we keep work-in-progress if the run is interrupted."""
    print(f"\n\n⚠️  Interrupted at stage: {current_stage}")
    print("Saving checkpoint before exit...")
    save_data_checkpoint()
    save_progress(current_stage)
    print("\n✅ Checkpoint saved! Run this script again to resume.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def write_final_pickle() -> None:
    """Dump the fully-downloaded list to train.pkl next to this script."""
    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_OUTPUT, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"✓ Wrote {len(train_data):,} documents to {FINAL_OUTPUT}")


def main() -> None:
    global train_data, current_stage, train_limit, checkpoint_interval

    parser = argparse.ArgumentParser(description="Download the FineWeb dataset to a large disk")
    parser.add_argument(
        "--big-dir",
        default=str(DEFAULT_BIG_DIR),
        help="Base directory for Hugging Face caches and checkpoints (default: /mnt/local)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Datasets hub path for the source dataset (default: HuggingFaceFW/fineweb)",
    )
    parser.add_argument(
        "--subset",
        default=DEFAULT_DATASET_CONFIG,
        help="Optional dataset subset/configuration name",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Split to stream from the dataset (default: train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of documents to download before stopping (default: {DEFAULT_LIMIT:,})",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"How often to write a checkpoint (default: {DEFAULT_CHECKPOINT_INTERVAL:,})",
    )
    args = parser.parse_args()

    resolved_big_dir = Path(args.big_dir).expanduser().resolve()
    configure_big_dir(resolved_big_dir)

    train_limit = max(0, args.limit)
    checkpoint_interval = max(1, args.checkpoint_interval)

    print("=" * 80)
    print("FineWeb Dataset Download - Resume Friendly (Train Only)")
    print(f"Target: {train_limit:,} documents")
    print("=" * 80)

    train_loaded = False

    if TRAIN_CHECKPOINT.exists():
        print("\n📦 Found existing checkpoint")
        with open(TRAIN_CHECKPOINT, 'rb') as f:
            train_data = pickle.load(f)
        print(f"Loaded {len(train_data):,} documents")
        train_loaded = True
    else:
        train_data = []

    if len(train_data) >= train_limit:
        print("\n✅ Stored documents already meet the target. Writing final pickle...")
        current_stage = "download_complete"
        save_progress(current_stage)
        write_final_pickle()
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print("\n🎉 Success! train.pkl is ready for tokenization.")
        return

    remaining = train_limit - len(train_data)

    print(f"\n📥 Downloading FineWeb dataset to gather {remaining:,} additional documents...")
    dataset_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }
    if args.subset:
        dataset_kwargs["name"] = args.subset

    split_iterable = load_dataset(**dataset_kwargs)

    current_stage = "downloading_train"
    save_progress(current_stage)

    skip_count = len(train_data)
    if skip_count:
        print(f"Skipping {skip_count:,} documents already stored in the checkpoint...")

    with tqdm(total=remaining, desc="FineWeb documents", unit="doc") as pbar:
        examples_skipped = 0

        for example in split_iterable:
            if examples_skipped < skip_count:
                examples_skipped += 1
                if examples_skipped % 1_000_000 == 0:
                    print(f"\rSkipped {examples_skipped:,}/{skip_count:,} documents...", end="")
                continue

            train_data.append(example)
            pbar.update(1)

            if len(train_data) % checkpoint_interval == 0:
                print(f"\n📊 Checkpoint at {len(train_data):,} documents")
                save_data_checkpoint()

            if len(train_data) >= train_limit:
                break

    print("\nFinalizing download...")
    save_data_checkpoint()
    current_stage = "download_complete"
    save_progress(current_stage)

    write_final_pickle()

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    print("\n🎉 Success! train.pkl is ready for tokenization.")


if __name__ == "__main__":
    main()
