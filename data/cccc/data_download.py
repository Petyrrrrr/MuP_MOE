#!/usr/bin/env python3
"""Download C4 train split with resume support and stop at 85M examples."""

import os
import sys
import pickle
import signal
from pathlib import Path

big_dir = "/mnt/b-large/"

# Set cache directories so downloads land on the large disk
os.environ['HF_HOME'] = big_dir + 'huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = big_dir + 'huggingface_cache/hub'
os.environ['HF_DATASETS_CACHE'] = big_dir + 'huggingface_cache/datasets'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from tqdm import tqdm
from datasets import load_dataset

TRAIN_LIMIT = 85_000_000
CHECKPOINT_INTERVAL = 20_000_000

CHECKPOINT_DIR = Path(big_dir + 'c4_checkpoint')
CHECKPOINT_DIR.mkdir(exist_ok=True)

TRAIN_CHECKPOINT = CHECKPOINT_DIR / 'train_data.pkl'
PROGRESS_FILE = CHECKPOINT_DIR / 'progress.txt'
FINAL_OUTPUT = CHECKPOINT_DIR / 'train.pkl'

train_data = []
current_stage = "starting"


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
    print(f"✓ Saved checkpoint with {len(train_data):,} train examples")


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
    print(f"✓ Wrote {len(train_data):,} examples to {FINAL_OUTPUT}")


def main() -> None:
    global train_data, current_stage

    print("=" * 80)
    print("C4 Dataset Download - Resume Friendly (Train Only)")
    print("Target: 85,000,000 examples")
    print("=" * 80)

    train_loaded = False

    if TRAIN_CHECKPOINT.exists():
        print("\n📦 Found existing train checkpoint")
        with open(TRAIN_CHECKPOINT, 'rb') as f:
            train_data = pickle.load(f)
        print(f"Loaded {len(train_data):,} train examples")
        train_loaded = True
    else:
        train_data = []

    if len(train_data) >= TRAIN_LIMIT:
        print("\n✅ Train data already meets the 85M target. Writing final pickle...")
        current_stage = "download_complete"
        save_progress(current_stage)
        write_final_pickle()
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print("\n🎉 Success! train.pkl is ready for tokenization.")
        return

    remaining_train = TRAIN_LIMIT - len(train_data)

    print(f"\n📥 Downloading C4 dataset to gather {remaining_train:,} additional examples...")
    dataset_stream = load_dataset("c4", "en", streaming=True)

    current_stage = "downloading_train"
    save_progress(current_stage)

    skip_count = len(train_data)
    if skip_count:
        print(f"Skipping {skip_count:,} examples already stored in the checkpoint...")

    with tqdm(total=remaining_train, desc="Train data", unit="ex") as pbar:
        examples_skipped = 0

        for example in dataset_stream['train']:
            if examples_skipped < skip_count:
                examples_skipped += 1
                if examples_skipped % 1_000_000 == 0:
                    print(f"\rSkipped {examples_skipped:,}/{skip_count:,} examples...", end="")
                continue

            train_data.append(example)
            pbar.update(1)

            if len(train_data) % CHECKPOINT_INTERVAL == 0:
                print(f"\n📊 Checkpoint at {len(train_data):,} examples")
                save_data_checkpoint()

            if len(train_data) >= TRAIN_LIMIT:
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
