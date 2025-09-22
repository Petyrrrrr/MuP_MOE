#!/usr/bin/env python3

import os
import sys
import pickle
import signal
from pathlib import Path

# Set cache directories
os.environ['HF_HOME'] = '/mnt/b-large/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/b-large/huggingface_cache/hub'
os.environ['HF_DATASETS_CACHE'] = '/mnt/b-large/huggingface_cache/datasets'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

# Configuration matching your last run
MAX_TOKENS_BILLIONS = 20.0
NUM_PROC = 8
CHECKPOINT_DIR = Path('/mnt/b-large/c4_checkpoint')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Files for checkpoints
TRAIN_CHECKPOINT = CHECKPOINT_DIR / 'train_data.pkl'
VAL_CHECKPOINT = CHECKPOINT_DIR / 'val_data.pkl'
PROGRESS_FILE = CHECKPOINT_DIR / 'progress.txt'

enc = tiktoken.get_encoding("gpt2")

# Global variables for signal handling
train_data = []
val_data = []
current_stage = "starting"

def save_progress(stage_name):
    """Save current progress stage"""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(stage_name)
    print(f"\n✓ Progress saved: {stage_name}")

def save_data_checkpoint():
    """Save current data to checkpoint files"""
    global train_data, val_data
    if train_data:
        print("\nSaving train data checkpoint...")
        with open(TRAIN_CHECKPOINT, 'wb') as f:
            pickle.dump(train_data, f)
        print(f"✓ Saved {len(train_data)} train examples")
    if val_data:
        print("Saving validation data checkpoint...")
        with open(VAL_CHECKPOINT, 'wb') as f:
            pickle.dump(val_data, f)
        print(f"✓ Saved {len(val_data)} validation examples")

def signal_handler(signum, frame):
    """Handle interruption gracefully"""
    print(f"\n\n⚠️  Interrupted at stage: {current_stage}")
    print("Saving checkpoint before exit...")
    save_data_checkpoint()
    save_progress(current_stage)
    print("\n✅ Checkpoint saved! Run this script again to resume.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    global train_data, val_data, current_stage

    print("=" * 80)
    print("C4 Dataset Download & Tokenization - Fast Resume")
    print("=" * 80)

    # Check for existing checkpoints and resume appropriately
    train_loaded = False
    val_loaded = False

    # Load train checkpoint if exists
    if TRAIN_CHECKPOINT.exists():
        print("\n📦 Found train checkpoint!")
        with open(TRAIN_CHECKPOINT, 'rb') as f:
            train_data = pickle.load(f)
        print(f"Loaded {len(train_data):,} train examples")
        train_loaded = True
    else:
        train_data = []

    # Load validation checkpoint if exists
    if VAL_CHECKPOINT.exists():
        print("📦 Found validation checkpoint!")
        with open(VAL_CHECKPOINT, 'rb') as f:
            val_data = pickle.load(f)
        print(f"Loaded {len(val_data):,} validation examples")
        val_loaded = True
    else:
        val_data = []

    # Same limits as your last run
    train_limit = 91_250_000  # Exact number from your output
    val_limit = 91_250        # Exact number from your output

    # Check if we need to download more data
    if len(train_data) >= train_limit and len(val_data) >= val_limit:
        print("\n✅ All data already downloaded! Proceeding to tokenization...")
        current_stage = "tokenization"
        save_progress(current_stage)
    else:
        # Need to continue downloading
        print(f"\n📥 Downloading C4 dataset...")

        dataset_stream = load_dataset("c4", "en", streaming=True)

        # Resume or start train download
        if len(train_data) < train_limit:
            remaining_train = train_limit - len(train_data)
            print(f"\n{'Resuming' if train_loaded else 'Starting'} train download...")
            print(f"Progress: {len(train_data):,} / {train_limit:,} examples")
            print(f"Remaining: {remaining_train:,} examples")

            current_stage = "downloading_train"
            save_progress(current_stage)

            # More efficient approach: iterate and skip manually
            skip_count = len(train_data)
            examples_to_download = remaining_train

            print(f"Skipping {skip_count:,} examples...")
            with tqdm(total=remaining_train, desc="Train data", unit="ex") as pbar:
                examples_skipped = 0

                for example in dataset_stream['train']:
                    # Skip examples we already have
                    if examples_skipped < skip_count:
                        examples_skipped += 1
                        if examples_skipped % 1000000 == 0:
                            print(f"\rSkipped {examples_skipped:,}/{skip_count:,} examples...", end="")
                        continue

                    # Download new examples
                    train_data.append(example)
                    pbar.update(1)

                    if len(train_data) % 20000000 == 0:
                        print(f"\n📊 Checkpoint at {len(train_data):,} examples")
                        save_data_checkpoint()


                    # Stop when we have enough
                    if len(train_data) >= train_limit:
                        break

        # Download validation data if needed
        if len(val_data) < val_limit:
            remaining_val = val_limit - len(val_data)
            print(f"\n{'Resuming' if val_loaded else 'Starting'} validation download...")
            print(f"Progress: {len(val_data):,} / {val_limit:,} examples")
            print(f"Remaining: {remaining_val:,} examples")

            current_stage = "downloading_validation"
            save_progress(current_stage)

            # More efficient approach for validation too
            skip_count_val = len(val_data)
            print(f"Skipping {skip_count_val:,} validation examples...")

            examples_skipped_val = 0
            with tqdm(total=remaining_val, desc="Validation data", unit="ex") as pbar:
                for example in dataset_stream['validation']:
                    if examples_skipped_val < skip_count_val:
                        examples_skipped_val += 1
                        continue

                    val_data.append(example)
                    pbar.update(1)

                    if len(val_data) >= val_limit:
                        break

        # Save final checkpoint
        save_data_checkpoint()
        save_progress("download_complete")
        print("\n✅ Download complete and saved!")

    # Convert to Dataset format
    print("\n📊 Converting to Dataset format...")
    current_stage = "converting_to_dataset"
    save_progress(current_stage)

    dataset = {
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data)
    }

    # Clear memory
    del train_data
    del val_data

    # Estimate tokens (quick sample)
    print("\n🔢 Estimating token count...")
    sample_size = min(1000, len(dataset['train']))
    sample_indices = np.random.choice(len(dataset['train']), sample_size, replace=False)
    total_tokens = sum(len(enc.encode_ordinary(dataset['train'][int(idx)]['text']))
                      for idx in tqdm(sample_indices, desc="Token estimation"))
    avg_tokens = total_tokens / sample_size

    print(f"Average tokens per example: {avg_tokens:.2f}")
    estimated_total = avg_tokens * len(dataset['train'])
    print(f"Estimated total tokens: {estimated_total:,.0f} ({estimated_total/1e9:.2f}B)")

    # Apply token limit if needed
    max_tokens = MAX_TOKENS_BILLIONS * 1e9
    max_examples = min(int(max_tokens / avg_tokens), len(dataset['train']))

    if max_examples < len(dataset['train']):
        print(f"\n📉 Limiting to {MAX_TOKENS_BILLIONS}B tokens ({max_examples:,} examples)")
        indices = np.arange(len(dataset['train']))
        np.random.shuffle(indices)
        dataset['train'] = dataset['train'].select(indices[:max_examples])

    # Create train/val split
    print("\n🔀 Creating train/validation split...")
    current_stage = "splitting_dataset"
    save_progress(current_stage)

    split_dataset = dataset['train'].train_test_split(test_size=0.001, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # Tokenization function
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    # Tokenize
    print("\n🔤 Tokenizing dataset...")
    current_stage = "tokenizing"
    save_progress(current_stage)

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=NUM_PROC,
    )

    # Save to binary files
    print("\n💾 Writing binary files...")
    current_stage = "writing_binary"
    save_progress(current_stage)

    output_dir = Path(__file__).parent
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f"\n{split}: {len(dset)} examples, {arr_len:,} tokens ({arr_len/1e9:.2f}B)")

        filename = output_dir / f'{split}.bin'
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        total_batches = 1024
        idx = 0

        for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}.bin'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"✓ Saved {filename}")

    # Final summary
    print("\n" + "=" * 80)
    print("✅ TOKENIZATION COMPLETE!")
    print("=" * 80)

    total_tokens = 0
    for split in tokenized.keys():
        token_count = np.sum(tokenized[split]['len'], dtype=np.uint64)
        total_tokens += token_count
        print(f"{split}: {token_count:,} tokens ({token_count/1e9:.2f}B)")
    print(f"Total: {total_tokens:,} tokens ({total_tokens/1e9:.2f}B)")

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    print("\n🎉 Success! Binary files are ready for training.")
    print(f"Files saved in: {output_dir}")

if __name__ == "__main__":
    main()