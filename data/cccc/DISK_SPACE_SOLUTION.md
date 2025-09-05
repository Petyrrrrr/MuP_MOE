# C4 Dataset Preparation - Disk Space Solution

## Problem
The root filesystem (`/`) is full (235GB used out of 235GB), causing the C4 dataset download to fail with "No space left on device" error. The Hugging Face cache alone is using 209GB.

## Solution
We've created modified scripts that redirect downloads to `/mnt/local` which has 14TB of available space.

## New Scripts Created

### 1. `prepare_large_disk.py`
- Full C4 dataset preparation (same as original `prepare.py`)
- Redirects cache to `/mnt/local/huggingface_cache`
- Use this for processing the entire C4 dataset

### 2. `prepare_limited_large_disk.py`
- Limited C4 dataset preparation (same as `prepare_limited.py`)
- Redirects cache to `/mnt/local/huggingface_cache`
- Supports `--max_tokens_billions` parameter (default: 10.0)
- Use this for processing a subset of C4

### 3. `manage_cache.sh`
- Interactive script to manage the old cache
- Options to move or delete the old cache to free up space

## Usage Instructions

### Step 1: Free up disk space
```bash
cd /home/ubuntu/MuP_MOE/data/cccc
./manage_cache.sh
```
Choose option 2 to delete the old cache and free up ~209GB immediately, or option 1 to move it to the new location.

### Step 2: Run the dataset preparation

For a limited dataset (10B tokens by default):
```bash
python prepare_limited_large_disk.py
```

For a custom token limit (e.g., 5B tokens):
```bash
python prepare_limited_large_disk.py --max_tokens_billions 5.0
```

## What Changed?

The new scripts set these environment variables before importing datasets:
```python
os.environ['HF_HOME'] = '/mnt/local/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/local/huggingface_cache/hub'
os.environ['HF_DATASETS_CACHE'] = '/mnt/local/huggingface_cache/datasets'
```

This ensures all downloads go to the large disk instead of the root filesystem.

## Permanent Solution (Optional)

To make this change permanent for all Hugging Face operations, add these lines to your `~/.bashrc`:
```bash
export HF_HOME=/mnt/local/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/mnt/local/huggingface_cache/hub
export HF_DATASETS_CACHE=/mnt/local/huggingface_cache/datasets
```

Then reload: `source ~/.bashrc`

## Notes
- The C4 dataset is very large (>300GB compressed)
- Processing will generate additional temporary files
- The final `.bin` files will be saved in the same directory as the scripts
- Make sure to monitor disk usage during processing
