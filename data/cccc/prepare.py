# saves the C4 (English) dataset to binary files for training in the same style
# as the common OpenWebText "prepare.py" template (nanoGPT-style).
#
# - Tokenizer: tiktoken GPT-2 BPE (50257 vocab), ids stored as uint16
# - Files written: train.bin (~train_tokens*2 bytes), val.bin (~val_tokens*2 bytes), meta.pkl
# - We stream C4 from Hugging Face and stop once we reach the requested token budgets
#   (so we do not need to download the full dataset).
#
# Example:
#   python prepare.py --out-dir data/c4 --config en --train-tokens 30000000000 --val-tokens 10000000
#
# Notes:
# * 30B tokens -> ~60 GB for train.bin (uint16), and 10M tokens -> ~20 MB for val.bin.
# * We do a counting pass first to know the exact number of tokens to pre-allocate for memmap,
#   and then a writing pass to fill it. This keeps the template flow similar to the OpenWebText script.
# * Each document is tokenized with encode_ordinary(...) and then we append the EOT token.
#
# Inspired by:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import math
import pickle
from typing import Iterable, Tuple

import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

# number of workers (kept for template parity;
# note: streaming .map doesn't use multiprocessing the same way)
num_proc = 8

def _get_eot_id(enc: "tiktoken.Encoding") -> int:
    # Try to use the encoding-provided EOT id; fall back to GPT-2's known token if not present
    eot = getattr(enc, "eot_token", None)
    if isinstance(eot, int):
        return eot
    # Fallback: GPT-2 endoftext is id 50256
    try:
        # safest: use the official special token with allowed_special
        e = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        if len(e) == 1:
            return e[0]
    except Exception:
        pass
    return 50256

def _iter_text(ds_iter: Iterable) -> Iterable[str]:
    for ex in ds_iter:
        txt = ex.get("text", None)
        if isinstance(txt, str) and txt:
            yield txt

def _count_tokens_for_split(config: str, split: str, target_tokens: int, enc: "tiktoken.Encoding", seed: int, shuffle_buffer: int) -> int:
    """Streaming pass to count up to target_tokens."""
    ds = load_dataset("c4", config, split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    eot_id = _get_eot_id(enc)
    total = 0
    for text in tqdm(_iter_text(ds), desc=f"counting {split}", dynamic_ncols=True):
        # +1 for eot token
        total += len(enc.encode_ordinary(text)) + 1
        if total >= target_tokens:
            break
    return total

def _write_split(config: str, split: str, out_path: str, total_tokens: int, enc: "tiktoken.Encoding", seed: int, shuffle_buffer: int, write_batch: int = 1_000_000) -> int:
    """Second pass to write exactly total_tokens tokens into a uint16 memmap."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    idx = 0
    buf = []
    ds = load_dataset("c4", config, split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    eot_id = _get_eot_id(enc)

    pbar = tqdm(total=total_tokens, desc=f"writing {split}", dynamic_ncols=True)
    for text in _iter_text(ds):
        ids = enc.encode_ordinary(text)
        ids.append(eot_id)
        # Clip last example if it would exceed total_tokens
        remaining = total_tokens - idx
        if remaining <= 0:
            break
        if len(ids) > remaining:
            ids = ids[:remaining]

        buf.extend(ids)
        if len(buf) >= write_batch:
            arr[idx : idx + len(buf)] = np.array(buf, dtype=np.uint16)
            idx += len(buf)
            pbar.update(len(buf))
            buf.clear()

        if idx >= total_tokens:
            break

    # flush any remainder
    if buf and idx < total_tokens:
        arr[idx : idx + len(buf)] = np.array(buf, dtype=np.uint16)
        idx += len(buf)
        pbar.update(len(buf))
        buf.clear()

    arr.flush()
    pbar.close()
    return idx

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare C4 (en) into train.bin/val.bin in nanoGPT-style.")
    parser.add_argument("--out-dir", type=str, default="data/c4", help="Output directory for .bin and meta.pkl")
    parser.add_argument("--config", type=str, default="en", help="C4 configuration (e.g., 'en', 'en.noclean')")
    parser.add_argument("--train-tokens", type=int, default=30_000_000_000, help="Approximate number of training tokens to write")
    parser.add_argument("--val-tokens", type=int, default=10_000_000, help="Approximate number of validation tokens to write")
    parser.add_argument("--seed", type=int, default=1337, help="Shuffle seed (only impacts streaming shuffle)")
    parser.add_argument("--shuffle-buffer", type=int, default=0, help="Buffer size for streaming shuffle; set >0 to enable shuffling")
    parser.add_argument("--write-batch", type=int, default=1_000_000, help="How many tokens to batch together before each memmap write")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # tokenizer setup
    enc = tiktoken.get_encoding("gpt2")
    eot_id = _get_eot_id(enc)
    assert enc.n_vocab <= 65535, f"Vocab too large for uint16: {enc.n_vocab}"
    assert 0 <= eot_id < 65536, f"EOT id out of range for uint16: {eot_id}"

    # count passes (to pre-allocate memmaps)
    train_total = _count_tokens_for_split(args.config, "train", args.train_tokens, enc, args.seed, args.shuffle_buffer)
    val_total   = _count_tokens_for_split(args.config, "validation", args.val_tokens, enc, args.seed, args.shuffle_buffer)

    # write passes
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path   = os.path.join(args.out_dir, "val.bin")

    wrote_train = _write_split(args.config, "train", train_path, train_total, enc, args.seed, args.shuffle_buffer, args.write_batch)
    wrote_val   = _write_split(args.config, "validation", val_path, val_total, enc, args.seed, args.shuffle_buffer, args.write_batch)

    # meta (kept similar to template)
    meta = {
        "dataset": "c4",
        "config": args.config,
        "tokenizer": "tiktoken/gpt2",
        "vocab_size": enc.n_vocab,
        "eot_token_id": eot_id,
        "train_tokens": int(wrote_train),
        "val_tokens": int(wrote_val),
    }
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # helpful prints, similar to template comments
    train_gb = wrote_train * 2 / (1024**3)
    val_mb = wrote_val * 2 / (1024**2)
    print(f"train.bin ~{train_gb:.2f} GB ({wrote_train:,} tokens)")
    print(f"val.bin   ~{val_mb:.2f} MB ({wrote_val:,} tokens)")

if __name__ == "__main__":
    main()
