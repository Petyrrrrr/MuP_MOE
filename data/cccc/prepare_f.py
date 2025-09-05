#!/usr/bin/env python3
# High-throughput C4 (English) preparation script
# Focus: download + tokenize as fast as possible (approximate token budget)
#
# Strategy
# --------
# * Use Hugging Face Datasets in streaming mode (no Apache Beam build).
# * Spawn multiple worker processes; each reads a disjoint contiguous shard:
#     ds = load_dataset(..., streaming=True).shard(num_shards=world, index=rank, contiguous=True)
# * Each worker tokenizes with tiktoken GPT-2 BPE and writes a part file:
#     {split}.part{rank}.bin (dtype=uint16, ids concatenated)
# * We stop when the worker surpasses its token goal (no exactness required).
# * At the end, parts are concatenated into {split}.bin and parts are removed.
#
# Defaults aim for ~10B train tokens and ~10M val tokens, but you can tweak.
#
# Reference template and writing pattern inspired by user-provided script.


import argparse
import math
import os
import sys
import time
import shutil
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset


# ----------------------------- helpers -----------------------------

def get_eot_id(enc: "tiktoken.Encoding") -> int:
    # Prefer encoding-provided special; fall back to canonical 50256
    e = getattr(enc, "eot_token", None)
    if isinstance(e, int):
        return e
    try:
        v = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        if len(v) == 1:
            return v[0]
    except Exception:
        pass
    return 50256


@dataclass
class WorkerCfg:
    config: str
    split: str
    rank: int
    world_size: int
    out_dir: str
    tmp_dir: str
    target_tokens: int
    flush_tokens: int
    seed: int
    shuffle_buffer: int
    trust_remote_code: bool


def _worker_main(cfg: WorkerCfg) -> Tuple[int, int, str]:
    """Return (tokens_written, docs_seen, part_path)."""
    enc = tiktoken.get_encoding("gpt2")
    eot = get_eot_id(enc)

    # Build streaming dataset pipeline in the worker
    ds = load_dataset(
        "c4",
        cfg.config,
        split=cfg.split,
        streaming=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    # Optional: a small shuffle buffer can help load-balance different-length docs
    if cfg.shuffle_buffer > 0:
        ds = ds.shuffle(seed=cfg.seed + cfg.rank, buffer_size=cfg.shuffle_buffer)

    # Disjoint contiguous shard for this worker
    if cfg.world_size > 1:
        ds = ds.shard(num_shards=cfg.world_size, index=cfg.rank, contiguous=True)

    os.makedirs(cfg.tmp_dir, exist_ok=True)
    part_path = os.path.join(cfg.tmp_dir, f"{cfg.split}.part{cfg.rank:02d}.bin")

    # Binary stream append (fast); we don't preallocate here
    f = open(part_path, "wb", buffering=1024 * 1024)  # 1MB buffered writes

    tokens_written = 0
    docs_seen = 0
    buf: List[int] = []
    target = cfg.target_tokens

    # Lightweight progress every N docs
    report_every = 2000

    try:
        for ex in ds:
            text = ex.get("text", None)
            if not text:
                continue

            ids = enc.encode_ordinary(text)
            ids.append(eot)

            buf.extend(ids)
            docs_seen += 1

            # Periodic flush to disk in big chunks
            if len(buf) >= cfg.flush_tokens:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                tokens_written += len(buf)
                buf.clear()

                # stop condition (approx; we allow slight overrun at next flush)
                if tokens_written >= target:
                    break

            if docs_seen % report_every == 0 and tokens_written < target:
                # progress line per worker
                print(f"[{cfg.split}][rank={cfg.rank}] docs={docs_seen:,} toks~={tokens_written:,}", flush=True)

        # Final flush
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            tokens_written += len(buf)
            buf.clear()
    finally:
        f.close()

    return tokens_written, docs_seen, part_path


def _concat_parts(parts: List[str], out_path: str) -> int:
    """Concatenate binary parts -> out_path. Return total tokens (uint16 count)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb", buffering=1024 * 1024) as w:
        for p in parts:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w, length=1024 * 1024 * 32)  # 32MB chunks
    total_bytes = os.path.getsize(out_path)
    return total_bytes // 2  # uint16


def _clean_parts(parts: List[str]):
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass


def _spawn_workers_and_collect(
    split: str,
    config: str,
    out_dir: str,
    tmp_dir: str,
    total_tokens: int,
    workers: int,
    flush_tokens: int,
    seed: int,
    shuffle_buffer: int,
    trust_remote_code: bool,
) -> Tuple[int, List[str]]:
    """Launch workers, wait, concat parts. Returns (tokens_total, part_paths)."""
    import multiprocessing as mp

    world = max(1, int(workers))
    per_worker = math.ceil(total_tokens / world)

    print(f"[{split}] target ~{total_tokens:,} tokens across {world} workers (~{per_worker:,} each).")

    # Start pool with 'spawn' for portability
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=world) as pool:
        jobs = []
        for rank in range(world):
            cfg = WorkerCfg(
                config=config,
                split=split,
                rank=rank,
                world_size=world,
                out_dir=out_dir,
                tmp_dir=tmp_dir,
                target_tokens=per_worker,
                flush_tokens=flush_tokens,
                seed=seed,
                shuffle_buffer=shuffle_buffer,
                trust_remote_code=trust_remote_code,
            )
            jobs.append(pool.apply_async(_worker_main, (cfg,)))

        parts = []
        tokens_sum = 0
        docs_sum = 0
        for j in jobs:
            toks, docs, path = j.get()
            parts.append(path)
            tokens_sum += toks
            docs_sum += docs
        print(f"[{split}] workers done: docs={docs_sum:,}, tokens~={tokens_sum:,} (pre-concat).")

    # concat
    out_path = os.path.join(out_dir, f"{split}.bin")
    print(f"[{split}] concatenating {len(parts)} parts -> {out_path} ...")
    tokens_total = _concat_parts(parts, out_path)
    print(f"[{split}] wrote {out_path} ({tokens_total:,} tokens; {(tokens_total*2)/(1024**3):.2f} GB).")

    # cleanup
    _clean_parts(parts)

    return tokens_total, parts


def main():
    parser = argparse.ArgumentParser(description="Fast C4 (en) prepare: multi-process streaming -> *.bin")
    parser.add_argument("--out-dir", type=str, default="data/c4_fast", help="Output directory")
    parser.add_argument("--config", type=str, default="en", help="C4 configuration, e.g., 'en' or 'en.noclean'")
    parser.add_argument("--train-tokens", type=int, default=10_000_000_000, help="Approx. train tokens to write")
    parser.add_argument("--val-tokens", type=int, default=10_000_000, help="Approx. val tokens to write")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for TRAIN split")
    parser.add_argument("--val-workers", type=int, default=2, help="Number of parallel workers for VAL split")
    parser.add_argument("--flush-tokens", type=int, default=1_000_000, help="Flush to disk after this many tokens per worker")
    parser.add_argument("--seed", type=int, default=1337, help="Shuffle seed")
    parser.add_argument("--shuffle-buffer", type=int, default=0, help="Optional streaming shuffle buffer per worker")
    parser.add_argument("--trust-remote-code", action="store_true", help="Skip interactive HF prompt for dataset code")
    parser.add_argument("--keep-parts", action="store_true", help="Keep per-worker part files after concat (debug)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_parts")
    os.makedirs(tmp_dir, exist_ok=True)

    # Train (parallel)
    train_tokens, _ = _spawn_workers_and_collect(
        split="train",
        config=args.config,
        out_dir=args.out_dir,
        tmp_dir=tmp_dir,
        total_tokens=args.train_tokens,
        workers=args.workers,
        flush_tokens=args.flush_tokens,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        trust_remote_code=args.trust_remote_code,
    )

    # Val (parallel but usually fewer workers)
    val_tokens, _ = _spawn_workers_and_collect(
        split="validation",
        config=args.config,
        out_dir=args.out_dir,
        tmp_dir=tmp_dir,
        total_tokens=args.val_tokens,
        workers=max(1, args.val_workers),
        flush_tokens=max(100_000, args.flush_tokens // 10),
        seed=args.seed + 1,
        shuffle_buffer=args.shuffle_buffer,
        trust_remote_code=args.trust_remote_code,
    )

    # meta
    enc = tiktoken.get_encoding("gpt2")
    meta = {
        "dataset": "c4",
        "config": args.config,
        "tokenizer": "tiktoken/gpt2",
        "vocab_size": enc.n_vocab,
        "eot_token_id": get_eot_id(enc),
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "notes": "fast streaming + multi-process; approximate token targets",
    }
    meta_path = os.path.join(args.out_dir, "meta.pkl")
    try:
        import pickle
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
    except Exception as e:
        print(f"WARNING: could not write meta.pkl: {e}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
