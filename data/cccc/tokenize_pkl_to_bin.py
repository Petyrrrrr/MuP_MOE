#!/usr/bin/env python3
"""
tokenize_pkl_to_bin.py

Usage:
  python tokenize_pkl_to_bin.py --in train.pkl --out . --encoding gpt2 --add-eos

What it does:
- Loads a pickled dataset (train.pkl) that is either:
  * a list/tuple of strings
  * a list/tuple of dicts with a 'text' field
  * a pandas DataFrame with a 'text' column
  * a dict containing any of the above under one of: 'text', 'texts', 'data', 'documents'
  * pre-tokenized (list[int] or list[list[int]])
- Tokenizes with tiktoken (default: gpt2) and writes contiguous token streams:
    train.bin (95% of tokens), val.bin (5% of tokens)
- Automatically chooses dtype (uint16 if max token id < 65536 else uint32)
- Writes a meta.pkl with basic info

Notes:
- Pickle files can execute code on load; ONLY use on trusted files.
- If tiktoken is not available, the script tries a Hugging Face GPT2TokenizerFast fallback
  (which may require an internet download). Prefer installing `pip install tiktoken`.
"""

import argparse
import os
import pickle
import sys
from typing import Iterable, List, Union, Any, Tuple, Optional

try:
    import numpy as np
except Exception as e:
    print("This script requires numpy. Please install it with `pip install numpy`.", file=sys.stderr)
    raise

# ---------------------- Tokenizer wrapper ----------------------

class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2", add_bos: bool = False, add_eos: bool = False):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.backend = None
        self.encoding_name = encoding_name

        self._eos_id = None
        self._bos_id = None
        try:
            import tiktoken  # type: ignore
            self.backend = ("tiktoken", tiktoken.get_encoding(encoding_name))
            try:
                # Best-effort EOS detection
                if hasattr(self.backend[1], "eot_token"):
                    self._eos_id = self.backend[1].eot_token
                else:
                    # Common GPT-2 style EOS
                    self._eos_id = 50256 if "gpt2" in encoding_name or "r50k" in encoding_name else None
                # GPT-2 has no canonical BOS; leave None
                self._bos_id = None
            except Exception:
                pass
        except Exception:
            # Fallback to HF GPT-2
            try:
                from transformers import GPT2TokenizerFast  # type: ignore
                tok = GPT2TokenizerFast.from_pretrained("gpt2")
                self.backend = ("hf_gpt2", tok)
                self._eos_id = tok.eos_token_id
                self._bos_id = None
                if encoding_name not in ("gpt2", "r50k_base"):
                    print(f"[warn] tiktoken not found; using GPT2TokenizerFast fallback, ignoring --encoding {encoding_name}", file=sys.stderr)
            except Exception:
                print(
                    "No tokenizer available. Please install tiktoken (`pip install tiktoken`) "
                    "or transformers (`pip install transformers`).",
                    file=sys.stderr
                )
                raise

    @property
    def bos_id(self) -> Optional[int]:
        return self._bos_id

    @property
    def eos_id(self) -> Optional[int]:
        return self._eos_id

    def encode(self, text: str) -> List[int]:
        if self.backend is None:
            raise RuntimeError("Tokenizer backend not initialized.")
        name, enc = self.backend
        if name == "tiktoken" or name == "hf_gpt2":
            ids = enc.encode(text, allowed_special="all")
        else:
            raise RuntimeError(f"Unknown tokenizer backend: {name}")
        if self.add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

# ---------------------- Data loading & inspection ----------------------

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def _is_list_of_ints(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and all(isinstance(i, int) for i in x)

def _is_list_of_strs(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and all(isinstance(i, str) for i in x)

def _is_list_of_dicts_with_text(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(i, dict) and ("text" in i) for i in x)

def _maybe_df_with_text(obj: Any) -> Optional[Iterable[str]]:
    try:
        import pandas as pd  # type: ignore
        if isinstance(obj, pd.DataFrame):
            if "text" in obj.columns:
                return (str(t) for t in obj["text"].astype(str).tolist())
    except Exception:
        pass
    return None

def extract_text_iter(obj: Any) -> Optional[Iterable[str]]:
    # 1) plain list/tuple of strings
    if _is_list_of_strs(obj):
        return (str(t) for t in obj)  # type: ignore
    # 2) list/tuple of dicts with 'text' key
    if _is_list_of_dicts_with_text(obj):
        return (str(d["text"]) for d in obj)  # type: ignore
    # 3) pandas DataFrame with 'text' column
    maybe = _maybe_df_with_text(obj)
    if maybe is not None:
        return maybe
    # 4) dict wrappers
    if isinstance(obj, dict):
        for key in ("texts", "text", "documents", "data"):
            if key in obj:
                val = obj[key]
                if isinstance(val, str):
                    return (val,)
                if _is_list_of_strs(val):
                    return (str(t) for t in val)  # type: ignore
                if _is_list_of_dicts_with_text(val):
                    return (str(d["text"]) for d in val)  # type: ignore
    # 5) single big string
    if isinstance(obj, str):
        return (obj,)
    return None

def extract_token_iter(obj: Any) -> Optional[Iterable[List[int]]]:
    # Flat list of ints -> treat as one document of tokens
    if _is_list_of_ints(obj):
        return (list(obj),)  # type: ignore
    # List/tuple where each element is a list of ints
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and all(_is_list_of_ints(el) for el in obj):
        return (list(el) for el in obj)  # type: ignore
    # Dict wrappers for common fields
    if isinstance(obj, dict):
        for key in ("tokens", "token_ids", "data"):
            if key in obj:
                val = obj[key]
                if _is_list_of_ints(val):
                    return (list(val),)  # type: ignore
                if isinstance(val, (list, tuple)) and len(val) > 0 and all(_is_list_of_ints(el) for el in val):
                    return (list(el) for el in val)  # type: ignore
    return None

def make_source_iter(obj: Any, text_key: str = "text"):
    """
    Returns (iterable, is_text). If is_text is True, items are strings and need tokenization.
    If False, items are lists of token ids and should be used as-is.
    """
    tok_iter = extract_token_iter(obj)
    if tok_iter is not None:
        return tok_iter, False
    txt_iter = extract_text_iter(obj)
    if txt_iter is not None:
        return txt_iter, True
    if isinstance(obj, dict) and text_key in obj:
        val = obj[text_key]
        if isinstance(val, str):
            return (val,), True
        if _is_list_of_strs(val):
            return (str(t) for t in val), True  # type: ignore
    raise ValueError(
        "Could not interpret the pickle contents as texts or pre-tokenized tokens. "
        "Expected a list of strings, list of dicts with 'text', a DataFrame with 'text', "
        "a dict containing 'texts'/'text'/..., or pre-tokenized lists of ints."
    )

# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Path to input train.pkl")
    parser.add_argument("--out", dest="out_dir", default=".", help="Output directory (default: current dir)")
    parser.add_argument("--encoding", default="gpt2",
                        help="tiktoken encoding name (e.g. gpt2, r50k_base, p50k_base, cl100k_base, o200k_base)")
    parser.add_argument("--add-bos", action="store_true", help="Prepend BOS token if available")
    parser.add_argument("--add-eos", action="store_true", help="Append EOS token if available")
    parser.add_argument("--split", type=float, default=0.95, help="Train split ratio (default: 0.95)")
    parser.add_argument("--text-key", default="text", help="If your dicts use a custom text key (default: 'text')")
    parser.add_argument("--shuffle-docs", action="store_true",
                        help="Shuffle document order before streaming (split remains by tokens). Deterministic (seed=0).")
    args = parser.parse_args()

    in_path = args.in_path
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] loading pickle: {in_path}")
    obj = load_pickle(in_path)

    # Build source iterator
    source_iter, is_text = make_source_iter(obj, text_key=args.text_key)

    # Optional doc-level shuffle (only if materializable without huge memory).
    if args.shuffle_docs:
        print("[info] --shuffle-docs enabled; collecting documents into memory to shuffle...")
        docs = list(source_iter)
        import random
        random.Random(0).shuffle(docs)
        source_iter = docs

    tokenizer = Tokenizer(args.encoding, add_bos=args.add_bos, add_eos=args.add_eos) if is_text else None

    # First pass: count total tokens & max token id
    print("[info] first pass: counting tokens and max id... (this may take a while)")
    total = 0
    max_id = 0
    for item in source_iter:
        ids = tokenizer.encode(item) if tokenizer is not None else item  # type: ignore[arg-type]
        if not ids:
            continue
        total += len(ids)
        m = max(ids)
        if m > max_id:
            max_id = m
    print(f"[info] total tokens: {total:,}")
    print(f"[info] max token id: {max_id}")

    # Choose dtype
    dtype = np.uint16 if max_id < 65536 else np.uint32
    dtype_name = "uint16" if dtype == np.uint16 else "uint32"
    print(f"[info] using dtype: {dtype_name}")

    # Compute split sizes
    n_train = int(total * args.split)
    n_val = total - n_train
    print(f"[info] split: {args.split*100:.1f}%/{(1-args.split)*100:.1f}% -> train={n_train:,}, val={n_val:,}")

    train_path = os.path.join(out_dir, "train.bin")
    val_path   = os.path.join(out_dir, "val.bin")

    if total == 0:
        print("[warn] total tokens is 0; writing empty files.")
        open(train_path, "wb").close()
        open(val_path, "wb").close()
        meta = {
            "encoding": args.encoding,
            "add_bos": bool(args.add_bos),
            "add_eos": bool(args.add_eos),
            "dtype": dtype_name,
            "n_train": int(n_train),
            "n_val": int(n_val),
            "total_tokens": int(total),
            "split": float(args.split),
        }
        with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print(f"[done] wrote empty train.bin, val.bin, and meta.pkl in {out_dir}")
        return

    # Create memmaps
    print(f"[info] creating memmaps:\n       {train_path}\n       {val_path}")
    train_mm = np.memmap(train_path, dtype=dtype, mode='w+', shape=(n_train,))
    val_mm   = np.memmap(val_path,   dtype=dtype, mode='w+', shape=(n_val,))

    # Second pass: (re)build the iterator and write
    source_iter2, is_text2 = make_source_iter(obj, text_key=args.text_key)
    if args.shuffle_docs:
        docs2 = list(source_iter2)
        import random
        random.Random(0).shuffle(docs2)
        source_iter2 = docs2

    assert is_text == is_text2, "Internal error: tokenizer mismatch between passes."
    print("[info] second pass: writing tokens to train.bin/val.bin...")

    wrote = 0
    t_ptr = 0
    v_ptr = 0
    report_every = 5_000_000  # report every 5M tokens

    def _write_ids(ids_list: List[int]):
        nonlocal wrote, t_ptr, v_ptr
        if not ids_list:
            return
        arr = np.asarray(ids_list, dtype=dtype)
        end_wrote = wrote + len(arr)
        if end_wrote <= n_train:
            train_mm[t_ptr:t_ptr+len(arr)] = arr
            t_ptr += len(arr)
        else:
            if wrote < n_train:
                h = n_train - wrote
                if h > 0:
                    train_mm[t_ptr:t_ptr+h] = arr[:h]
                    t_ptr += h
                tail = arr[h:]
                if len(tail) > 0:
                    val_mm[v_ptr:v_ptr+len(tail)] = tail
                    v_ptr += len(tail)
            else:
                val_mm[v_ptr:v_ptr+len(arr)] = arr
                v_ptr += len(arr)
        wrote = end_wrote

    for item in source_iter2:
        ids = tokenizer.encode(item) if tokenizer is not None else item  # type: ignore
        if ids:
            _write_ids(ids)
        if wrote and wrote % report_every == 0:
            print(f"[info] wrote {wrote:,}/{total:,} tokens...")

    # Finalize
    train_mm.flush()
    val_mm.flush()
    del train_mm
    del val_mm

    # Write meta
    meta = {
        "encoding": args.encoding,
        "add_bos": bool(args.add_bos),
        "add_eos": bool(args.add_eos),
        "dtype": dtype_name,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "total_tokens": int(total),
        "split": float(args.split),
    }
    meta_path = os.path.join(out_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[done] wrote train.bin, val.bin, and meta.pkl in {out_dir}")
    print(f"[meta] {meta}")

if __name__ == "__main__":
    main()
