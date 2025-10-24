#!/usr/bin/env python3
"""Thin wrapper around the C4 tokenizer script for FineWeb data."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    c4_tokenizer = Path(__file__).resolve().parent.parent / "cccc" / "tokenize_pkl_to_bin.py"
    runpy.run_path(str(c4_tokenizer), run_name="__main__")


if __name__ == "__main__":
    main()

