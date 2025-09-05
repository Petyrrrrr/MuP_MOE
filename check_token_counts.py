#!/usr/bin/env python3
"""
Check token counts in .bin files

Usage examples:
    # Check in current directory
    python check_token_counts.py
    
    # Check in specific directory
    python check_token_counts.py --dir /home/ubuntu/MuP_MOE/data/cccc
    python check_token_counts.py --dir /home/ubuntu/MuP_MOE/data/openwebtext
    
    # Check specific files
    python check_token_counts.py --dir /path/to/data --files train.bin val.bin test.bin
"""

import os
import numpy as np
import argparse
from pathlib import Path

def format_number(n):
    """Format large numbers with commas and billions notation"""
    billions = n / 1e9
    return f"{n:,} ({billions:.3f}B)"

def check_bin_file(filepath):
    """Check token count in a single .bin file"""
    if not os.path.exists(filepath):
        return None
    
    arr = np.memmap(filepath, dtype=np.uint16, mode='r')
    return len(arr)

def main():
    parser = argparse.ArgumentParser(description='Check token counts in .bin files')
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing the .bin files (default: current directory)')
    parser.add_argument('--files', nargs='+', default=['train.bin', 'val.bin'],
                        help='List of .bin files to check (default: train.bin val.bin)')
    args = parser.parse_args()
    
    # Convert to Path object
    directory = Path(args.dir)
    
    print(f"Checking token counts in: {directory.absolute()}")
    print("=" * 60)
    
    total_tokens = 0
    file_sizes = []
    
    for filename in args.files:
        filepath = directory / filename
        token_count = check_bin_file(filepath)
        
        if token_count is None:
            print(f"\n{filename}: NOT FOUND")
            continue
        
        total_tokens += token_count
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        file_sizes.append((filename, token_count, file_size_mb))
        
        print(f"\n{filename}:")
        print(f"  Tokens: {format_number(token_count)}")
        print(f"  File size: {file_size_mb:.1f} MB")
    
    if file_sizes:
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        
        for filename, count, size_mb in file_sizes:
            print(f"{filename:<15} {format_number(count):<30} {size_mb:>10.1f} MB")
        
        print("-" * 60)
        print(f"{'TOTAL':<15} {format_number(total_tokens):<30} {sum(s for _, _, s in file_sizes):>10.1f} MB")
        
        # Calculate splits percentage
        if len(file_sizes) == 2 and all(name in ['train.bin', 'val.bin'] for name, _, _ in file_sizes):
            train_tokens = next((c for n, c, _ in file_sizes if n == 'train.bin'), 0)
            val_tokens = next((c for n, c, _ in file_sizes if n == 'val.bin'), 0)
            if total_tokens > 0:
                print(f"\nSplit ratio: train={train_tokens/total_tokens*100:.2f}%, val={val_tokens/total_tokens*100:.2f}%")

if __name__ == '__main__':
    main()
