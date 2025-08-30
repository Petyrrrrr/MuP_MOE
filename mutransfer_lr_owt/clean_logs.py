#!/usr/bin/env python3
"""
Script to clean existing log files by removing ANSI escape sequences and empty lines.
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

def clean_log_file(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Clean a single log file by removing ANSI sequences and empty lines.
    
    Args:
        input_file: Path to the input log file
        output_file: Path to the output file (if None, overwrites input)
    """
    # If no output file specified, create a temp file and replace
    if output_file is None:
        output_file = input_file + ".tmp"
        replace_original = True
    else:
        replace_original = False
    
    # ANSI escape sequence pattern
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    cleaned_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # Remove ANSI escape sequences
                line = ansi_escape.sub('', line)
                
                # Remove carriage returns
                line = line.replace('\r', '')
                
                # Strip trailing whitespace
                line = line.rstrip()
                
                # Only keep non-empty lines
                if line:
                    cleaned_lines.append(line)
    
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    # Write cleaned content
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(line + '\n')
        
        if replace_original:
            # Replace original file
            os.replace(output_file, input_file)
            print(f"Cleaned: {input_file}")
        else:
            print(f"Cleaned: {input_file} -> {output_file}")
            
    except Exception as e:
        print(f"Error writing {output_file}: {e}")
        if replace_original and os.path.exists(output_file):
            os.remove(output_file)

def clean_directory(directory: str, pattern: str = "*.err") -> None:
    """
    Clean all matching files in a directory.
    
    Args:
        directory: Directory path containing log files
        pattern: Glob pattern for files to clean (default: *.err)
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory does not exist: {directory}")
        return
    
    files = list(dir_path.glob(pattern))
    
    if not files:
        print(f"No files matching pattern '{pattern}' found in {directory}")
        return
    
    print(f"Found {len(files)} files to clean")
    
    for file_path in files:
        clean_log_file(str(file_path))
    
    print(f"Finished cleaning {len(files)} files")

def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Clean a single file:    python clean_logs.py <file_path>")
        print("  Clean directory (*.err): python clean_logs.py <directory>")
        print("  Clean directory pattern: python clean_logs.py <directory> <pattern>")
        print("")
        print("Examples:")
        print("  python clean_logs.py /path/to/logfile.err")
        print("  python clean_logs.py /home/ubuntu/MuP_MOE/std_out/mutransfer_lr_owt/20250830_034517/stderr/")
        print("  python clean_logs.py /path/to/logs/ '*.log'")
        sys.exit(1)
    
    target = sys.argv[1]
    
    if os.path.isfile(target):
        # Clean single file
        output = sys.argv[2] if len(sys.argv) > 2 else None
        clean_log_file(target, output)
    elif os.path.isdir(target):
        # Clean directory
        pattern = sys.argv[2] if len(sys.argv) > 2 else "*.err"
        clean_directory(target, pattern)
    else:
        print(f"Error: '{target}' is not a valid file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()