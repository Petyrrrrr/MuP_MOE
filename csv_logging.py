"""
Authored by Gavia Gray (https://github.com/gngdb)

Wrapper for wandb logging with efficient CSV logging and correct config JSON writing.
The CSV structure maintains a consistent order of keys based on their first appearance,
using a simple list for ordering. This ensures data integrity and allows for graceful
failure and manual recovery if needed.

Example usage:
  run = wandb.init(config=your_config)
  wrapper = LogWrapper(run, out_dir='path/to/output')

  ...
    # in train loop
    wrapper.log({"train/loss": 0.5, "train/accuracy": 0.9, "val/loss": 0.6, "val/accuracy": 0.85})
    wrapper.print("Train: {loss=:.4f}, {accuracy=:.2%}", prefix="train/")
    wrapper.print("Val: {loss=:.4f}, {accuracy=:.2%}", prefix="val/")
    wrapper.step()

  ...
    # at the end of your script
    wrapper.close()

  # If the script terminates unexpectedly, you can still recover the CSV using bash:
  # cat path/to/output/log_header.csv.tmp path/to/output/log_data.csv.tmp > path/to/output/log.csv
"""

import re
import os
import csv
import json
import atexit


def exists(x): return x is not None

def transform_format_string(s):
    """
    Transforms a string containing f-string-like expressions to a format
    compatible with str.format().

    This function converts expressions like '{var=}' or '{var=:formatting}'
    to 'var={var}' or 'var={var:formatting}' respectively. This allows
    for f-string-like syntax to be used with str.format().

    Args:
        s (str): The input string containing f-string-like expressions.

    Returns:
        str: The transformed string, compatible with str.format().

    Examples:
        >>> transform_format_string("Value is {x=}")
        "Value is x={x}"
        >>> transform_format_string("Formatted value is {x=:.2f}")
        "Formatted value is x={x:.2f}"
    """
    pattern = r'\{(\w+)=(:.[^}]*)?\}'
    return re.sub(pattern, lambda m: f"{m.group(1)}={{{m.group(1)}{m.group(2) or ''}}}", s)

class CSVLogWrapper:
    def __init__(self, logf=None, config={}, out_dir=None, flush_every: int = 100):
        self.logf = logf
        self.config = config
        self.log_dict = {}
        self.out_dir = out_dir
        self.csv_data_file = None
        self.csv_header_file = None
        self.csv_writer = None
        self.step_count = 0
        self.flush_every = flush_every  # how often to flush; 0 = never flush mid-run
        self.ordered_keys = []
        self.header_updated = False
        self.is_finalized = False
        self.no_sync_keyword = 'no_sync' # Keyword to prevent syncing to wandb
        
        # Expert usage tracking
        self.expert_usage_writers = {}
        self.expert_usage_files = {}
        self.num_experts = config.get('num_exp', 1)
        self.n_layer = config.get('n_layer', 0)
        self.track_expert_usage = self.num_experts > 1 and self.n_layer > 0

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            self.setup_csv_writer()
            self.write_config()
            if self.track_expert_usage:
                self.setup_expert_usage_writers()

        atexit.register(self.close)

    def setup_csv_writer(self):
        self.csv_data_path = os.path.join(self.out_dir, 'log_data.csv.tmp')
        self.csv_header_path = os.path.join(self.out_dir, 'log_header.csv.tmp')
        self.csv_data_file = open(self.csv_data_path, 'w', newline='')
        self.csv_header_file = open(self.csv_header_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_data_file)
    
    def setup_expert_usage_writers(self):
        """Setup CSV writers for expert usage tracking per layer."""
        for layer_idx in range(self.n_layer):
            csv_path = os.path.join(self.out_dir, f'expert_usage_layer_{layer_idx}.csv')
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            
            # Write header with expert columns
            header = [f'E{i}' for i in range(self.num_experts)]
            csv_writer.writerow(header)
            
            self.expert_usage_files[layer_idx] = csv_file
            self.expert_usage_writers[layer_idx] = csv_writer

    def write_config(self):
        if self.config:
            config_path = os.path.join(self.out_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(dict(**self.config), f, indent=2)

    def log(self, data):
        self.log_dict.update(data)
        for key in data:
            if key not in self.ordered_keys:
                self.ordered_keys.append(key)
                self.header_updated = True
    
    def log_expert_usage(self, moe_layer_stats):
        """Log expert usage statistics to per-layer CSV files."""
        if not self.track_expert_usage or not moe_layer_stats:
            return
        
        for stats in moe_layer_stats:
            layer_idx = stats['layer']
            if layer_idx in self.expert_usage_writers:
                # Convert usage strings back to floats and write
                usage_values = [float(u) for u in stats['usage']]
                self.expert_usage_writers[layer_idx].writerow(usage_values)
                
                # Flush periodically
                if self.flush_every and (self.step_count % self.flush_every == 0):
                    self.expert_usage_files[layer_idx].flush()

    def update_header(self):
        if self.header_updated:
            header = ['step'] + self.ordered_keys
            with open(self.csv_header_path, 'w', newline='') as header_file:
                csv.writer(header_file).writerow(header)
            self.header_updated = False

    def print(self, format_string, prefix=None):
        format_string = transform_format_string(format_string)

        if prefix:
            # Filter keys with the given prefix and remove the prefix
            filtered_dict = {k.replace(prefix, ''): v for k, v in self.log_dict.items() if k.startswith(prefix)}
        else:
            filtered_dict = self.log_dict
        # replace any '/' in keys with '_'
        filtered_dict = {k.replace('/', '_'): v for k, v in filtered_dict.items()}

        try:
            print(format_string.format(**filtered_dict))
        except KeyError as e:
            print(f"KeyError: {e}. Available keys: {', '.join(filtered_dict.keys())}")
            raise e

    def step(self):
        if exists(self.logf) and self.log_dict:
            self.logf({k: v for k, v in self.log_dict.items() if self.no_sync_keyword not in k})

        if self.csv_writer and self.log_dict:
            self.update_header()

            # Prepare the row data
            row_data = [self.step_count] + [self.log_dict.get(key, '') for key in self.ordered_keys]
            self.csv_writer.writerow(row_data)
            if self.flush_every and (self.step_count % self.flush_every == 0):
                self.csv_data_file.flush()

        self.step_count += 1
        self.log_dict.clear()

    def close(self):
        if self.csv_data_file:
            self.csv_data_file.flush()  # Ensure all data is written
            self.csv_data_file.close()
            self.csv_data_file = None  # Clear the reference
        
        # Close expert usage files
        for layer_idx, file in self.expert_usage_files.items():
            if file:
                file.flush()
                file.close()
        self.expert_usage_files.clear()
        self.expert_usage_writers.clear()
        
        # Small delay to let Windows release file handles
        import time
        time.sleep(0.05)
        
        self.finalize_csv()

    def finalize_csv(self):
        if self.is_finalized:
            return

        csv_final_path = os.path.join(self.out_dir, 'log.csv')

        try:
            with open(csv_final_path, 'w', newline='') as final_csv:
                # Copy header
                with open(self.csv_header_path, 'r') as header_file:
                    final_csv.write(header_file.read())

                # Copy data
                with open(self.csv_data_path, 'r') as data_file:
                    final_csv.write(data_file.read())
            self.is_finalized = True

            # Remove the temporary files with retry logic for Windows
            import time
            for file_path in [self.csv_header_path, self.csv_data_path]:
                for attempt in range(3):
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        break
                    except PermissionError:
                        if attempt < 2:  # Retry up to 3 times
                            time.sleep(0.1)  # Wait 100ms before retry
                        else:
                            # If all retries fail, just ignore the error
                            pass
        except Exception:
            # If finalization fails completely, just mark as finalized to prevent repeated attempts
            self.is_finalized = True
