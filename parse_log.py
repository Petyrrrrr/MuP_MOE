import re

def parse_validation_loss(log_file_path):
    result = {}
    pattern = r'Validation - step (\d+): val loss ([\d.]+)'

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    result[step] = loss
    except FileNotFoundError:
        print(f"Error: File {log_file_path} not found")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}

    return result


def find_job_file(base_path, time_dir, job_prefix):
    """
    Find a job log file and extract parameters from its filename.

    Args:
        base_path (str): Base directory path (e.g., "../std_out/mutransfer_lr_owt")
        time_dir (str): Time directory name (e.g., "20250924_044616")
        job_prefix (str): Job prefix (e.g., "job_0008")

    Returns:
        tuple: (filename, params_dict) where params_dict contains:
               {width, num_exp, lr, seed}
               Returns (None, None) if no matching file found
    """
    import os
    import glob

    # Construct the search path
    search_pattern = os.path.join(base_path, time_dir, "stdout", f"{job_prefix}_*.log")

    # Find matching files
    matches = glob.glob(search_pattern)

    if not matches:
        return None, None

    # Take the first match if multiple exist
    full_path = matches[0]
    filename = os.path.basename(full_path)

    # Parse filename: job_0008_gpu7_w256_exp4_lr4.52e-02_seed0.log
    pattern = r'job_\d+_gpu\d+_w(\d+)_exp(\d+)_lr([\d.e\-+]+)_seed(\d+)\.log'
    match = re.match(pattern, filename)

    if not match:
        return filename, None

    params = {
        'width': int(match.group(1)),
        'num_exp': int(match.group(2)),
        'lr': float(match.group(3)),
        'seed': int(match.group(4))
    }

    return filename, params


if __name__ == "__main__":
    # Test parse_validation_loss
    test_file = "/home/ubuntu/MuP_MOE/std_out/mutransfer_lr_owt/20250924_044616/stdout/job_0012_gpu3_w256_exp8_lr5.66e-03_seed0.log"
    result = parse_validation_loss(test_file)
    print(f"Parsed {len(result)} validation steps")
    print("Sample output:", dict(list(result.items())[:3]))

    # Test find_job_file
    print("\n--- Testing find_job_file ---")
    base_path = "/home/ubuntu/MuP_MOE/std_out/mutransfer_lr_owt"
    time_dir = "20250924_044616"
    job_prefix = "job_0008"

    filename, params = find_job_file(base_path, time_dir, job_prefix)
    print(f"Filename: {filename}")
    print(f"Parameters: {params}")