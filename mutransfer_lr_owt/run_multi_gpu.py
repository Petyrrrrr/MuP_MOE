#!/usr/bin/env python3
"""
Multi-GPU runner for muP hyperparameter transfer with MOE
Efficiently distributes different runs across available GPUs
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import queue
import threading

# Configuration
NUM_GPUS = 8
BASE_COMMAND = "python3 train.py"

def create_log_directory(base_path: str = "/home/ubuntu/MuP_MOE/std_out/mutransfer_lr_owt") -> str:
    """Create and return the log directory path."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(base_path) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir), timestamp

def generate_configurations() -> List[Dict]:
    """Generate all configurations to run."""
    configs = []
    
    widths = [512]
    num_exps = [8, 4, 2]
    lrs = [0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 
           0.0009765625, 0.00048828125, 0.000244140625, 0.0001220703125, 0.00006103515625]
    seeds = [1]
    
    for width in widths:
        for num_exp in num_exps:
            for lr in lrs:
                for seed in seeds:
                    config = {
                        'width': width,
                        'num_exp': num_exp,
                        'lr': lr,
                        'seed': seed,
                        'head_size': 64,
                        'n_heads': width // 64,
                        'min_lr': lr,
                        'mup_base_width': 256,
                        'mup_width_multiplier': width / 256,
                        'num_act': num_exp // 2
                    }
                    configs.append(config)
    
    return configs

def build_command(config: Dict, timestamp: str) -> Tuple[str, Dict]:
    """Build the command string for a given configuration."""
    out_dir = f"run_data/mutransfer_lr_owt/out_{timestamp}/width{config['width']}_depth8_experts{config['num_exp']}_active{config['num_act']}_seed{config['seed']}_lr{config['lr']}"
    
    cmd_args = [
        BASE_COMMAND,
        f"--out_dir={out_dir}",
        "--eval_interval=1",
        "--log_interval=1",
        "--eval_iters=1",
        "--eval_only=False",
        "--skip_val_loss=True",
        "--always_save_checkpoint=False",
        "--never_save_checkpoint=True",
        "--init_from=scratch",
        "--wandb_log=False",
        "--csv_log=True",
        "--warmup_iters=1000",
        "--dataset=openwebtext",
        "--gradient_accumulation_steps=16",
        "--batch_size=16",
        "--block_size=1024",
        "--n_layer=8",
        f"--n_head={config['n_heads']}",
        f"--n_embd={config['width']}",
        "--dropout=0.0",
        "--bias=False",
        "--init_std=0.02",
        f"--learning_rate={config['lr']}",
        "--lr_decay_iters=2000",
        f"--min_lr={config['min_lr']}",
        "--max_iters=1000",
        "--weight_decay=0.0",
        "--beta1=0.9",
        "--beta2=0.95",
        "--grad_clip=3.0",
        "--decay_lr=False",
        "--mup_enabled=True",
        f"--mup_width_multiplier={config['mup_width_multiplier']}",
        "--mup_input_alpha=1.0",
        "--mup_output_alpha=1.0",
        f"--num_exp={config['num_exp']}",
        f"--num_act={config['num_act']}",
        "--moe_tau=1.0",
        f"--moe_bias_lr={config['lr']}",
        "--moe_bias_momentum=0.9",
        "--moe_bias_momentum_enabled=True",
        "--moe_load_balance_method=bias",
        "--moe_aux_loss_weight=1.0",
        f"--seed={config['seed']}",
        "--backend=nccl",
        "--device=cuda",
        "--dtype=float16",
        "--compile=False"
    ]
    
    return " ".join(cmd_args), out_dir

def worker(gpu_id: int, job_queue: queue.Queue, log_dir: str, timestamp: str):
    """Worker thread that processes jobs on a specific GPU."""
    while True:
        try:
            job_id, config = job_queue.get(timeout=1)
        except queue.Empty:
            break
        
        if config is None:  # Poison pill
            break
        
        # Build command
        cmd, out_dir = build_command(config, timestamp)
        
        # Create log file paths
        log_file = f"{log_dir}/job_{job_id:04d}_gpu{gpu_id}_w{config['width']}_exp{config['num_exp']}_lr{config['lr']}_seed{config['seed']}.log"
        err_file = f"{log_dir}/job_{job_id:04d}_gpu{gpu_id}_w{config['width']}_exp{config['num_exp']}_lr{config['lr']}_seed{config['seed']}.err"
        
        print(f"[GPU {gpu_id}] Starting job {job_id}: width={config['width']}, num_exp={config['num_exp']}, lr={config['lr']}, seed={config['seed']}")
        print(f"[GPU {gpu_id}] Log: {log_file}")
        
        # Set environment for this GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Run the command
        with open(log_file, 'w') as stdout_file, open(err_file, 'w') as stderr_file:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=stdout_file,
                stderr=stderr_file,
                env=env
            )
            process.wait()
        
        print(f"[GPU {gpu_id}] Completed job {job_id}")
        job_queue.task_done()

def main():
    """Main function to orchestrate the multi-GPU runs."""
    print(f"Starting multi-GPU runner with {NUM_GPUS} GPUs")
    
    # Create log directory
    log_dir, timestamp = create_log_directory()
    print(f"Logs will be saved to: {log_dir}")
    
    # Generate all configurations
    configs = generate_configurations()
    print(f"Total number of jobs: {len(configs)}")
    
    # Create job queue
    job_queue = queue.Queue()
    
    # Add all jobs to the queue
    for job_id, config in enumerate(configs):
        job_queue.put((job_id, config))
    
    # Create and start worker threads
    threads = []
    for gpu_id in range(NUM_GPUS):
        t = threading.Thread(target=worker, args=(gpu_id, job_queue, log_dir, timestamp))
        t.start()
        threads.append(t)
    
    # Wait for all jobs to complete
    job_queue.join()
    
    # Send poison pills to stop workers
    for _ in range(NUM_GPUS):
        job_queue.put((None, None))
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    
    print(f"All jobs completed!")
    print(f"Results saved in: run_data/mutransfer_lr_owt/out_{timestamp}")
    print(f"Logs saved in: {log_dir}")

if __name__ == "__main__":
    main()