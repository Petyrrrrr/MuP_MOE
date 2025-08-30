#!/usr/bin/env python3
"""
Advanced multi-GPU runner with monitoring and dynamic load balancing.
"""

import os
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from dataclasses import dataclass
import threading

@dataclass
class Job:
    """Represents a single training job."""
    job_id: int
    config: Dict
    gpu_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class MultiGPURunner:
    """Manages multi-GPU job execution with monitoring."""
    
    def __init__(self, num_gpus: int = 8, max_jobs_per_gpu: int = 1):
        self.num_gpus = num_gpus
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.log_dir = None
        self.timestamp = None
        self.jobs = []
        self.lock = threading.Lock()
        self.completed_jobs = 0
        self.failed_jobs = 0
        
    def create_log_directory(self, base_path: str = "/home/ubuntu/MuP_MOE/std_out/mutransfer_lr_owt") -> str:
        """Create and return the log directory path."""
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(base_path) / self.timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.log_dir / "stdout").mkdir(exist_ok=True)
        (self.log_dir / "stderr").mkdir(exist_ok=True)
        (self.log_dir / "metadata").mkdir(exist_ok=True)
        
        return str(self.log_dir)
    
    def generate_configurations(self) -> List[Dict]:
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
                            'num_act': num_exp // 2,
                            'n_layer': 8  # Fixed for now
                        }
                        configs.append(config)
        
        return configs
    
    def build_command(self, config: Dict) -> Tuple[str, str]:
        """Build the command string for a given configuration."""
        out_dir = f"run_data/mutransfer_lr_owt/out_{self.timestamp}/width{config['width']}_depth{config['n_layer']}_experts{config['num_exp']}_active{config['num_act']}_seed{config['seed']}_lr{config['lr']}"
        
        cmd_args = [
            "python3", "train.py",
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
            f"--n_layer={config['n_layer']}",
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
    
    def run_job(self, job: Job, gpu_id: int) -> bool:
        """Run a single job on the specified GPU."""
        job.gpu_id = gpu_id
        job.status = "running"
        job.start_time = time.time()
        
        # Build command
        cmd, out_dir = self.build_command(job.config)
        
        # Create descriptive filename
        job_desc = f"job_{job.job_id:04d}_gpu{gpu_id}_w{job.config['width']}_exp{job.config['num_exp']}_lr{job.config['lr']:.2e}_seed{job.config['seed']}"
        
        # File paths
        log_file = self.log_dir / "stdout" / f"{job_desc}.log"
        err_file = self.log_dir / "stderr" / f"{job_desc}.err"
        meta_file = self.log_dir / "metadata" / f"{job_desc}.json"
        
        # Save metadata
        metadata = {
            "job_id": job.job_id,
            "gpu_id": gpu_id,
            "config": job.config,
            "command": cmd,
            "out_dir": out_dir,
            "start_time": job.start_time,
            "log_file": str(log_file),
            "err_file": str(err_file)
        }
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[GPU {gpu_id}] Starting job {job.job_id}: w={job.config['width']}, exp={job.config['num_exp']}, lr={job.config['lr']:.2e}")
        
        # Set environment for this GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Run the command
        try:
            with open(log_file, 'w') as stdout_file, open(err_file, 'w') as stderr_file:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env
                )
                return_code = process.wait()
            
            job.end_time = time.time()
            
            if return_code == 0:
                job.status = "completed"
                with self.lock:
                    self.completed_jobs += 1
                print(f"[GPU {gpu_id}] Completed job {job.job_id} in {job.duration():.1f}s")
                
                # Update metadata with completion info
                metadata["end_time"] = job.end_time
                metadata["duration"] = job.duration()
                metadata["status"] = "completed"
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return True
            else:
                job.status = "failed"
                with self.lock:
                    self.failed_jobs += 1
                print(f"[GPU {gpu_id}] Failed job {job.job_id} with return code {return_code}")
                
                # Update metadata with failure info
                metadata["end_time"] = job.end_time
                metadata["duration"] = job.duration()
                metadata["status"] = "failed"
                metadata["return_code"] = return_code
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return False
                
        except Exception as e:
            job.status = "failed"
            job.end_time = time.time()
            with self.lock:
                self.failed_jobs += 1
            print(f"[GPU {gpu_id}] Exception in job {job.job_id}: {e}")
            
            # Update metadata with error info
            metadata["end_time"] = job.end_time
            metadata["duration"] = job.duration()
            metadata["status"] = "failed"
            metadata["error"] = str(e)
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return False
    
    def print_progress(self):
        """Print progress information."""
        total_jobs = len(self.jobs)
        pending_jobs = sum(1 for j in self.jobs if j.status == "pending")
        running_jobs = sum(1 for j in self.jobs if j.status == "running")
        
        print(f"\n[Progress] Total: {total_jobs} | Completed: {self.completed_jobs} | "
              f"Failed: {self.failed_jobs} | Running: {running_jobs} | Pending: {pending_jobs}")
    
    def run(self):
        """Main execution method."""
        print(f"Starting multi-GPU runner with {self.num_gpus} GPUs")
        print(f"Max jobs per GPU: {self.max_jobs_per_gpu}")
        
        # Create log directory
        self.create_log_directory()
        print(f"Logs will be saved to: {self.log_dir}")
        
        # Generate all configurations
        configs = self.generate_configurations()
        print(f"Total number of jobs: {len(configs)}")
        
        # Create job objects
        self.jobs = [Job(job_id=i, config=cfg) for i, cfg in enumerate(configs)]
        
        # Save job list
        job_list_file = self.log_dir / "job_list.json"
        with open(job_list_file, 'w') as f:
            json.dump([{"job_id": j.job_id, "config": j.config} for j in self.jobs], f, indent=2)
        
        # Use ThreadPoolExecutor for parallel execution
        max_workers = self.num_gpus * self.max_jobs_per_gpu
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_job = {}
            gpu_assignment = 0
            
            for job in self.jobs:
                # Round-robin GPU assignment
                gpu_id = gpu_assignment % self.num_gpus
                gpu_assignment += 1
                
                future = executor.submit(self.run_job, job, gpu_id)
                future_to_job[future] = job
            
            # Monitor progress
            start_time = time.time()
            last_progress_time = start_time
            
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                
                # Print progress every 30 seconds
                current_time = time.time()
                if current_time - last_progress_time > 30:
                    self.print_progress()
                    elapsed = current_time - start_time
                    print(f"Elapsed time: {elapsed/60:.1f} minutes")
                    last_progress_time = current_time
        
        # Final summary
        print("\n" + "="*60)
        print("All jobs completed!")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Successful jobs: {self.completed_jobs}/{len(self.jobs)}")
        print(f"Failed jobs: {self.failed_jobs}/{len(self.jobs)}")
        print(f"Results saved in: run_data/mutransfer_lr_owt/out_{self.timestamp}")
        print(f"Logs saved in: {self.log_dir}")
        
        # Save summary
        summary_file = self.log_dir / "summary.json"
        summary = {
            "timestamp": self.timestamp,
            "total_jobs": len(self.jobs),
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "total_time_minutes": (time.time() - start_time) / 60,
            "num_gpus": self.num_gpus,
            "max_jobs_per_gpu": self.max_jobs_per_gpu
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Entry point."""
    runner = MultiGPURunner(num_gpus=8, max_jobs_per_gpu=1)
    runner.run()

if __name__ == "__main__":
    main()