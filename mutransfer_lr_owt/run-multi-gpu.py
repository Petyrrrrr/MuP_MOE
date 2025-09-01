#!/usr/bin/env python3
"""
Ultimate multi-GPU runner combining clean output processing with advanced monitoring features.
"""

import os
import subprocess
import time
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from dataclasses import dataclass, asdict
import threading
import queue

@dataclass
class Job:
    """Represents a single training job."""
    job_id: int
    config: Dict
    gpu_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    return_code: Optional[int] = None
    
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class CleanOutputProcessor:
    """Handles real-time output cleaning and logging."""
    
    def __init__(self, output_file: str, is_stderr: bool = False):
        self.output_file = output_file
        self.is_stderr = is_stderr
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.file_handle = open(output_file, 'w')
        self.lock = threading.Lock()
        
    def clean_line(self, line: str) -> str:
        """Remove ANSI escape sequences and clean up the line."""
        # Remove ANSI escape sequences
        line = self.ansi_escape.sub('', line)
        # Remove carriage returns
        line = line.replace('\r', '')
        # Strip trailing whitespace
        line = line.rstrip()
        return line
    
    def write_line(self, line: str):
        """Write a cleaned line to the output file."""
        cleaned_line = self.clean_line(line)
        if cleaned_line:  # Only write non-empty lines
            with self.lock:
                self.file_handle.write(cleaned_line + '\n')
                self.file_handle.flush()
    
    def close(self):
        """Close the file handle."""
        self.file_handle.close()

class MultiGPURunner:
    """Manages multi-GPU job execution with clean output and advanced monitoring."""
    
    def __init__(self, num_gpus: int = 8, max_jobs_per_gpu: int = 1, clean_output: bool = True):
        self.num_gpus = num_gpus
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.clean_output = clean_output
        self.log_dir = None
        self.timestamp = None
        self.jobs = []
        self.lock = threading.Lock()
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.job_queue = queue.Queue()
        self.active_processors = {}  # Track active output processors
        
    def create_log_directory(self, base_path: str = "/home/ubuntu/MuP_MOE/std_out/mutransfer_lr_owt") -> str:
        """Create and return the log directory path with organized structure."""
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
        
        widths = [256]
        num_exps = [16, 8, 4, 2]
        lrs = [0.015625, 0.0078125, 0.00390625, 0.001953125,]
        seeds = [1]
        max_iters = 4000  # Configuration parameter for max iterations
        warmup_iters = 1000  # Configuration parameter for warmup iterations
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
                            'n_layer': 8,  # Fixed for now
                            'max_iters': max_iters,
                            'warmup_iters': warmup_iters,  # warmup_iters equals max_iters
                            'router_lr_mult': 0.5,  # Multiplier for router learning rate
                            'moe_bias_lr' : lr,
                            'moe_tau' : 0.1,
                            'init_std' : 0.02,
                            'gradient_accumulation' : 8,
                            'batch_size' : 16
                        }
                        configs.append(config)
        
        return configs
    
    def build_command(self, config: Dict) -> Tuple[List[str], str]:
        """Build the command list for a given configuration."""
        out_dir = f"run_data/mutransfer_lr_owt/out_{self.timestamp}/width{config['width']}_depth{config['n_layer']}_experts{config['num_exp']}_active{config['num_act']}_seed{config['seed']}_lr{config['lr']}"
        
        cmd_args = [
            "python3", "-u", "train.py",  # -u for unbuffered output
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
            f"--warmup_iters={config['warmup_iters']}",
            "--dataset=openwebtext",
            f"--gradient_accumulation_steps={config['gradient_accumulation']}",
            f"--batch_size={config['batch_size']}",
            "--block_size=1024",
            f"--n_layer={config['n_layer']}",
            f"--n_head={config['n_heads']}",
            f"--n_embd={config['width']}",
            "--dropout=0.0",
            "--bias=False",
            f"--init_std={config['init_std']}",
            f"--learning_rate={config['lr']}",
            "--lr_decay_iters=2000",
            f"--min_lr={config['min_lr']}",
            f"--max_iters={config['max_iters']}",
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
            f"--moe_tau={config['moe_tau']}",
            f"--moe_bias_lr={config['moe_bias_lr']}",
            "--moe_bias_momentum=0.9",
            f"--router_lr_mult={config.get('router_lr_mult', 1.0)}",
            "--moe_bias_momentum_enabled=True",
            "--moe_load_balance_method=bias",
            "--moe_aux_loss_weight=1.0",
            f"--seed={config['seed']}",
            "--backend=nccl",
            "--device=cuda",
            "--dtype=float16",
            "--compile=False"
        ]
        
        return cmd_args, out_dir
    
    def process_output_stream(self, stream, processor: CleanOutputProcessor):
        """Process output stream line by line with cleaning."""
        try:
            for line in stream:
                processor.write_line(line)
        finally:
            processor.close()
    
    def run_job(self, job: Job, gpu_id: int) -> bool:
        """Run a single job on the specified GPU with clean output processing."""
        job.gpu_id = gpu_id
        job.status = "running"
        job.start_time = time.time()
        
        # Build command
        cmd_args, out_dir = self.build_command(job.config)
        
        # Create descriptive filename
        job_desc = f"job_{job.job_id:04d}_gpu{gpu_id}_w{job.config['width']}_exp{job.config['num_exp']}_lr{job.config['lr']:.2e}_seed{job.config['seed']}"
        
        # File paths
        log_file = self.log_dir / "stdout" / f"{job_desc}.log"
        err_file = self.log_dir / "stderr" / f"{job_desc}.err"
        meta_file = self.log_dir / "metadata" / f"{job_desc}.json"
        
        # Save initial metadata
        metadata = {
            "job_id": job.job_id,
            "gpu_id": gpu_id,
            "config": job.config,
            "command": " ".join(cmd_args),
            "out_dir": out_dir,
            "start_time": job.start_time,
            "log_file": str(log_file),
            "err_file": str(err_file),
            "clean_output": self.clean_output
        }
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[GPU {gpu_id}] Starting job {job.job_id}: w={job.config['width']}, exp={job.config['num_exp']}, lr={job.config['lr']:.2e}")
        
        # Set environment for this GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered output
        
        # Run the command
        try:
            if self.clean_output:
                # Use clean output processing
                process = subprocess.Popen(
                    cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    universal_newlines=True,
                    bufsize=1  # Line buffered
                )
                
                # Create output processors
                stdout_processor = CleanOutputProcessor(str(log_file), is_stderr=False)
                stderr_processor = CleanOutputProcessor(str(err_file), is_stderr=True)
                
                # Create threads to process stdout and stderr
                stdout_thread = threading.Thread(
                    target=self.process_output_stream,
                    args=(process.stdout, stdout_processor)
                )
                stderr_thread = threading.Thread(
                    target=self.process_output_stream,
                    args=(process.stderr, stderr_processor)
                )
                
                stdout_thread.start()
                stderr_thread.start()
                
                # Wait for process to complete
                return_code = process.wait()
                
                # Wait for output threads to finish
                stdout_thread.join()
                stderr_thread.join()
                
            else:
                # Direct file output (no cleaning)
                with open(log_file, 'w') as stdout_file, open(err_file, 'w') as stderr_file:
                    process = subprocess.Popen(
                        cmd_args,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        env=env
                    )
                    return_code = process.wait()
            
            job.end_time = time.time()
            job.return_code = return_code
            
            if return_code == 0:
                job.status = "completed"
                with self.lock:
                    self.completed_jobs += 1
                print(f"[GPU {gpu_id}] Completed job {job.job_id} in {job.duration():.1f}s")
                success = True
            else:
                job.status = "failed"
                with self.lock:
                    self.failed_jobs += 1
                print(f"[GPU {gpu_id}] Failed job {job.job_id} with return code {return_code}")
                success = False
            
            # Update metadata with completion info
            metadata.update({
                "end_time": job.end_time,
                "duration": job.duration(),
                "status": job.status,
                "return_code": return_code
            })
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return success
            
        except Exception as e:
            job.status = "failed"
            job.end_time = time.time()
            with self.lock:
                self.failed_jobs += 1
            print(f"[GPU {gpu_id}] Exception in job {job.job_id}: {e}")
            
            # Update metadata with error info
            metadata.update({
                "end_time": job.end_time,
                "duration": job.duration(),
                "status": "failed",
                "error": str(e)
            })
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log exception to stderr file
            with open(err_file, 'a') as f:
                f.write(f"\nException occurred: {str(e)}\n")
            
            return False
    
    def print_progress(self):
        """Print detailed progress information."""
        total_jobs = len(self.jobs)
        pending_jobs = sum(1 for j in self.jobs if j.status == "pending")
        running_jobs = sum(1 for j in self.jobs if j.status == "running")
        
        print(f"\n{'='*60}")
        print(f"[Progress Report]")
        print(f"Total Jobs: {total_jobs}")
        print(f"Completed: {self.completed_jobs} ({self.completed_jobs/total_jobs*100:.1f}%)")
        print(f"Failed: {self.failed_jobs} ({self.failed_jobs/total_jobs*100:.1f}%)")
        print(f"Running: {running_jobs}")
        print(f"Pending: {pending_jobs}")
        print(f"{'='*60}\n")
    
    def save_job_summary(self):
        """Save a detailed summary of all jobs."""
        jobs_data = []
        for job in self.jobs:
            job_data = {
                "job_id": job.job_id,
                "config": job.config,
                "gpu_id": job.gpu_id,
                "status": job.status,
                "duration": job.duration(),
                "return_code": job.return_code
            }
            jobs_data.append(job_data)
        
        summary_file = self.log_dir / "job_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(jobs_data, f, indent=2)
    
    def run(self):
        """Main execution method."""
        print(f"Starting Ultimate Multi-GPU Runner")
        print(f"GPUs: {self.num_gpus}")
        print(f"Max jobs per GPU: {self.max_jobs_per_gpu}")
        print(f"Output cleaning: {'Enabled' if self.clean_output else 'Disabled'}")
        print(f"{'='*60}")
        
        # Create log directory
        self.create_log_directory()
        print(f"Logs directory: {self.log_dir}")
        
        # Generate all configurations
        configs = self.generate_configurations()
        print(f"Total jobs: {len(configs)}")
        
        # Create job objects
        self.jobs = [Job(job_id=i, config=cfg) for i, cfg in enumerate(configs)]
        
        # Save initial job list
        job_list_file = self.log_dir / "job_list.json"
        with open(job_list_file, 'w') as f:
            json.dump([{"job_id": j.job_id, "config": j.config} for j in self.jobs], f, indent=2)
        
        # Save runner configuration
        runner_config = {
            "timestamp": self.timestamp,
            "num_gpus": self.num_gpus,
            "max_jobs_per_gpu": self.max_jobs_per_gpu,
            "clean_output": self.clean_output,
            "total_jobs": len(self.jobs)
        }
        
        config_file = self.log_dir / "runner_config.json"
        with open(config_file, 'w') as f:
            json.dump(runner_config, f, indent=2)
        
        print(f"Starting job execution...")
        print(f"{'='*60}\n")
        
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
                    
                    # Estimate remaining time
                    completed = self.completed_jobs + self.failed_jobs
                    if completed > 0:
                        avg_time = elapsed / completed
                        remaining = len(self.jobs) - completed
                        est_remaining = avg_time * remaining
                        print(f"Elapsed: {elapsed/60:.1f} min | Est. remaining: {est_remaining/60:.1f} min")
                    
                    last_progress_time = current_time
        
        # Final summary
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"ALL JOBS COMPLETED")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Successful jobs: {self.completed_jobs}/{len(self.jobs)} ({self.completed_jobs/len(self.jobs)*100:.1f}%)")
        print(f"Failed jobs: {self.failed_jobs}/{len(self.jobs)} ({self.failed_jobs/len(self.jobs)*100:.1f}%)")
        
        if total_time > 0:
            print(f"Average job time: {total_time/len(self.jobs):.1f} seconds")
            print(f"Throughput: {len(self.jobs)/total_time*60:.2f} jobs/minute")
        
        print(f"\nResults directory: run_data/mutransfer_lr_owt/out_{self.timestamp}")
        print(f"Logs directory: {self.log_dir}")
        
        # Save final job summary
        self.save_job_summary()
        
        # Save final summary
        summary_file = self.log_dir / "final_summary.json"
        summary = {
            "timestamp": self.timestamp,
            "total_jobs": len(self.jobs),
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "total_time_minutes": total_time / 60,
            "average_job_time_seconds": total_time / len(self.jobs) if len(self.jobs) > 0 else 0,
            "throughput_jobs_per_minute": len(self.jobs) / total_time * 60 if total_time > 0 else 0,
            "num_gpus": self.num_gpus,
            "max_jobs_per_gpu": self.max_jobs_per_gpu,
            "clean_output": self.clean_output
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")

def main():
    """Entry point with command-line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Multi-GPU Runner")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="Max concurrent jobs per GPU")
    parser.add_argument("--no-clean", action="store_true", help="Disable output cleaning")
    
    args = parser.parse_args()
    
    runner = MultiGPURunner(
        num_gpus=args.gpus,
        max_jobs_per_gpu=args.jobs_per_gpu,
        clean_output=not args.no_clean
    )
    runner.run()

if __name__ == "__main__":
    main()