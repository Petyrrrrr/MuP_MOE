"""
Core training logic for GPT model with muP and MoE support.
Extracted from train.py to improve modularity.
"""

import os
import time
import math
import pickle
import csv
from contextlib import nullcontext
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from tqdm import tqdm
from utils import router_mult

class Trainer:
    def __init__(self, model, optimizer, config, device, master_process, ddp_settings=None):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            optimizer: The optimizer to use
            config: Configuration dictionary with all training parameters
            device: The device to train on
            master_process: Whether this is the master process (for logging)
            ddp_settings: Optional DDP settings dict with 'ddp', 'ddp_local_rank', 'ddp_world_size'
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.master_process = master_process
        self.ddp_settings = ddp_settings or {}
        self.ddp = ddp_settings.get('ddp', False) if ddp_settings else False
        
        # Extract frequently used config values
        self.eval_interval = config['eval_interval']
        self.log_interval = config['log_interval']
        self.eval_iters = config['eval_iters']
        self.eval_only = config['eval_only']
        self.always_save_checkpoint = config['always_save_checkpoint']
        self.never_save_checkpoint = config['never_save_checkpoint']
        self.gradient_accumulation_steps = config['gradient_accumulation_steps']
        self.grad_clip = config['grad_clip']
        self.max_iters = config['max_iters']
        self.out_dir = config['out_dir']
        self.wandb_log = config['wandb_log']
        self.csv_log = config['csv_log']
        self.n_layer = config['n_layer']
        self.n_embd = config['n_embd']
        self.num_exp = config['num_exp']
        self.learning_rate = config['learning_rate']
        self.dtype = config['dtype']
        self.compile = config['compile']
        self.mup_enabled = config['mup_enabled']
        self.mup_enable_coord_check_logging = config['mup_enable_coord_check_logging']
        self.moe_load_balance_method = config['moe_load_balance_method']
        self.moe_bias_lr = config['moe_bias_lr']
        self.moe_bias_momentum_enabled = config['moe_bias_momentum_enabled']
        self.router_lr_mult = config.get('router_lr_mult', 1.0)  # Default to 1.0 if not specified
        self.skip_val_loss = config['skip_val_loss']
        self.max_nan_losses = config.get('max_nan_losses', 50)  # Default to 50 if not specified
        self.bias_update_interval = config.get('bias_update_interval')
        
        # Get raw model (unwrap DDP if needed)
        self.raw_model = model.module if self.ddp else model
        
        # Initialize scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == 'float16'))
        
        # Setup context for mixed precision
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else nullcontext()
        
        # Initialize NaN loss counter
        self.nan_loss_count = 0
        
        # Initialize wandb and csv loggers if needed (handled by main script)
        self.wandb_run = None
        self.csv_logger = None
    
    def update_moe_stats(self, raw_model, num_exp, moe_load_balance_method, moe_bias_lr, moe_bias_momentum_enabled, iter_num):
        moe_layer_stats = []
        if num_exp > 1:
            with torch.no_grad():
                for i, block in enumerate(raw_model.transformer.h):
                    if hasattr(block, 'use_moe') and block.use_moe:
                        mlp_moe = block.mlp
                        if mlp_moe.total_tokens > 0 and iter_num % self.bias_update_interval == self.bias_update_interval - 1:
                            # Calculate average usage per expert
                            avg_usage = mlp_moe.tokens_per_expert / mlp_moe.total_tokens
                            target_usage = mlp_moe.num_act / mlp_moe.n_exp
                            # Store detailed stats for each layer
                            momentum_values = mlp_moe.bias_momentum_buffer.tolist() if moe_bias_momentum_enabled else None
                            moe_layer_stats.append({
                                'layer': i,
                                'usage': [f'{u:.3f}' for u in avg_usage.tolist()],
                                'bias': [f'{b:.3f}' for b in mlp_moe.bias.tolist()],
                                'momentum': [f'{m:.3f}' for m in momentum_values] if momentum_values else None,
                                'target': target_usage
                            })
                            # Update bias only if using bias method
                            if moe_load_balance_method == "bias":
                                mlp_moe.update_router_bias(avg_usage, target_usage, moe_bias_lr, iter_num, disable = False)
                            elif moe_load_balance_method == "aux_loss":
                                pass
                            mlp_moe.tokens_per_expert.zero_()
                            mlp_moe.total_tokens.zero_()
        return moe_layer_stats
    
    def setup_coord_check_hooks(self, model, mup_enable_coord_check_logging):
        if mup_enable_coord_check_logging:
            coord_check_dict = {
                'token_embedding': [],
                'attn': [],
                'mlp': [],
                'lm_head': [],
            }
            def hook(module, input, output, key):
                with torch.no_grad():
                    coord_check_dict[key].append(output.abs().mean().item())
            coord_check_handles = []
            for module_name, module in model.named_modules():
                if module_name == 'transformer.wte':
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='token_embedding')))
                elif module_name.endswith('.attn'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='attn')))
                elif module_name.endswith('.mlp'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='mlp')))
                elif module_name == 'lm_head':
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='lm_head')))
            return coord_check_dict, coord_check_handles
        else:
            return None, None
    
    def training_step(self, iter_num, scaler, ctx, gradient_accumulation_steps, grad_clip, get_batch_fn):
        """
        Execute forward/backward pass with gradient accumulation.
        
        Returns:
            Tuple of (loss, coord_check_dict, grad_norm) for logging
        """
        #set model to train
        self.model.train()
        loss = None
        grad_norm = None
        loss_sum = 0.0
        # Setup coordinate checking if enabled and first iteration
        coord_check_dict = None
        if iter_num % self.log_interval == 0:
            coord_check_dict, coord_check_handles = self.setup_coord_check_hooks(
                self.model, self.mup_enable_coord_check_logging
            )
        
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(gradient_accumulation_steps):
            if self.ddp:
                self.model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
            X, Y = get_batch_fn('train')  # NEW: different micro-batch each micro-step
            with ctx:
                logits, loss = self.model(X, Y)
                loss_sum += loss.item()
                loss_for_backward = loss / gradient_accumulation_steps
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss_for_backward).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            grad_norm = total_norm.item()
            
            # Only print warnings when gradient norm is problematic
            if self.master_process:
                if torch.isnan(total_norm):
                    print("WARNING: Gradient norm is NaN!")
                elif total_norm > grad_clip * 2:
                    print(f"WARNING: Large gradient norm {total_norm:.2f} (clip={grad_clip})")
                if total_norm > 1e5:
                    raise Exception(f"Gradient norm {total_norm:.2f} is too large, exceeding 1e5")
        # step the optimizer and scaler if training in fp16
        scaler.step(self.optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        self.optimizer.zero_grad(set_to_none=True)
        
        # Clean up coordinate check hooks if they were created
        if coord_check_dict is not None and self.mup_enable_coord_check_logging:
            for handle in coord_check_handles:
                handle.remove()
        
        return loss_sum/gradient_accumulation_steps, coord_check_dict, grad_norm
    
    def run_training_loop(self, get_batch_fn, estimate_loss_fn, get_lr_fn):
        """
        Main training loop orchestration.
        
        Args:
            get_batch_fn: Function to get batches, accepts 'train' or 'val'
            estimate_loss_fn: Function to estimate loss over multiple batches
            get_lr_fn: Function to get learning rate for current iteration
        """
        
        # Initialize state from config
        iter_num = self.config.get('iter_num', 0)
        best_val_loss = self.config.get('best_val_loss', 1e9)
        model_args = self.config.get('model_args', {})
        
        # Get logging objects from config
        if self.csv_log:
            self.csv_logger = self.config.get('csv_logger')
        if self.wandb_log:
            self.wandb_run = self.config.get('wandb_run')
        
        # Initialize for training loop
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        running_mfu = -1.0
        
        # Initialize tqdm progress bar (only on master process)
        pbar = None
        moe_pbars = []
        if self.master_process:
            pbar = tqdm(initial=iter_num, total=self.max_iters, desc="Training", 
                        unit="iter", dynamic_ncols=True, position=0)
            
            # Create separate progress bars for each MOE layer if MOE is enabled
            if self.num_exp > 1:
                for i in range(self.n_layer):
                    layer_pbar = tqdm(total=0, desc=f"L{i}: Initializing...", 
                                    unit="", leave=False, position=i+1, 
                                    bar_format='{desc}')
                    moe_pbars.append(layer_pbar)
        
        while True:
            # determine and set the learning rate for this iteration
            
            lr = get_lr_fn(iter_num)
            for param_group in self.optimizer.param_groups:
                if param_group.get('is_router', False):
                    param_group['lr'] = self.router_lr_mult * lr * param_group.get('lr_scale', 1.0) * router_mult(iter_num, self.max_iters)
                else:
                    param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
            
            # evaluate the loss on train/val sets and write checkpoints
            # if iter_num % self.eval_interval == 0 and self.master_process:
            #     losses = estimate_loss_fn()
            #     if np.isnan(losses['train']):
            #         self.nan_loss_count += 1
            #         if self.nan_loss_count > self.max_nan_losses:
            #             raise Exception(f'NaN loss encountered {self.nan_loss_count} times, exceeding max_nan_losses={self.max_nan_losses}')
            #         print(f"Warning: NaN loss detected ({self.nan_loss_count}/{self.max_nan_losses}), skipping update and continuing training")
            #         iter_num += 1
            #         continue
                
            #     log_dict = {
            #         "iter": iter_num,
            #         "train/loss": losses['train'],
            #         "val/loss": losses['val'],
            #         "lr": lr,
            #         "mfu": running_mfu*100,  # convert to percentage
            #     }
                
            #     # Add coordinate check logging if enabled
            #     if self.mup_enable_coord_check_logging and hasattr(self, '_last_coord_check_dict'):
            #         if self._last_coord_check_dict is not None:
            #             for key in self._last_coord_check_dict:
            #                 log_dict[key + '_act_abs_mean'] = np.mean(self._last_coord_check_dict[key])
                
            #     if self.wandb_log and self.wandb_run:
            #         self.wandb_run.log(log_dict)
            #     if self.csv_log and self.csv_logger:
            #         self.csv_logger.log(log_dict)
            #         self.csv_logger.step()
                
            #     if (not self.never_save_checkpoint) and (losses['val'] < best_val_loss or self.always_save_checkpoint):
            #         best_val_loss = losses['val']
            #         if iter_num > 0:
            #             checkpoint = {
            #                 'model': self.raw_model.state_dict(),
            #                 'optimizer': self.optimizer.state_dict(),
            #                 'model_args': model_args,
            #                 'iter_num': iter_num,
            #                 'best_val_loss': best_val_loss,
            #                 'config': self.config,
            #             }
            #             print(f"saving checkpoint to {self.out_dir}")
            #             torch.save(checkpoint, os.path.join(self.out_dir, 'ckpt.pt'))
            
            if iter_num == 0 and self.eval_only:
                break
            
            # Perform training step
            loss, coord_check_dict, grad_norm = self.training_step(iter_num, self.scaler, self.ctx, 
                self.gradient_accumulation_steps, self.grad_clip, get_batch_fn
            )
            
            # Store coord check dict for next eval
            if coord_check_dict is not None:
                self._last_coord_check_dict = coord_check_dict
            

            
            # Update router biases for MOE layers and collect stats for tqdm
            moe_layer_stats = self.update_moe_stats(
                self.raw_model, self.num_exp, self.moe_load_balance_method, 
                self.moe_bias_lr, self.moe_bias_momentum_enabled, iter_num
            )
            
            # Log expert usage to CSV files
            if self.csv_logger and moe_layer_stats:
                self.csv_logger.log_expert_usage(moe_layer_stats)
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.log_interval == 0 and self.master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                lossf = float(loss) if isinstance(loss, (float, int)) else (loss.item() if loss is not None else float('nan'))
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = self.raw_model.estimate_mfu(self.config['batch_size'] * self.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                
                # Update main progress bar with loss info
                postfix_dict = {
                    'loss': f'{lossf:.4f}',
                    'time': f'{dt*1000:.1f}ms',
                    'mfu': f'{running_mfu*100:.1f}%'
                }
                if grad_norm is not None:
                    postfix_dict['grad'] = f'{grad_norm:.3f}'
                pbar.set_postfix(postfix_dict)
                
                # Update MOE layer progress bars in place
                if moe_layer_stats and moe_pbars:
                    for stats in moe_layer_stats:
                        layer_idx = stats['layer']
                        if layer_idx < len(moe_pbars):
                            usage_str = ','.join(stats['usage'])
                            bias_str = ','.join(stats['bias'])
                            layer_desc = f"L{layer_idx}: usage[{usage_str}] bias[{bias_str}]"
                            moe_pbars[layer_idx].set_description(layer_desc)
            
            # Update progress bar
            if self.master_process:
                pbar.update(1)
            
            iter_num += 1
            local_iter_num += 1
            
            # termination conditions
            if iter_num > self.max_iters or iter_num % 1000 == 1:
                # Perform validation sweep before ending training
                if self.master_process:
                    print()
                    print("Performing validation sweep at iter_num " + str(iter_num))
                    print()
                    # Single validation pass that collects both loss and MOE stats
                    collect_moe = self.num_exp > 1
                    losses = estimate_loss_fn(override_skip_val=False, collect_moe_stats=collect_moe)
                    
                    if not np.isnan(losses['train']):  # Only log if not NaN
                        log_dict = {
                            "iter": iter_num,
                            "train/loss": losses['train'],
                            "val/loss": losses['val'],
                            "lr": lr,
                            "mfu": running_mfu*100, # convert to percentage
                        }
                        if iter_num > self.max_iters:
                            if self.mup_enable_coord_check_logging and hasattr(self, '_last_coord_check_dict'):
                                if self._last_coord_check_dict is not None:
                                    for key in self._last_coord_check_dict:
                                        log_dict[key + '_act_abs_mean'] = np.mean(self._last_coord_check_dict[key])
                            if self.wandb_log and self.wandb_run:
                                self.wandb_run.log(log_dict)
                            if self.csv_log and self.csv_logger:
                                self.csv_logger.log(log_dict)
                                self.csv_logger.step()
                                self.csv_logger.close()  # Ensure final row is written
                        print(f"Validation - step {iter_num}: val loss {losses['val']:.4f}")
                        
                        # Save router weights every 1000 iterations (including at max_iters)
                        if collect_moe and iter_num % 1000 == 1:
                            # Collect router weights from all layers
                            router_weights = []
                            for i, block in enumerate(self.raw_model.transformer.h):
                                if hasattr(block, 'use_moe') and block.use_moe:
                                    # Get router weight matrix (n_exp x n_embd)
                                    router_weight = block.mlp.router.weight.detach().cpu().numpy()
                                    router_weights.append(router_weight)
                            
                            if router_weights:
                                # Stack into (depth x n_exp x n_embd) array
                                router_weights_array = np.stack(router_weights, axis=0)
                                
                                
                                assert router_weights_array.shape == (self.n_layer, self.num_exp, self.n_embd)
                                
                                # Save to run_data directory
                                router_weights_path = os.path.join(self.out_dir, f'router_weights_iter_{iter_num}.npy')
                                np.save(router_weights_path, router_weights_array)
                                print(f"Router weights saved to {router_weights_path} (shape: {router_weights_array.shape})")
                        
                        # Print and save MOE expert usage statistics if collected
                        if collect_moe and 'moe_expert_usage' in losses:
                            expert_usage_matrix = losses['moe_expert_usage']
                            
                            print("\nValidation set expert usage statistics:")
                            for i, layer_usage in enumerate(expert_usage_matrix):
                                # Get layer info for display
                                target_usage = self.raw_model.transformer.h[i].mlp.num_act / self.raw_model.transformer.h[i].mlp.n_exp
                                usage_str = ','.join([f'{u:.3f}' for u in layer_usage])
                                bias_str = ','.join([f'{b:.3f}' for b in self.raw_model.transformer.h[i].mlp.bias.tolist()])
                                max_deviation = max(abs(u - target_usage) for u in layer_usage)
                                print(f"L{i}: usage[{usage_str}] bias[{bias_str}] target={target_usage:.3f} max_deviation={max_deviation:.3f}")
                            
                            # Save expert usage matrix to CSV (num_layers x num_experts format)
                            if iter_num > self.max_iters:
                                val_csv_path = os.path.join(self.out_dir, 'log_val.csv')
                                with open(val_csv_path, 'w', newline='') as f:
                                    writer = csv.writer(f)
                                    header = [f'E{i}' for i in range(self.num_exp)]
                                    writer.writerow(header)
                                    for layer_usage in expert_usage_matrix:
                                        writer.writerow([f'{usage:.6f}' for usage in layer_usage])
                                print(f"Expert usage matrix saved to {val_csv_path}")
            if iter_num > self.max_iters:
                break
        
        # Close progress bars
        if self.master_process:
            pbar.close()
            for moe_pbar in moe_pbars:
                moe_pbar.close()
