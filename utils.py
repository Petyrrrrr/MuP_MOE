import os
import math
import torch
import numpy as np

def get_batch(split, data_dir, block_size, batch_size, device_type, device):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, skip_val_loss, get_batch_fn, ctx, collect_moe_stats=False, raw_model=None):
    """
    Estimate an arbitrarily accurate loss over train/val splits using many batches.
    
    Args:
        model: The model to evaluate
        eval_iters: Number of iterations to average loss over
        skip_val_loss: Whether to skip validation loss calculation
        get_batch_fn: Function to get batches, should accept 'train' or 'val' as argument
        ctx: Context manager for autocast if using mixed precision
        collect_moe_stats: Whether to collect MOE expert usage statistics during validation
        raw_model: Raw model (unwrapped from DDP) needed for MOE stats collection
    
    Returns:
        Dictionary with 'train' and 'val' loss values, and optionally 'moe_expert_usage'
    """
    out = {}
    model.eval()
    splits = ['train'] if skip_val_loss else ['train', 'val']
    
    # Initialize MOE stats collection if requested
    expert_usage_matrix = []
    if collect_moe_stats and not skip_val_loss and raw_model is not None:
        # Reset all counters before validation pass
        for block in raw_model.transformer.h:
            if hasattr(block, 'use_moe') and block.use_moe:
                block.mlp.tokens_per_expert.zero_()
                block.mlp.total_tokens.zero_()
    
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            with ctx:
                # Enable token counting for MOE during validation if requested
                if collect_moe_stats and split == 'val' and raw_model is not None:
                    # Temporarily enable training mode for token counting
                    original_training_modes = {}
                    for i, block in enumerate(raw_model.transformer.h):
                        if hasattr(block, 'use_moe') and block.use_moe:
                            mlp_moe = block.mlp
                            original_training_modes[i] = mlp_moe.training
                            mlp_moe.training = True
                
                logits, loss = model(X, Y)
                
                # Restore original training modes
                if collect_moe_stats and split == 'val' and raw_model is not None:
                    for i, mode in original_training_modes.items():
                        raw_model.transformer.h[i].mlp.training = mode
                
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    # Collect expert usage statistics if requested
    if collect_moe_stats and not skip_val_loss and raw_model is not None:
        for i, block in enumerate(raw_model.transformer.h):
            if hasattr(block, 'use_moe') and block.use_moe:
                mlp_moe = block.mlp
                if mlp_moe.total_tokens > 0:
                    # Calculate average usage per expert
                    avg_usage = mlp_moe.tokens_per_expert / mlp_moe.total_tokens
                    expert_usage_matrix.append(avg_usage.cpu().tolist())
                    # Reset counters after collection
                    mlp_moe.tokens_per_expert.zero_()
                    mlp_moe.total_tokens.zero_()
        
        out['moe_expert_usage'] = expert_usage_matrix
    
    if skip_val_loss:
        out['val'] = -1
    model.train()
    return out

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    return learning_rate
    # if it <= warmup_iters:
    #     return learning_rate * it / warmup_iters
    # if it > lr_decay_iters:
    #     return min_lr
    # decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    # coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # return min_lr + coeff * (learning_rate - min_lr)
