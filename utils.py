import os
import json
import math
from pathlib import Path

import torch
import numpy as np

def bias_mult(it, max_iters):
    max_iter = 10 * max_iters
    warmup_iter = max_iter * 0.05
    if it < warmup_iter:
        return 1.0 * it / warmup_iter
    else:
        return 0.5 * (1.0 + math.cos(math.pi * (it - warmup_iter) / (max_iter - warmup_iter)))

def router_mult(iter_num, max_iters):
    return 1.0

def bias_update(usage, target):
    return (usage - target)


class DeterministicBatchLoader:
    """Serve deterministic micro-batches from pre-split batch files."""

    def __init__(self, data_dir, block_size, batch_size, grad_accum_steps, world_size):
        self.data_dir = Path(data_dir)
        self.split_root = self.data_dir / 'split'
        if not self.split_root.exists():
            raise FileNotFoundError(f"Deterministic split directory not found: {self.split_root}")

        self.block_size = block_size
        self.tokens_per_sequence = block_size + 1
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.world_size = world_size
        self.microbatches_per_iter = grad_accum_steps * world_size

        if self.grad_accum_steps <= 0 or self.world_size <= 0:
            raise ValueError("grad_accum_steps and world_size must be positive")

        self._metadata = {}
        self._num_batches = {}
        self._load_split_metadata('train')
        self._load_split_metadata('val')

        layout_shape = self._metadata['train']['layout']['shape']
        if len(layout_shape) != 2:
            raise ValueError(f"Unexpected layout shape in metadata: {layout_shape}")

        self.global_batch_size = layout_shape[0]
        expected_global_batch = self.batch_size * self.microbatches_per_iter
        if self.global_batch_size != expected_global_batch:
            raise ValueError(
                "Pre-split batch shape mismatch: expected {} sequences per batch, found {}."
                .format(expected_global_batch, self.global_batch_size)
            )

        tokens_per_sequence = self._metadata['train']['tokens_per_sequence']
        if tokens_per_sequence != self.tokens_per_sequence:
            raise ValueError(
                "Token sequence length mismatch: expected {}, found {} in metadata."
                .format(self.tokens_per_sequence, tokens_per_sequence)
            )

        self._cache = {
            'train': {'index': None, 'data': None},
            'val': {'index': None, 'data': None},
        }
        self._eval_cursors = {split: 0 for split in self._metadata}

    def _load_split_metadata(self, split):
        meta_path = self.split_root / split / 'metadata.json'
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata for split '{split}': {meta_path}")
        with meta_path.open('r', encoding='utf-8') as fh:
            metadata = json.load(fh)

        if metadata.get('dtype') != 'uint16':
            raise ValueError(f"Unexpected dtype in {meta_path}: {metadata.get('dtype')}")

        if metadata.get('tokens_per_input_sample') != self.block_size:
            raise ValueError(
                f"Block size mismatch for split '{split}': expected {self.block_size}, "
                f"found {metadata.get('tokens_per_input_sample')}"
            )

        self._metadata[split] = metadata
        self._num_batches[split] = metadata.get('num_batches', 0)

    def validate_max_iters(self, max_iters):
        if max_iters > self._num_batches['train']:
            raise ValueError(
                f"Requested max_iters={max_iters} exceeds available deterministic train batches "
                f"({self._num_batches['train']})."
            )

    def _load_batch(self, split, batch_idx):
        cache = self._cache[split]
        if cache['index'] != batch_idx:
            batch_path = self.split_root / split / f"batch_{batch_idx:05d}.bin"
            if not batch_path.exists():
                raise FileNotFoundError(f"Missing batch file: {batch_path}")
            memmap_mode = 'r+'
            try:
                cache['data'] = np.memmap(
                    batch_path,
                    dtype=np.uint16,
                    mode=memmap_mode,
                    shape=(self.global_batch_size, self.tokens_per_sequence),
                )
            except PermissionError:
                cache['data'] = np.memmap(
                    batch_path,
                    dtype=np.uint16,
                    mode='r',
                    shape=(self.global_batch_size, self.tokens_per_sequence),
                )
            cache['index'] = batch_idx
        return cache['data']

    def _slice_microbatch(self, split, batch_idx, micro_step, rank):
        if micro_step < 0 or micro_step >= self.grad_accum_steps:
            raise IndexError(f"micro_step {micro_step} out of range for grad_accum_steps={self.grad_accum_steps}")
        if rank < 0 or rank >= self.world_size:
            raise IndexError(f"rank {rank} out of range for world_size={self.world_size}")

        batch = self._load_batch(split, batch_idx)
        global_micro_index = micro_step * self.world_size + rank
        start = global_micro_index * self.batch_size
        end = start + self.batch_size
        if end > batch.shape[0]:
            raise IndexError(
                f"Slice [{start}:{end}] exceeds batch size {batch.shape[0]} for split '{split}' "
                f"(micro_step={micro_step}, rank={rank})"
            )
        return batch[start:end]

    def get_train_tokens(self, iter_num, micro_step, rank):
        if iter_num < 0 or iter_num >= self._num_batches['train']:
            raise IndexError(
                f"iter_num {iter_num} out of bounds for {self._num_batches['train']} train batches"
            )
        return self._slice_microbatch('train', iter_num, micro_step, rank)

    def get_train_batch(self, iter_num):
        if iter_num < 0 or iter_num >= self._num_batches['train']:
            raise IndexError(
                f"iter_num {iter_num} out of bounds for {self._num_batches['train']} train batches"
            )
        return self._load_batch('train', iter_num)

    def _total_eval_microbatches(self, split):
        return self._num_batches[split] * self.grad_accum_steps

    def next_eval_tokens(self, split, rank):
        if split not in self._metadata:
            raise ValueError(f"Unknown split '{split}'")
        total_micro = self._total_eval_microbatches(split)
        if total_micro == 0:
            raise ValueError(f"No batches available for split '{split}'")

        cursor = self._eval_cursors[split] % total_micro
        global_micro_index = cursor * self.world_size + rank
        batch_idx = (global_micro_index // self.microbatches_per_iter) % self._num_batches[split]
        micro_step = (global_micro_index % self.microbatches_per_iter) // self.world_size

        tokens = self._slice_microbatch(split, batch_idx, micro_step, rank)
        self._eval_cursors[split] = (cursor + 1) % total_micro
        return tokens

    def reset_eval_cursors(self):
        for split in self._eval_cursors:
            self._eval_cursors[split] = 0

    @property
    def train_num_batches(self):
        return self._num_batches['train']

    @property
    def val_num_batches(self):
        return self._num_batches['val']

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

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, max_iters, min_lr, decay_lr = False):
    max_iter = 10 * max_iters
    warmup_iter = max_iter * 0.05
    if it < warmup_iter:
        return 1.0 * learning_rate * it / warmup_iter
    else:
        return 0.5 * (1.0 + math.cos(math.pi * (it - warmup_iter) / (max_iter - warmup_iter))) * learning_rate
    # if it <= warmup_iters:
    #     return learning_rate * it / warmup_iters
    # else:
    #     return learning_rate * 0.5 * (1.0 + math.cos(math.pi * (it - warmup_iters) / (max_iters - warmup_iters)))
