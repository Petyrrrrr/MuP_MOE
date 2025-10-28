"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from functools import partial
import warnings
import random
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

# Suppress cuDNN SDPA stride warnings (harmless performance optimization messages)
warnings.filterwarnings("ignore", message=".*cuDNN SDPA backward.*", category=UserWarning)

from model import GPTConfig, GPT
from utils import get_batch, estimate_loss, get_lr, DeterministicBatchLoader
from trainer import Trainer

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
skip_val_loss = False # If True, will only measure train loss
always_save_checkpoint = True # if True, always save a checkpoint after each eval
never_save_checkpoint = False # if True, never save a checkpoint
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'cccc'
wandb_run_name = 'run' + str(time.time())
# csv logging
csv_log = False # If enabled, logs stats to a csv file
flush_every = 100 # how often to flush, set to 0 to only flush on close
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
init_std = 0.02 # Initialization standard deviation for weights
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 300 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# mup settings
mup_enabled = False # Whether to use muP. If False then all other mup variables are ignored
mup_disable_attention_scaling = False # Uses 1/sqrt(d_head) attn scaling instead of 1/d_head (Only needed for the step-by-step coord check in the blog)
mup_disable_hidden_lr_scaling = False # Disables muP hidden LR adjustment (Only needed for the step-by-step coord check in the blog)
mup_width_multiplier = 1.0 # mup_width_multiplier = width / base_width where base_width is typically 256
mup_input_alpha = 1.0 # Optional tunable multiplier applied to input embedding forward pass output
mup_output_alpha = 1.0 # Optional tunable multiplier applied to output unembedding forward pass output
mup_enable_coord_check_logging = False # If True will track the output.abs().mean() of various layers throughout training
# MOE settings
num_exp = 1 # Number of experts (set to 1 to disable MOE)
num_act = 1 # Number of active experts (top-k)
moe_tau = 1.0 # Temperature for router softmax
moe_bias_lr = 1e-2 # Learning rate for router bias updates (only used with bias method)
moe_bias_momentum = 0.9 # EMA decay factor for bias gradient momentum (only used with bias method)
moe_bias_momentum_enabled = True # Enable momentum for router bias updates (only used with bias method)
moe_load_balance_method = "bias" # "bias" or "aux_loss" - method for load balancing  
moe_aux_loss_weight = 0.01 # Auxiliary loss coefficient (only used with aux_loss method)
router_lr_mult = 1.0 # Multiplier for router learning rate (default 1.0)
alpha = 2.0 # Hidden layer size multiplier (hidden_size = alpha * n_embd)
max_nan_losses = 50 # Maximum number of NaN losses before raising error
bias_update_interval = 1 # Update bias every n iterations
# seed
seed = 1337
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

exec(open('configurator.py').read()) # overrides from command line or config file

config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

assert not (never_save_checkpoint and always_save_checkpoint)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    config['gradient_accumulation_steps'] = gradient_accumulation_steps
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
random.seed(seed + seed_offset)
np.random.seed(seed + seed_offset)
torch.cuda.manual_seed(seed + seed_offset)

torch.cuda.manual_seed_all(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data directory for the data loader
data_dir = os.path.join('data', dataset)

# Set up deterministic batch loader if split batches are available
deterministic_loader = None
if dataset in ['cccc', 'fineweb']:
    split_dir = os.path.join(data_dir, 'split')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Expected deterministic batches in {split_dir}. Run split_data_batches.py first."
        )
    deterministic_loader = DeterministicBatchLoader(
        data_dir=data_dir,
        block_size=block_size,
        batch_size=batch_size,
        grad_accum_steps=gradient_accumulation_steps,
        world_size=ddp_world_size,
    )
    deterministic_loader.validate_max_iters(max_iters)
    if master_process:
        print(f"Using deterministic batch loader from {split_dir}")

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, init_std=init_std, 
                  mup_enabled=mup_enabled, mup_disable_attention_scaling=mup_disable_attention_scaling,
                  mup_disable_hidden_lr_scaling=mup_disable_hidden_lr_scaling,
                  mup_width_multiplier=mup_width_multiplier, mup_input_alpha=mup_input_alpha,
                  mup_output_alpha=mup_output_alpha, num_exp=num_exp, num_act=num_act,
                  moe_tau=moe_tau, moe_bias_lr=moe_bias_lr, moe_bias_momentum=moe_bias_momentum,
                  moe_bias_momentum_enabled=moe_bias_momentum_enabled, moe_load_balance_method=moe_load_balance_method,
                  moe_aux_loss_weight=moe_aux_loss_weight, alpha=alpha, max_iters=max_iters, bias_update_interval=bias_update_interval) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # For MoE models, enable find_unused_parameters since not all experts are used for each token
    find_unused = num_exp > 1
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=find_unused)

# Cache for deterministic train batches (per iteration)
_det_train_cache = {'iter': None, 'tensor': None}


def _to_device_long(tensor, *, non_blocking):
    return tensor.to(device=device, dtype=torch.long, non_blocking=non_blocking)


def get_batch_wrapper(split, *, iter_num=None, micro_step=None):
    if deterministic_loader is not None:
        non_blocking = device_type == 'cuda'
        if split == 'train' and iter_num is not None and micro_step is not None:
            if _det_train_cache['iter'] != iter_num:
                global_tokens = deterministic_loader.get_train_batch(iter_num)
                tensor = torch.from_numpy(global_tokens)
                _det_train_cache['iter'] = iter_num
                _det_train_cache['tensor'] = tensor

            global_tensor = _det_train_cache['tensor']
            micro_batch_size = batch_size
            start = (micro_step * ddp_world_size + ddp_rank) * micro_batch_size
            micro_tokens = global_tensor.narrow(0, start, micro_batch_size)
            x = micro_tokens[:, :-1]
            y = micro_tokens[:, 1:]
            return _to_device_long(x, non_blocking=non_blocking), _to_device_long(y, non_blocking=non_blocking)

        eval_split = split
        if split == 'train':
            eval_split = 'train'
        tokens_np = deterministic_loader.next_eval_tokens(eval_split, ddp_rank)
        tokens_tensor = torch.from_numpy(tokens_np)
        x = tokens_tensor[:, :-1]
        y = tokens_tensor[:, 1:]
        return _to_device_long(x, non_blocking=non_blocking), _to_device_long(y, non_blocking=non_blocking)

    return get_batch(split, data_dir, block_size, batch_size, device_type, device)

def estimate_loss_wrapper(override_skip_val=None, collect_moe_stats=False):
    # Allow overriding skip_val_loss for final validation sweep
    skip_val = override_skip_val if override_skip_val is not None else skip_val_loss
    # Get raw model for MOE stats collection
    raw_model = model.module if ddp else model
    if deterministic_loader is not None:
        deterministic_loader.reset_eval_cursors()
    return estimate_loss(raw_model, eval_iters, skip_val, get_batch_wrapper, ctx, collect_moe_stats, raw_model)

def get_lr_wrapper(it):
    return get_lr(it, learning_rate, warmup_iters, lr_decay_iters, max_iters, min_lr, decay_lr = decay_lr)

# logging
wandb_run = None
csv_logger = None
if master_process:
    if wandb_log:
        import wandb
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    if csv_log:
        from csv_logging import CSVLogWrapper
        def log(log_dict):
            pass
        csv_logger = CSVLogWrapper(log, config=config, out_dir=out_dir, flush_every=flush_every)

# Initialize trainer
ddp_settings = {
    'ddp': ddp,
    'ddp_local_rank': ddp_local_rank if ddp else None,
    'ddp_world_size': ddp_world_size,
    'ddp_rank': ddp_rank,
} if ddp else None

# Store additional config values needed by trainer
config['iter_num'] = iter_num
config['best_val_loss'] = best_val_loss
config['model_args'] = model_args
config['wandb_run'] = wandb_run
config['csv_logger'] = csv_logger
config['batch_size'] = batch_size  # Add batch_size to config for MFU calculation

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    config=config,
    device=device,
    master_process=master_process,
    ddp_settings=ddp_settings
)

# Run training
trainer.run_training_loop(
    get_batch_fn=get_batch_wrapper,
    estimate_loss_fn=estimate_loss_wrapper,
    get_lr_fn=get_lr_wrapper
)

# Cleanup DDP if used
if ddp:
    destroy_process_group()
