import argparse
import contextlib
import io
import os
from typing import Tuple
import torch
from model import GPTConfig, GPT

#use cpu to count parameters
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def build_config(args: argparse.Namespace) -> GPTConfig:
    if args.n_act > args.n_exp:
        raise ValueError("n_act cannot exceed n_exp.")
    if args.n_exp < 1:
        raise ValueError("n_exp must be at least 1.")
    if args.n_act < 1:
        raise ValueError("n_act must be at least 1.")

    return GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        init_std=args.init_std,
        mup_enabled=args.mup_enabled,
        mup_disable_attention_scaling=args.mup_disable_attention_scaling,
        mup_disable_hidden_lr_scaling=args.mup_disable_hidden_lr_scaling,
        mup_width_multiplier=args.mup_width_multiplier,
        mup_input_alpha=args.mup_input_alpha,
        mup_output_alpha=args.mup_output_alpha,
        num_exp=args.n_exp,
        num_act=args.n_act,
        moe_tau=args.moe_tau,
        moe_bias_lr=args.moe_bias_lr,
        moe_bias_momentum=args.moe_bias_momentum,
        moe_bias_momentum_enabled=args.moe_bias_momentum_enabled,
        moe_load_balance_method=args.moe_load_balance_method,
        moe_aux_loss_weight=args.moe_aux_loss_weight,
        alpha=args.alpha,
        max_iters=args.max_iters,
        bias_update_interval=args.bias_update_interval,
    )


def count_parameters(model: GPT) -> Tuple[int, int]:
    with torch.no_grad():
        total = sum(p.numel() for p in model.parameters())

        expert_params = 0
        for block in model.transformer.h:
            if getattr(block, "use_moe", False):
                for expert in block.mlp.experts:
                    expert_params += sum(p.numel() for p in expert.parameters())
        return total, expert_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate total and active parameters for MuP-MoE GPT models.")
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-exp", type=int, default=3)
    parser.add_argument("--n-act", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", dest="bias", action="store_true")
    parser.add_argument("--no-bias", dest="bias", action="store_false")
    parser.set_defaults(bias=False)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--mup-enabled", action="store_true")
    parser.add_argument("--mup-disable-attention-scaling", action="store_true")
    parser.add_argument("--mup-disable-hidden-lr-scaling", action="store_true")
    parser.add_argument("--mup-width-multiplier", type=float, default=1.0)
    parser.add_argument("--mup-input-alpha", type=float, default=1.0)
    parser.add_argument("--mup-output-alpha", type=float, default=1.0)
    parser.add_argument("--moe-tau", type=float, default=1.0)
    parser.add_argument("--moe-bias-lr", type=float, default=1e-2)
    parser.add_argument("--moe-bias-momentum", type=float, default=0.9)
    parser.add_argument("--disable-moe-bias-momentum", action="store_true")
    parser.add_argument("--moe-load-balance-method", choices=["bias", "aux_loss"], default="bias")
    parser.add_argument("--moe-aux-loss-weight", type=float, default=0.01)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--bias-update-interval", type=int, default=1)

    args = parser.parse_args()
    args.mup_disable_attention_scaling = args.mup_disable_attention_scaling
    args.mup_disable_hidden_lr_scaling = args.mup_disable_hidden_lr_scaling
    args.moe_bias_momentum_enabled = not args.disable_moe_bias_momentum

    config = build_config(args)

    # Suppress GPT constructor printout so we control the final output
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = GPT(config)

    total_params, expert_params = count_parameters(model)
    non_expert_params = total_params - expert_params
    if config.num_exp > 0:
        active_expert_params = expert_params * (config.num_act / config.num_exp)
    else:
        active_expert_params = 0.0
    active_params = non_expert_params + active_expert_params

    def fmt(value: float) -> str:
        return f"{int(round(value)):,} ({value / 1e6:.2f}M)"

    print("Total initialized parameters:", fmt(float(total_params)))
    print("Approximate active parameters:", fmt(active_params))


if __name__ == "__main__":
    main()
