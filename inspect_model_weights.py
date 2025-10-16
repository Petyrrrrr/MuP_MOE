#!/usr/bin/env python3
"""
Script to inspect model architecture and print all parameter names and shapes.
"""

from model import GPT, GPTConfig

def inspect_model_weights():
    # Create a small model configuration
    config = GPTConfig(
        n_layer=2,      # Just 2 layers for inspection
        n_head=4,
        n_embd=128,
        block_size=256,
        mup_enabled=True,
        num_exp=4,      # 4 experts for MoE
        num_act=2,      # 2 active experts
        moe_load_balance_method='aux_loss'
    )
    
    # Create model
    model = GPT(config)
    
    print("="*80)
    print("MODEL PARAMETER STRUCTURE")
    print("="*80)
    
    # Group parameters by type
    param_groups = {
        'embeddings': [],
        'attention': [],
        'mlp': [],
        'router': [],
        'layernorm': [],
        'other': []
    }
    
    # Categorize all parameters
    for name, param in model.named_parameters():
        shape_str = f"shape={list(param.shape)}"
        numel_str = f"numel={param.numel():,}"
        param_info = f"{name:<70} {shape_str:<20} {numel_str}"
        
        if 'wte' in name or 'wpe' in name:
            param_groups['embeddings'].append(param_info)
        elif 'attn' in name:
            param_groups['attention'].append(param_info)
        elif 'router' in name:
            param_groups['router'].append(param_info)
        elif 'ln' in name or 'layer_norm' in name:
            param_groups['layernorm'].append(param_info)
        elif 'mlp' in name or 'c_fc' in name or 'c_proj' in name or 'experts' in name:
            param_groups['mlp'].append(param_info)
        else:
            param_groups['other'].append(param_info)
    
    # Print grouped parameters
    for group_name, params in param_groups.items():
        if params:
            print(f"\n{group_name.upper()} PARAMETERS:")
            print("-" * 80)
            for param in sorted(params):
                print(param)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Count by parameter type
    print("\nParameter counts by type:")
    for name, param in model.named_parameters():
        if name.endswith('.weight'):
            param_type = name.split('.')[-2] + '.weight'
        elif name.endswith('.bias'):
            param_type = name.split('.')[-2] + '.bias'
        else:
            param_type = 'other'
    
    # More detailed breakdown
    weight_types = {}
    for name, param in model.named_parameters():
        print(name)
        # Extract the weight type
        if '.weight' in name:
            if 'c_attn' in name:
                key = 'c_attn.weight (attention QKV projection)'
            elif 'c_proj' in name and 'attn' in name:
                key = 'attn.c_proj.weight (attention output projection)'
            elif 'c_fc' in name:
                key = 'c_fc.weight (MLP first layer)'
            elif 'c_proj' in name and 'mlp' in name:
                key = 'mlp.c_proj.weight (MLP output projection)'
            elif 'router' in name:
                key = 'router.weight (MoE router)'
            elif 'wte' in name:
                key = 'wte.weight (token embeddings)'
            elif 'wpe' in name:
                key = 'wpe.weight (position embeddings)'
            elif 'experts' in name:
                key = 'experts.weight (MoE expert weights)'
            else:
                key = 'other.weight'
        elif '.bias' in name:
            key = name.split('.')[-2] + '.bias'
        else:
            key = 'other'
        
        if key not in weight_types:
            weight_types[key] = {'count': 0, 'params': 0}
        weight_types[key]['count'] += 1
        weight_types[key]['params'] += param.numel()
    
    print("\nWeight type breakdown:")
    for wtype, info in sorted(weight_types.items()):
        print(f"  {wtype:<50} count={info['count']:3d}, params={info['params']:,}")
    
    # Show hierarchy
    print("\n" + "="*80)
    print("MODEL HIERARCHY (first layer example)")
    print("="*80)
    
    print("transformer.")
    print("├── wte (token embeddings)")
    print("├── wpe (position embeddings)")
    print("├── h (transformer blocks)")
    print("│   └── 0 (first block)")
    print("│       ├── ln_1 (layer norm 1)")
    print("│       ├── attn (attention)")
    print("│       │   ├── c_attn (Q,K,V projection)")
    print("│       │   └── c_proj (output projection)")
    print("│       ├── ln_2 (layer norm 2)")
    print("│       └── mlp (MLP or MoE)")
    if config.num_exp > 1:
        print("│           ├── router (expert routing)")
        print("│           ├── experts (expert networks)")
        print("│           │   └── [0...n_exp-1]")
        print("│           │       ├── c_fc (first layer)")
        print("│           │       └── c_proj (output layer)")
    else:
        print("│           ├── c_fc (first layer)")
        print("│           └── c_proj (output layer)")
    print("└── ln_f (final layer norm)")

if __name__ == "__main__":
    inspect_model_weights()