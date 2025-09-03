"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
) -> Union[torch.Tensor, int]:
    """
    Computes auxiliary load balancing loss as in Switch Transformer.
    
    Args:
        gate_logits: Tuple of tensors of shape [batch_size * sequence_length, num_experts]
        num_experts: Number of experts
        top_k: Number of experts to route per token
        
    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
    
    if len(gate_logits) == 0:
        return 0
    
    # Concatenate gate logits from all layers
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    
    # Compute routing weights (softmax probabilities)
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    
    # Get top-k expert selections
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # Create expert mask (one-hot for selected experts)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    
    # Compute the percentage of tokens routed to each expert
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    
    # Compute the average probability of routing to each expert
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    
    # Compute auxiliary loss: sum(tokens_per_expert * router_prob_per_expert) * num_experts
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.mup_enabled = config.mup_enabled
        self.mup_disable_attention_scaling = config.mup_disable_attention_scaling
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.mup_enabled and not self.mup_disable_attention_scaling:
            ### Begin muP code ###
            attention_scale = 1.0 / k.size(-1)
            ### End muP code ###
        else:
            attention_scale = 1.0 / math.sqrt(k.size(-1))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True, scale=attention_scale)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * attention_scale
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.alpha * config.n_embd)
        self.c_fc    = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = int(config.alpha * config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        if self.config.mup_enabled:
            # muP: scale activations in forward pass
            h = self.gelu(self.c_fc(x) / math.sqrt(self.config.n_embd))
            hidden_size = int(self.config.alpha * self.config.n_embd)
            out = self.c_proj(h) / hidden_size
        else:
            h = self.gelu(self.c_fc(x))
            out = self.c_proj(h)
        out = self.dropout(out)
        return out

class MLP_MOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_exp = config.num_exp
        self.num_act = config.num_act  # top_k
        self.tau = config.moe_tau if hasattr(config, 'moe_tau') else 1.0
        
        # Router
        self.router = nn.Linear(config.n_embd, self.n_exp, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.n_exp))
        
        # Experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.n_exp)])
        
        # For tracking tokens per expert (needed for learning rate calculation)
        self.register_buffer('tokens_per_expert', torch.zeros(self.n_exp))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
        # Momentum buffer for bias gradients (EMA of gradients)
        self.register_buffer('bias_momentum_buffer', torch.zeros(self.n_exp))
        self.moe_bias_momentum = config.moe_bias_momentum if hasattr(config, 'moe_bias_momentum') else 0.9
        self.moe_bias_momentum_enabled = config.moe_bias_momentum_enabled if hasattr(config, 'moe_bias_momentum_enabled') else False
        
    def h_func(self, x):
        return F.softmax(x, dim=-1)
    
    def s_func(self, x):
        return torch.sigmoid(x)
    
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        
        # Router forward pass
        logit = self.router(x_flat) / math.sqrt(C)
       
        score = self.s_func(logit)  # (B*T, n_exp)
        mu_add_bias = self.h_func(logit / self.tau) + self.bias  # (B*T, n_exp)
        
        # Top-k selection
        _, topk_indices = mu_add_bias.topk(self.num_act, dim=-1)  # (B*T, num_act)
        mask = torch.zeros_like(mu_add_bias)  # (B*T, n_exp)
        mask.scatter_(1, topk_indices, 1)
        
        gate = (mask.detach() * score).unsqueeze(-1)  # (B*T, n_exp, 1)
        
        
        # Which experts are used by at least one token?
        active_experts = torch.nonzero(mask.any(dim=0), as_tuple=True)[0]  # (E_active,)
        
        output = None
        C_out = None
        
        for i in active_experts.tolist():
            tok_idx = torch.nonzero(mask[:, i], as_tuple=True)[0]  # (n_i,)
            if tok_idx.numel() == 0:
                continue
            
            y_i = self.experts[i](x_flat.index_select(0, tok_idx))  # (n_i, C_out_i)
            
            if output is None:
                C_out = y_i.size(-1)
                output = x_flat.new_zeros(B*T, C_out) 

            # gates for those routes (using gate, not score)
            g_i = gate.index_select(0, tok_idx)[:, i, :]  # (n_i, 1)
            
            output.index_add_(0, tok_idx, g_i * y_i)
        
        # handle case where no experts are selected (shouldn't happen but be safe)
        if output is None:
            output = torch.zeros_like(x_flat)
            
        # reshape back
        output = output.view(B, T, -1)  # (B, T, C_out)
        


        # Always track tokens per expert for monitoring/display purposes
        if self.training:
            self.tokens_per_expert += mask.sum(dim=0).detach() # has shape (n_exp,)
            self.total_tokens += mask.shape[0]
        
        # Return gate logits for auxiliary loss if using aux_loss method
        if self.config.moe_load_balance_method == "aux_loss":
            # Return logits with bias but before softmax for aux loss
            logits_with_bias = logit / self.tau + self.bias
            return output, mask.detach(), logits_with_bias
        else:
            return output, mask.detach()
    
    def update_router_bias(self, avg_usage, target_usage, lr_bias, disable_momentum = False):
        """Update router bias to encourage balanced expert usage with optional momentum"""
        gradient = avg_usage - target_usage  # (n_exp,)
        
        if self.moe_bias_momentum_enabled and not disable_momentum:
            # Update momentum buffer (EMA of gradients)
            self.bias_momentum_buffer = (self.moe_bias_momentum * self.bias_momentum_buffer + 
                                        (1 - self.moe_bias_momentum) * gradient)
            # Apply smoothed gradient
            self.bias.data -= lr_bias * self.bias_momentum_buffer
        else:
            # Direct gradient update (original behavior)
            self.bias.data -= lr_bias * gradient

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if hasattr(config, 'num_exp') and config.num_exp > 1:
            self.mlp = MLP_MOE(config)
            self.use_moe = True
        else:
            self.mlp = MLP(config)
            self.use_moe = False

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.use_moe:
            mlp_result = self.mlp(self.ln_2(x))
            if len(mlp_result) == 3:  # aux_loss method returns gate logits
                mlp_out, mask, gate_logits = mlp_result
                x = x + mlp_out
                return x, mask, gate_logits
            else:  # bias method
                mlp_out, mask = mlp_result
                x = x + mlp_out
                return x, mask
        else:
            x = x + self.mlp(self.ln_2(x))
            return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    init_std: float = 0.02
    mup_enabled: bool = False # Whether to use muP. If False then all other mup variables are ignored
    mup_disable_attention_scaling: bool = False # Disables mup attention scaling
    mup_disable_hidden_lr_scaling: bool = False # Disables mup hidden LR scaling
    mup_width_multiplier: float = 1 # `mup_width_multiplier = width / base_width` where base_width is typically 256
    mup_input_alpha: float = 1 # Optional tunable multiplier applied to input embedding forward pass output
    mup_output_alpha: float = 1 # Optional tunable multiplier applied to output unembedding forward pass output
    # MOE parameters
    num_exp: int = 1 # Number of experts (set to 1 to disable MOE)
    num_act: int = 1 # Number of active experts (top-k)
    moe_tau: float = 1.0 # Temperature for router softmax
    moe_bias_lr: float = 1e-2 # Learning rate for router bias updates (only used with bias method)
    moe_bias_momentum: float = 0.9 # EMA decay factor for bias gradient momentum (only used with bias method)
    moe_bias_momentum_enabled: bool = True # Enable momentum for router bias updates (only used with bias method)
    moe_load_balance_method: str = "bias" # "bias" or "aux_loss" - method for load balancing
    moe_aux_loss_weight: float = 0.01 # Auxiliary loss coefficient (only used with aux_loss method)
    alpha: float = 4.0 # Hidden layer size multiplier (hidden_size = alpha * n_embd)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if config.mup_enabled:
                ### Begin muP code ###
                # Adjust hidden weight initialization variance by 1 / mup_width_multiplier
                if pn.endswith('c_attn.weight') or pn.endswith('c_fc.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))
                elif pn.endswith('c_proj.weight'):
                    # Handle both regular MLP and MOE experts
                    if 'experts' in pn:
                        # MOE expert output projection
                        torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer * config.mup_width_multiplier))
                    else:
                        torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer * config.mup_width_multiplier))
                elif pn.endswith('router.weight'):
                    # Router initialization
                    torch.nn.init.normal_(p, mean=0.0, std=config.init_std)
                ### End muP code ###
            elif pn.endswith('c_proj.weight'):
                # Handle both regular MLP and MOE experts for non-muP
                torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer))
            elif pn.endswith('router.weight'):
                # Router initialization for non-muP
                torch.nn.init.normal_(p, mean=0.0, std=config.init_std)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        if self.config.mup_enabled:
            ### Begin muP code ###
            x *= self.config.mup_input_alpha
            ### End muP code ###
        expert_masks = []
        gate_logits_list = []
        for block in self.transformer.h:
            if block.use_moe:
                block_result = block(x)
                if len(block_result) == 3:  # aux_loss method
                    x, mask, gate_logits = block_result
                    expert_masks.append(mask)
                    gate_logits_list.append(gate_logits)
                    # Monitor gate logits for extreme values
                    if gate_logits is not None:
                        gate_max = gate_logits.max().item()
                        gate_min = gate_logits.min().item()
                        if abs(gate_max) > 20 or abs(gate_min) > 20:
                            print(f"WARNING: Gate logits extreme: [{gate_min:.2f}, {gate_max:.2f}]")
                else:  # bias method
                    x, mask = block_result
                    expert_masks.append(mask)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.mup_enabled:
                ### Begin muP code ###
                # Scaling `x` instead of `logits` allows coord check to log change
                x *= self.config.mup_output_alpha / self.config.mup_width_multiplier
                ### End muP code ###
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Add auxiliary loss if using aux_loss method
            if self.config.moe_load_balance_method == "aux_loss" and gate_logits_list:
                aux_loss = load_balancing_loss_func(
                    gate_logits=tuple(gate_logits_list),
                    num_experts=self.config.num_exp,
                    top_k=self.config.num_act
                )
                total_loss = ce_loss + self.config.moe_aux_loss_weight * aux_loss
                # Return tuple of (total_loss, ce_loss, aux_loss) for aux_loss method
                loss = (total_loss, ce_loss, aux_loss)
            else:
                # For bias method or no MOE, just return the cross-entropy loss
                loss = ce_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Collect MOE router parameters separately
        router_params = {}
        router_biases = {}
        
        # Identify MOE blocks and compute tokens per expert
        for i, block in enumerate(self.transformer.h):
            if hasattr(block, 'use_moe') and block.use_moe:
                # Collect router weights and biases
                router_name = f'transformer.h.{i}.mlp.router.weight'
                bias_name = f'transformer.h.{i}.mlp.bias'
                if router_name in param_dict:
                    router_params[router_name] = param_dict[router_name]
                if bias_name in param_dict:
                    router_biases[bias_name] = param_dict[bias_name]
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        if self.config.mup_enabled and not self.config.mup_disable_hidden_lr_scaling:
            ### Begin muP code ###
            mup_decay_params = []
            decay_params = []
            nodecay_params = []
            router_param_list = []
            
            for n, p in param_dict.items():
                if n in router_params:
                    # Router parameters get special treatment
                    router_param_list.append((n, p))
                elif n in router_biases:
                    # Router biases: include in optimizer for aux_loss, exclude for bias method
                    if self.config.moe_load_balance_method == "aux_loss":
                        nodecay_params.append(p)  # Router bias is a bias parameter (no weight decay)
                    else:
                        continue  # Skip router biases for bias method (updated manually)
                elif p.dim() >= 2:
                    
                    if n.endswith('c_attn.weight') or n.endswith('c_fc.weight') or n.endswith('c_proj.weight'):
                        mup_decay_params.append(p)
                    else:
                        decay_params.append(p)
                else:
                    nodecay_params.append(p)
            
            optim_groups = [
                {'params': mup_decay_params, 'weight_decay': weight_decay, 'lr_scale': 1/self.config.mup_width_multiplier},
                {'params': decay_params, 'weight_decay': weight_decay, 'lr_scale': 1},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1}
            ]
            
            # Add router parameter groups with dynamic learning rates
            # These will be computed dynamically in the training loop based on tokens per expert
            for router_name, router_param in router_param_list:
                layer_idx = int(router_name.split('.')[2])  # Extract layer index
                optim_groups.append({
                    'params': [router_param],
                    'weight_decay': weight_decay,
                    'lr_scale': 1,  # Will be dynamically adjusted in training loop
                    'is_router': True,
                    'layer_idx': layer_idx
                })
            
            num_mup_decay_params = sum(p.numel() for p in mup_decay_params)
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            num_router_params = sum(p.numel() for n, p in router_param_list)
            print(f"num mup decayed parameter tensors: {len(mup_decay_params)}, with {num_mup_decay_params:,} parameters")
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f"num router parameter tensors: {len(router_param_list)}, with {num_router_params:,} parameters")
            ### End muP code ###
        else:
            # Non-muP case
            decay_params = []
            nodecay_params = []
            router_param_list = []
            
            for n, p in param_dict.items():
                if n in router_params:
                    router_param_list.append((n, p))
                elif n in router_biases:
                    # Router biases: include in optimizer for aux_loss, exclude for bias method
                    if self.config.moe_load_balance_method == "aux_loss":
                        nodecay_params.append(p)  # Router bias is a bias parameter (no weight decay)
                    else:
                        continue  # Skip router biases for bias method (updated manually)
                elif p.dim() >= 2:
                    decay_params.append(p)
                else:
                    nodecay_params.append(p)
                    
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            
            # Add router parameter groups
            for router_name, router_param in router_param_list:
                layer_idx = int(router_name.split('.')[2])
                optim_groups.append({
                    'params': [router_param],
                    'weight_decay': weight_decay,
                    'lr_scale': 1,
                    'is_router': True,
                    'layer_idx': layer_idx
                })
            
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            num_router_params = sum(p.numel() for n, p in router_param_list)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f"num router parameter tensors: {len(router_param_list)}, with {num_router_params:,} parameters")
            
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
