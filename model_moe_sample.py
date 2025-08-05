### This is not meant to be run for nano-gpt, but only for reference in setting up the MOE model with router with bias
### Please setup nanoGPT FFN layers similar to this structure

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def h_func(x):
    return F.softmax(x, dim=-1)

def s_func(x): #leaky Relu
    return torch.sigmoid(x)

# -------- model --------
class Expert(nn.Module):
    def __init__(self, d: int, n_hid: int):
        super().__init__()
        self.d, self.n = d, n_hid
        self.w1 = nn.Linear(d, n_hid, bias=False)
        self.w2 = nn.Linear(n_hid, 1, bias=False)
        nn.init.normal_(self.w1.weight, 0, 1.0)
        nn.init.normal_(self.w2.weight, 0, 1.0)
    def forward(self, x):
        h = F.relu(self.w1(x) / math.sqrt(self.d))
        return self.w2(h) / self.n

class MoE(nn.Module):
    def __init__(self, d: int, n_hid: int, n_exp: int, tau: float, scale_factor: float = 1.0):
        super().__init__()
        self.d, self.n_exp, self.tau = d, n_exp, tau
        self.router = nn.Linear(d, n_exp, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_exp)) # (n_exp,)
        nn.init.normal_(self.router.weight, 0, 1.0)
        self.experts = nn.ModuleList([Expert(d, n_hid) for _ in range(n_exp)])
        self.scale_factor = scale_factor

    def forward(self, x):
        logit = self.router(x) / math.sqrt(self.d)  # (B,n)
        score = s_func(logit) 
        mu_add_bias = h_func(logit / self.tau) + self.bias # (B,n)
        idx = mu_add_bias.argmax(dim=-1) # (B,)
        mask = F.one_hot(idx, num_classes=self.n_exp).type_as(score)# (B,n)
        gate = (mask.detach() * score).unsqueeze(-1) # (B,n,1)

        expert_out = torch.stack([e(x) for e in self.experts], dim=1) # (B,n,1)
        f = (gate * expert_out).sum(dim=1).squeeze(-1) # (B,)
        f = f * self.scale_factor

        return f, mask.detach()

class MoE_GRANULARITY(nn.Module):
    def __init__(self, d: int, n_hid: int, n_exp: int, tau: float, top_k: int, scale_factor: float = 1.0):
        super().__init__()
        self.d, self.n_exp, self.tau = d, n_exp, tau
        self.router = nn.Linear(d, n_exp, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_exp)) # (n_exp,)
        nn.init.normal_(self.router.weight, 0, 1.0)
        self.experts = nn.ModuleList([Expert(d, n_hid) for _ in range(n_exp)])
        self.scale_factor = scale_factor
        self.top_k = top_k

    def forward(self, x):
        logit = self.router(x) / math.sqrt(self.d)  # (B,n)
        score = s_func(logit) 
        mu_add_bias = h_func(logit / self.tau) + self.bias # (B,n)
        _, topk_indices = mu_add_bias.topk(self.top_k, dim=-1)  # (B, k), (B, k)
        mask = torch.zeros(mu_add_bias.shape[0], self.n_exp).type_as(score)  # (B, n_exp)
        mask.scatter_(1, topk_indices, 1)  # (B, n_exp) with k ones per row
        gate = (mask.detach() * score).unsqueeze(-1) # (B,n,top_k,1)

        expert_out = torch.stack([e(x) for e in self.experts], dim=1) # (B,n,1)
        f = (gate * expert_out).sum(dim=1).squeeze(-1) # (B,)
        f = f * self.scale_factor
        return f, mask.detach()
    
    def update_router_bias(self, mask, LR_BIAS):
        q_hat = mask.float().mean(dim=0) # (n_exp,)
        self.bias.data -= LR_BIAS * (q_hat - self.top_k/self.n_exp) # (n_exp,)

