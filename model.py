import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 1024
    block_size = 256
    n_layer = 4
    n_head = 6
    n_embd = 192
    dropout = 0.0
    tie_weights = True

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def apply_rope(x, cos, sin):
    # x: (B, T, H, D)
    # cos, sin: (T, D // 2)
    D = x.shape[-1]
    x1 = x[..., :D//2]
    x2 = x[..., D//2:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.cat([rx1, rx2], dim=-1)

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        # Omit bias to save parameters
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)
        
        # Apply Rotary Positional Embeddings (RoPE)
        q = apply_rope(q, cos, sin).transpose(1, 2)
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class SwiGLUMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h_ff = int(2.667 * cfg.n_embd)
        self.w_g = nn.Linear(cfg.n_embd, self.h_ff, bias=False)
        self.w_u = nn.Linear(cfg.n_embd, self.h_ff, bias=False)
        self.w_d = nn.Linear(self.h_ff, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w_d(F.silu(self.w_g(x)) * self.w_u(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        
        # Initialize inv_freq buffer for RoPE
        head_dim = cfg.n_embd // cfg.n_head
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
            
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            # Deeper networks benefit from slightly smaller std dev init
            std = 0.02 / math.sqrt(2 * self.cfg.n_layer) if "proj" in str(m) or "w_d" in str(m) else 0.02
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Compute dynamic RoPE frequencies for sequence length T
        t = torch.arange(T, device=idx.device).float()
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
