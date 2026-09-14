"""Decoder-only transformer over a byte vocabulary.

One vocabulary covers both sides, so the constrained decoder only has to mask logits and
never translates between tokenizers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 256
BOS = 257
EOS = 258
VOCAB_SIZE = 259


@dataclass
class Config:
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_ff: int = 2048
    max_len: int = 160
    dropout: float = 0.0
    tied_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by n_head {self.n_head}")
        return self.d_model // self.n_head

    def n_params(self) -> int:
        attn = 4 * self.d_model * self.d_model + 4 * self.d_model
        mlp = 2 * self.d_model * self.d_ff + self.d_ff + self.d_model
        block = attn + mlp + 4 * self.d_model
        emb = VOCAB_SIZE * self.d_model + self.max_len * self.d_model
        head = 0 if self.tied_embeddings else VOCAB_SIZE * self.d_model + VOCAB_SIZE
        return self.n_layer * block + emb + head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class TinyCronLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        if cfg.tied_embeddings:
            self.head.weight = self.tok.weight
        self.apply(self._init)
        # Scaled residual init, or the pre-LN residual stream grows with depth.
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ):
        """`position_ids` and `valid_mask` exist for batched decoding of unequal-length
        prompts. Training never has a real token following a PAD, so a padded batch that
        lets one attend to PAD is off-distribution; masking those keys out and numbering
        each row's real tokens from zero is what makes a batched decode equivalent to the
        one-row-at-a-time decode the browser does."""
        b, t = idx.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.cfg.max_len}")
        positions = position_ids if position_ids is not None else torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(positions))
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=idx.device))
        mask = causal[None, None, :, :]
        if valid_mask is not None:
            mask = mask & valid_mask[:, None, None, :]
            # A query whose every key is masked would softmax over all -inf and produce NaN,
            # so each position is always allowed to attend to itself.
            mask = mask | torch.eye(t, dtype=torch.bool, device=idx.device)[None, None, :, :]
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.norm_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss
