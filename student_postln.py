"""Ablation: Post-LN instead of Pre-LN. Everything else identical."""
import torch
from torch import nn
from torch.nn import functional as F

from rope import build_rope_cache, apply_rope


class Block(nn.Module):
    def __init__(self, width=320, heads=8, context=256, dropout=0.25):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.context = context
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = build_rope_cache(length, self.head_dim, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.norm1(x + self.drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width))))
        return self.norm2(x + self.drop(self.mlp(x)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width = config["width"]
        dropout = config.get("dropout", 0.25)
        self.token = nn.Embedding(config["vocab"], width)
        self.blocks = nn.ModuleList(
            [Block(width, config["heads"], self.context, dropout) for _ in range(config["depth"])]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)