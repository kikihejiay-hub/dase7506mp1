"""Rotary Position Embedding (RoPE), no learnable parameters."""
import torch


def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    """Precompute cos/sin for RoPE."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)  # [seq_len, half]
    cos = angles.cos()  # [seq_len, half]
    sin = angles.sin()  # [seq_len, half]
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply RoPE to q or k. x: [batch, heads, seq, head_dim]."""
    # Split last dim into two halves
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # cos/sin shape: [seq, half] -> [1, 1, seq, half]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    # Rotate
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)