"""Minimal DINOv2 ViT-S/14 architecture, vendored.

This file declares a Vision Transformer whose ``state_dict`` keys match the
official DINOv2 ViT-S/14 release exactly, so weights downloaded from

    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

load with ``strict=False`` and a 0-key mismatch (only ``mask_token`` is dropped
because it is training-only). No dependency on the ``facebookresearch/dinov2``
repository or ``timm``: the only imports are ``torch`` and ``torch.nn``.

Architecture (DINOv2 ViT-S/14, ImageNet self-supervised):
    patch_size  = 14
    embed_dim   = 384
    depth       = 12
    num_heads   = 6
    mlp_ratio   = 4.0
    qkv_bias    = True
    LayerScale  = init 1e-5 per block (gamma stored as ``ls{1,2}.gamma``)

Image input contract:
    * shape  (B, 3, 224, 224)   -- 224 = 16 * 14, so 16x16 = 256 patches
    * range  ImageNet-normalized floats

Forward returns the CLS token after the final ``norm`` (a (B, 384) tensor).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """Conv2d(3 -> embed_dim, kernel=patch, stride=patch) + flatten."""

    def __init__(self, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # (B, C, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        return x


class LayerScale(nn.Module):
    """y = x * gamma   (per-channel learned scale, init small)."""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Attention(nn.Module):
    """Standard multi-head self-attention with fused qkv.

    Key shapes follow DINOv2 conventions:
        qkv.weight : (3 * embed_dim, embed_dim)
        qkv.bias   : (3 * embed_dim,)
        proj.weight: (embed_dim, embed_dim)
        proj.bias  : (embed_dim,)
    """

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True) -> None:
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                # each (B, H, N, Hd)
        # Use the math-friendly path; SDPA is also fine but adds version-coupling.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm Transformer block with LayerScale on each residual branch."""

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_value=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, hidden_dim=int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_value=init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVitS14(nn.Module):
    """DINOv2 ViT-S/14 with state_dict key parity.

    Forward returns the CLS token after the final LayerNorm, shape ``(B, 384)``.
    Inputs must already be ImageNet-normalized at 224x224.
    """

    EMBED_DIM = 384
    PATCH_SIZE = 14
    DEPTH = 12
    NUM_HEADS = 6
    NUM_PATCHES = (224 // 14) ** 2   # = 256
    NUM_TOKENS = NUM_PATCHES + 1     # + cls

    def __init__(self, init_values: float = 1e-5) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(self.PATCH_SIZE, 3, self.EMBED_DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.NUM_TOKENS, self.EMBED_DIM))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.EMBED_DIM,
                    num_heads=self.NUM_HEADS,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    init_values=init_values,
                )
                for _ in range(self.DEPTH)
            ]
        )
        self.norm = nn.LayerNorm(self.EMBED_DIM, eps=1e-6)

    @torch.no_grad()
    def _interpolate_pos_encoding(self, num_tokens: int) -> torch.Tensor:
        """Hook for non-default input sizes. The 224x224 path uses pos_embed
        verbatim; only larger eval images would trigger interpolation, which
        we do not exercise but keep here so the API is honest."""
        if num_tokens == self.NUM_TOKENS:
            return self.pos_embed
        # Bicubic-resample patch portion; concat cls separately.
        cls = self.pos_embed[:, :1]
        patch = self.pos_embed[:, 1:]
        old = patch.shape[1]
        side_old = int(math.isqrt(old))
        side_new = int(math.isqrt(num_tokens - 1))
        patch = patch.reshape(1, side_old, side_old, -1).permute(0, 3, 1, 2)
        patch = nn.functional.interpolate(
            patch, size=(side_new, side_new), mode="bicubic", align_corners=False
        )
        patch = patch.permute(0, 2, 3, 1).reshape(1, side_new * side_new, -1)
        return torch.cat([cls, patch], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                                    # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                             # (B, 1+N, C)
        x = x + self._interpolate_pos_encoding(x.size(1)).to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                             # CLS token


_OFFICIAL_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
)


def _resize_pos_embed(pos_embed: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """DINOv2 ships pos_embed at 518x518 (1370 tokens). Interpolate the
    patch portion bicubic to ``target_tokens-1`` patches, leaving cls intact.
    """
    if pos_embed.shape[1] == target_tokens:
        return pos_embed
    cls = pos_embed[:, :1]                        # (1, 1, C)
    patch = pos_embed[:, 1:]                      # (1, N_old, C)
    n_old = patch.shape[1]
    side_old = int(math.isqrt(n_old))
    side_new = int(math.isqrt(target_tokens - 1))
    if side_old * side_old != n_old or side_new * side_new != target_tokens - 1:
        raise ValueError(
            f"pos_embed sizes must be perfect squares; got {n_old} -> {target_tokens-1}"
        )
    patch = patch.reshape(1, side_old, side_old, -1).permute(0, 3, 1, 2)
    patch = nn.functional.interpolate(
        patch, size=(side_new, side_new), mode="bicubic", align_corners=False
    )
    patch = patch.permute(0, 2, 3, 1).reshape(1, side_new * side_new, -1)
    return torch.cat([cls, patch], dim=1)


def load_official_dinov2_vits14(weights_path: str | None = None) -> DinoVitS14:
    """Build the architecture and load Meta's official DINOv2 ViT-S/14 weights.

    If ``weights_path`` is ``None``, downloads to the standard ``torch.hub``
    cache the first time, then re-uses it. ``mask_token`` (training-only) is
    dropped; ``pos_embed`` is bicubic-resized from 518x518 (1370 tokens) to
    our 224x224 grid (257 tokens).
    """
    model = DinoVitS14()
    if weights_path is None:
        sd = torch.hub.load_state_dict_from_url(
            _OFFICIAL_URL, map_location="cpu", check_hash=False, progress=False
        )
    else:
        sd = torch.load(weights_path, map_location="cpu")
    sd = {k: v for k, v in sd.items() if k != "mask_token"}
    if "pos_embed" in sd:
        sd["pos_embed"] = _resize_pos_embed(sd["pos_embed"], DinoVitS14.NUM_TOKENS)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected DINOv2 keys: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"Missing DINOv2 keys: {missing[:5]}")
    return model
