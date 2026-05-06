"""Img2GPS submission model (Project A) -- frozen DINOv2 + soft retrieval head.

The DINOv2 ViT-S/14 implementation is **inlined below** so this file loads as a
single module on the Hugging Face backend (dynamic ``import model`` with no
sibling ``dinov2_vit.py`` on ``sys.path``).

Architecture
============

    encoder      = DINOv2 ViT-S/14 (vendored in this file), frozen
    image -> emb : forward -> CLS token (B, 384) -> L2-normalize
    gallery      : (N_train, 384) L2-normalized training embeddings (buffer)
    gallery_gps  : (N_train, 2)   raw [lat, lon] degrees (buffer)
    temperature  : scalar         softmax sharpness (parameter, learned on val)
    top_k        : scalar int     restrict softmax to the K most similar
                                  gallery entries (buffer; default 10)

    sim                = emb @ gallery.T                       # (B, N_train)
    topk_sim, topk_idx = sim.topk(K, dim=-1)                   # (B, K)
    weights            = softmax(topk_sim * temperature)       # (B, K)
    pred               = sum(weights * gallery_gps[topk_idx])  # (B, 2)

Spec compliance (Project_submission.pdf section 3.1)
----------------------------------------------------
- ``Model`` / ``IMG2GPS`` with no-arg constructor; ``get_model()``; ``predict``;
  ``forward`` returns raw degrees. Target normalization N/A for retrieval.

Checkpoint: flat ``state_dict`` with ``encoder.*``, ``gallery_emb``, ``gallery_gps``,
``temperature`` (see staff ``eval_project_a.py``).
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# DINOv2 ViT-S/14 (inlined — must not import a sibling module)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _Attention(nn.Module):
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
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
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
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = _LayerScale(dim, init_value=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, hidden_dim=int(dim * mlp_ratio))
        self.ls2 = _LayerScale(dim, init_value=init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVitS14(nn.Module):
    """DINOv2 ViT-S/14; ``state_dict`` keys match the official release."""

    EMBED_DIM = 384
    PATCH_SIZE = 14
    DEPTH = 12
    NUM_HEADS = 6
    NUM_PATCHES = (224 // 14) ** 2
    NUM_TOKENS = NUM_PATCHES + 1

    def __init__(self, init_values: float = 1e-5) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(self.PATCH_SIZE, 3, self.EMBED_DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.NUM_TOKENS, self.EMBED_DIM))
        self.blocks = nn.ModuleList(
            [
                _Block(
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
        if num_tokens == self.NUM_TOKENS:
            return self.pos_embed
        cls = self.pos_embed[:, :1]
        patch = self.pos_embed[:, 1:]
        old = patch.shape[1]
        side_old = int(math.isqrt(old))
        side_new = int(math.isqrt(num_tokens - 1))
        patch = patch.reshape(1, side_old, side_old, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=(side_new, side_new), mode="bicubic", align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, side_new * side_new, -1)
        return torch.cat([cls, patch], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._interpolate_pos_encoding(x.size(1)).to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


_OFFICIAL_DINO_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
)


def _resize_pos_embed(pos_embed: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if pos_embed.shape[1] == target_tokens:
        return pos_embed
    cls = pos_embed[:, :1]
    patch = pos_embed[:, 1:]
    n_old = patch.shape[1]
    side_old = int(math.isqrt(n_old))
    side_new = int(math.isqrt(target_tokens - 1))
    if side_old * side_old != n_old or side_new * side_new != target_tokens - 1:
        raise ValueError(
            f"pos_embed sizes must be perfect squares; got {n_old} -> {target_tokens-1}"
        )
    patch = patch.reshape(1, side_old, side_old, -1).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, size=(side_new, side_new), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, side_new * side_new, -1)
    return torch.cat([cls, patch], dim=1)


def load_official_dinov2_vits14(weights_path: str | None = None) -> DinoVitS14:
    """Load Meta DINOv2 ViT-S/14; ``pos_embed`` resized 518→224 grid."""
    model = DinoVitS14()
    if weights_path is None:
        sd = torch.hub.load_state_dict_from_url(
            _OFFICIAL_DINO_URL, map_location="cpu", check_hash=False, progress=False
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


# ---------------------------------------------------------------------------
# Submission model
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")

_FALLBACK_GALLERY_GPS = (39.951564082397, -75.19132408239702)
_DEFAULT_TOP_K = 10


class Model(nn.Module):
    """Frozen DINOv2 ViT-S/14 + soft retrieval over a learned gallery."""

    def __init__(self, weights_path: Optional[str] = _DEFAULT_WEIGHTS) -> None:
        super().__init__()

        self.encoder = DinoVitS14()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        gallery_emb = F.normalize(torch.zeros(1, DinoVitS14.EMBED_DIM), dim=-1)
        gallery_emb[0, 0] = 1.0
        gallery_gps = torch.tensor([_FALLBACK_GALLERY_GPS], dtype=torch.float32)
        self.register_buffer("gallery_emb", gallery_emb)
        self.register_buffer("gallery_gps", gallery_gps)
        self.register_buffer("top_k", torch.tensor(_DEFAULT_TOP_K, dtype=torch.int64))
        self.temperature = nn.Parameter(torch.tensor(20.0))

        real = self._find_real_weights(weights_path)
        if real is not None:
            self._load_weights(real)

    @staticmethod
    def _find_real_weights(weights_path: Optional[str]) -> Optional[str]:
        candidates = []
        if weights_path:
            candidates.append(weights_path)
        candidates.extend(
            [
                _DEFAULT_WEIGHTS,
                os.path.join(os.getcwd(), "model.pt"),
                os.path.join(os.getcwd(), "Img2GPS", "model.pt"),
            ]
        )
        seen = set()
        for p in candidates:
            if not p or p in seen:
                continue
            seen.add(p)
            if os.path.isfile(p):
                return p
        return None

    def _load_weights(self, weights_path: str) -> None:
        ckpt = torch.load(weights_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            return

        emb = sd.get("gallery_emb", None)
        gps = sd.get("gallery_gps", None)
        if isinstance(emb, torch.Tensor) and isinstance(gps, torch.Tensor):
            if emb.shape[0] != gps.shape[0]:
                raise ValueError(
                    f"gallery size mismatch in checkpoint: emb {tuple(emb.shape)} "
                    f"vs gps {tuple(gps.shape)}"
                )
            self.gallery_emb = emb.to(torch.float32)
            self.gallery_gps = gps.to(torch.float32)

        self.load_state_dict(
            {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)},
            strict=False,
        )

    def forward(self, batch) -> torch.Tensor:
        x = self._coerce_batch(batch).float()
        emb = self.encoder(x)
        emb = F.normalize(emb, dim=-1)
        sim = emb @ self.gallery_emb.t()                                 # (B, N)
        temp = self.temperature.clamp(min=1e-2, max=200.0)
        k = max(1, min(int(self.top_k.item()), sim.shape[-1]))
        topk_sim, topk_idx = sim.topk(k, dim=-1)                         # (B, K)
        weights = torch.softmax(topk_sim * temp, dim=-1)                 # (B, K)
        topk_gps = self.gallery_gps[topk_idx]                            # (B, K, 2)
        pred = (weights.unsqueeze(-1) * topk_gps).sum(dim=1)             # (B, 2)
        return pred

    @torch.no_grad()
    def predict(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self._coerce_batch(batch)
        x = x.to(device)
        return self.forward(x).cpu()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._coerce_batch(x).float()
        emb = self.encoder(x)
        return F.normalize(emb, dim=-1)

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    @classmethod
    def _coerce_batch(cls, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            return torch.stack([cls._to_tensor(item) for item in batch], dim=0)
        x = cls._to_tensor(batch)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x


class IMG2GPS(Model):
    """Alias kept for evaluator compatibility (spec section 3.1)."""


def get_model() -> Model:
    return Model()
