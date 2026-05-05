"""Img2GPS submission model (Project A) -- frozen DINOv2 + soft retrieval head.

Architecture
============

    encoder      = DINOv2 ViT-S/14 (vendored, see ``dinov2_vit.py``), frozen
    image -> emb : forward -> CLS token (B, 384) -> L2-normalize
    gallery      : (N_train, 384) L2-normalized training embeddings (buffer)
    gallery_gps  : (N_train, 2)   raw [lat, lon] degrees (buffer)
    temperature  : scalar         softmax sharpness (parameter, learned on val)

    sim          = emb @ gallery.T            # (B, N_train)
    weights      = softmax(sim * temperature) # (B, N_train)
    pred         = weights @ gallery_gps      # (B, 2)  raw degrees

Why retrieval (vs the previous ResNet-18 + 2-D regression)
----------------------------------------------------------
- 1k photos over a ~64,000 m^2 rectangle gives ~64 m^2 / photo.
  At test time, every test image lives near a training image; the
  hard floor on a 1-NN predictor here is sqrt(64)/2 ~= 4 m. A direct
  regressor cannot beat this on small data because it must share
  parameters across all photos and ends up averaging.
- DINOv2 embeddings are state-of-the-art for low-data CV (no
  fine-tuning needed for this scale of data), and they sit in a
  cosine-similarity-friendly space.
- The output is a *convex combination* of training GPS coords with
  weights coming from softmax, so the model can never predict
  outside the convex hull of training locations -- which is the
  correct inductive bias when test photos come from the same
  rectangle as training.

Spec compliance (Project_submission.pdf section 3.1)
----------------------------------------------------
- ``Model`` and ``IMG2GPS`` classes instantiable with no arguments.   yes
- ``get_model()`` factory present.                                    yes
- ``forward(batch)`` returns ``[lat, lon]`` in raw degrees.           yes
- ``predict(batch)`` accepts a list/tensor of inputs.                 yes
- Target normalization stats: not used (retrieval has no target
  standardization), so the "stats must be hard-coded in model.py"
  clause is vacuously satisfied.

Checkpoint contract
-------------------
``model.pt`` is a flat ``Model.state_dict()`` saved as::

    {
        "state_dict": {
            "encoder.cls_token"             : ...,
            "encoder.pos_embed"             : ...,
            "encoder.patch_embed.proj.weight": ...,
            ...                                    # all DINOv2 ViT-S/14 keys
            "gallery_emb"                   : (N_train, 384),
            "gallery_gps"                   : (N_train, 2),
            "temperature"                   : 0-d tensor,
        },
        "version": "dino_retrieval_v1",
    }

This is the format the staff evaluator (``eval_project_a.py``) expects:
it filters by exact key + shape match, so we pre-size the gallery
buffers in ``__init__`` by peeking at the file before the evaluator's
``_load_checkpoint`` runs.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .dinov2_vit import DinoVitS14
except ImportError:
    from dinov2_vit import DinoVitS14


_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")


# Single-spot fallback used when no checkpoint is present so that the
# model is still constructible and produces in-region predictions.
# Computed once over the original 89-row metadata.csv (population mean).
_FALLBACK_GALLERY_GPS = (39.951564082397, -75.19132408239702)


class Model(nn.Module):
    """Frozen DINOv2 ViT-S/14 + soft retrieval over a learned gallery.

    Constructed with no arguments. If ``model.pt`` is present at the
    canonical path, both the DINOv2 weights and the gallery load from it
    (ie. no internet needed at inference). If ``model.pt`` is absent,
    the encoder stays at its (random) init -- predictions are garbage,
    but the object is still constructible (this matters for the staff
    evaluator's "instantiate then load checkpoint" flow).
    """

    def __init__(self, weights_path: Optional[str] = _DEFAULT_WEIGHTS) -> None:
        super().__init__()

        self.encoder = DinoVitS14()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Placeholder buffers (size 1) -- replaced below if a real
        # checkpoint is reachable.
        gallery_emb = F.normalize(torch.zeros(1, DinoVitS14.EMBED_DIM), dim=-1)
        gallery_emb[0, 0] = 1.0
        gallery_gps = torch.tensor([_FALLBACK_GALLERY_GPS], dtype=torch.float32)
        self.register_buffer("gallery_emb", gallery_emb)
        self.register_buffer("gallery_gps", gallery_gps)
        self.temperature = nn.Parameter(torch.tensor(20.0))

        # The staff evaluator instantiates with ``weights_path="__no_weights__.pth"``
        # to suppress eager loading, then calls its own ``_load_checkpoint``
        # which filters by exact shape. So we MUST resize gallery buffers
        # here before that filter runs. We probe a list of likely paths
        # (the explicit one first, then canonical fallbacks). If we find a
        # real file, we fully load from it -- which is also idempotent
        # with the evaluator's later flat-state-dict load.
        real = self._find_real_weights(weights_path)
        if real is not None:
            self._load_weights(real)

    # ------------------------------------------------------------------
    # Checkpoint discovery + loading
    # ------------------------------------------------------------------

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

        # Resize gallery buffers from the on-disk shapes BEFORE loading,
        # so PyTorch's strict=False load (and the staff evaluator's
        # equivalent filter) treats them as shape-matched.
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

        # Strict=False so unrelated keys (e.g. a stray ``version`` string,
        # or weights from a slightly different architecture) are ignored.
        # Tensor keys whose names + shapes match get loaded.
        self.load_state_dict(
            {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)},
            strict=False,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, batch) -> torch.Tensor:
        x = self._coerce_batch(batch).float()
        emb = self.encoder(x)                                   # (B, 384)
        emb = F.normalize(emb, dim=-1)
        sim = emb @ self.gallery_emb.t()                        # (B, N_train)
        # Clamp temperature so a bad val tune cannot blow up softmax.
        temp = self.temperature.clamp(min=1e-2, max=200.0)
        weights = torch.softmax(sim * temp, dim=-1)             # (B, N_train)
        pred = weights @ self.gallery_gps                       # (B, 2)
        return pred

    @torch.no_grad()
    def predict(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self._coerce_batch(batch)
        x = x.to(device)
        return self.forward(x).cpu()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Public hook for ``train.py``: returns L2-normalized CLS embeddings."""
        x = self._coerce_batch(x).float()
        emb = self.encoder(x)
        return F.normalize(emb, dim=-1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
