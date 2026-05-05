"""Img2GPS regression model (Project A) — soft-cluster classification head.

Why this architecture (vs the ResNet-18 baseline that came before):

The naive setup of "ResNet-18 + 2-D regression head + MSE in standardized
space" has a strong attractor toward predicting ``y_mean`` for any image
the model has not seen something similar to during training. With ~89
training photos covering only a sub-region of the test rectangle, the
regressor consistently collapsed to the training centroid, scoring worse
than the constant-mean baseline (48.7 m) on the leaderboard.

This module reframes the task as **soft classification over K location
clusters** (K-means centroids of the training GPS labels):

    logits  = backbone(image)              # (B, K)
    weights = softmax(logits)              # (B, K)
    pred    = weights @ cluster_centers    # (B, 2)  raw lat/lon degrees

* The output is still ``[lat, lon]`` in raw degrees, so the spec contract
  in ``Project_submission.pdf`` §3.1 / §4.1 is preserved.
* Mean-collapse is impossible: the model can only output points inside
  the convex hull of the cluster centers, and CE gradients push it to
  pick a *specific* cluster rather than averaging toward the centroid.
* Soft weighting at inference still gives sub-cluster precision (a
  50/50 mixture between two adjacent clusters predicts their midpoint).
* Backbone is ``mobilenet_v3_small`` (torchvision) for 4-5x faster CPU
  inference vs ResNet-18 — the leaderboard scores inference time, and
  ResNet-18 was burning ~100 ms on the backend's CPU.

Spec compliance (Project_submission.pdf §3.1):
* ``Model`` and ``IMG2GPS`` classes instantiable with no arguments.  ✓
* ``get_model()`` factory present.                                    ✓
* ``forward(batch)`` returns ``[lat, lon]`` in raw degrees.           ✓
* ``predict(batch)`` accepts a list/tensor of inputs.                 ✓
* Target normalization stats (``_TARGET_MEAN``, ``_TARGET_STD``) are
  hard-coded in this file. Cluster centers also have a hard-coded
  default (``_build_default_centers``) — a near-square lat/lon grid
  over the Penn test rectangle — that ``model.pt`` overrides via
  ``load_state_dict`` with K-means centroids of the training set. K
  itself is auto-detected from the checkpoint at construction time so
  zero-arg ``Model()`` works regardless of which K was trained.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional, Sequence

import torch
from torch import nn
from torchvision import models


# ---------------------------------------------------------------------------
# Hard-coded constants (per spec: normalization stats hard-coded in model.py)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")

# Phone-only training-split target mean/std (kept for backward compat with
# any code that still reads them; the cluster head does not use them).
_TARGET_MEAN = (39.951541900634766, -75.19132232666016)
_TARGET_STD = (0.0002309196861460805, 0.0005374249303713441)

# Default number of location clusters when nothing else specifies K (no
# checkpoint, no constructor argument). With ~71 training images and ~48
# unique GPS locations, K=80 puts us in an instance-retrieval regime
# where each cluster ends up with 1-2 training images. The actual K used
# at inference time is whatever ``model.pt`` was trained with — the
# constructor reads it from the checkpoint when available.
_NUM_CLUSTERS = 80

# Bounding box used to seed default cluster centers when there is no
# trained checkpoint to load (33rd & Walnut -> 34th & Spruce).
_LAT_LO, _LAT_HI = 39.9508, 39.9525
_LON_LO, _LON_HI = -75.1928, -75.1900


def _build_default_centers(num_clusters: int) -> torch.Tensor:
    """Generate a near-square lat/lon grid of ``num_clusters`` points
    over the Penn test rectangle. Used only when ``model.pt`` is not
    available; ``train.py`` always overwrites ``cluster_centers`` with
    the K-means centroids of the training split before the first epoch.
    """
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")
    cols = max(1, int(round(math.sqrt(num_clusters))))
    rows = max(1, math.ceil(num_clusters / cols))
    pts = []
    for i in range(num_clusters):
        r = i // cols
        c = i % cols
        lat = _LAT_LO + (_LAT_HI - _LAT_LO) * (r / max(rows - 1, 1))
        lon = _LON_LO + (_LON_HI - _LON_LO) * (c / max(cols - 1, 1))
        pts.append([lat, lon])
    return torch.tensor(pts, dtype=torch.float32)


def _peek_num_clusters(weights_path: str) -> Optional[int]:
    """Read K from a saved checkpoint without instantiating the model.

    Looks for the ``cluster_centers`` buffer first, then falls back to
    the final classifier weight's row count. Returns ``None`` on any
    failure so callers can transparently fall back to the default K.
    """
    try:
        ckpt = torch.load(weights_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(sd, dict):
            return None
        if "cluster_centers" in sd:
            return int(sd["cluster_centers"].shape[0])
        for key, val in sd.items():
            if key.endswith("classifier.3.weight") and hasattr(val, "shape"):
                return int(val.shape[0])
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Backbone factory (compatible with old + new torchvision APIs)
# ---------------------------------------------------------------------------


def _mobilenet_v3_small(pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            pass
        try:
            return models.mobilenet_v3_small(pretrained=True)
        except Exception:
            pass
    try:
        return models.mobilenet_v3_small(weights=None)
    except TypeError:
        return models.mobilenet_v3_small(pretrained=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Soft-cluster classifier for Img2GPS.

    The backbone produces ``K`` logits per image. ``forward()`` softmaxes
    them and returns the weighted average of the ``K`` cluster centers in
    raw lat/lon degrees, matching the Project A spec output contract.
    """

    def __init__(
        self,
        weights_path: Optional[str] = _DEFAULT_WEIGHTS,
        num_clusters: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Resolve K: explicit constructor arg > checkpoint introspection
        # > module-level default. This lets ``Model()`` (the spec-required
        # zero-arg form) auto-pick up whatever K was trained, and lets
        # the training loop pass an explicit K to experiment with.
        if num_clusters is None and weights_path and os.path.exists(weights_path):
            num_clusters = _peek_num_clusters(weights_path)
        if num_clusters is None:
            num_clusters = _NUM_CLUSTERS
        self.num_clusters = int(num_clusters)

        self.backbone = _mobilenet_v3_small(pretrained=True)
        # MobileNetV3-Small classifier: Linear(576,1024) -> Hardswish ->
        # Dropout -> Linear(1024, num_classes). We keep the existing
        # bottleneck and only swap the final logit layer for K clusters.
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, self.num_clusters)

        # Buffers: persisted in ``model.pt`` so the trained centers and
        # legacy normalization stats round-trip correctly.
        self.register_buffer("cluster_centers", _build_default_centers(self.num_clusters))
        self.register_buffer("y_mean", torch.tensor(_TARGET_MEAN, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(_TARGET_STD, dtype=torch.float32))

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            self.load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def set_cluster_centers(self, centers: Sequence[Sequence[float]]) -> None:
        """Overwrite ``cluster_centers`` (called by ``train.py`` after K-means)."""
        centers_t = torch.as_tensor(centers, dtype=torch.float32).view(self.num_clusters, 2)
        self.cluster_centers.copy_(centers_t)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (B, K) logits — used by the training loop's CE loss."""
        return self.backbone(x.float())

    # Kept for backward compat with any caller from the previous regression
    # implementation. Returns the same value forward() does.
    def predict_standardized(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def normalize_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Legacy helper, retained for callers that still use the old API."""
        return (y - self.y_mean) / self.y_std

    # ------------------------------------------------------------------
    # Inference path used by the evaluator
    # ------------------------------------------------------------------

    def forward(self, batch):
        x = self._coerce_batch(batch)
        logits = self.backbone(x.float())
        weights = torch.softmax(logits, dim=-1)
        return weights @ self.cluster_centers

    def predict(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.stack([self._to_tensor(item) for item in batch], dim=0).to(device)
            return self.forward(x).cpu()

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
    """Alias kept for evaluator compatibility (spec §3.1)."""


def get_model() -> Model:
    return Model()
