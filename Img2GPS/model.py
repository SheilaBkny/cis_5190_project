"""Img2GPS submission model (Project A) — ResNet-18 + 2-D regression head.

Architecture (matches `Img2GPS/notebooks/Release_baseline_model.ipynb`):

    backbone   = torchvision.models.resnet18
    backbone.fc = nn.Linear(512, 2)            # standardized [lat, lon]
    pred       = backbone(x) * y_std + y_mean   # raw [lat, lon] degrees

Training (see the notebook): ImageNet ResNet-18, full fine-tuning under
Adam @ lr=1e-3 with weight_decay=1e-4 and grad-norm clipping at 1.0,
StepLR(step_size=5, gamma=0.1), MSE loss in standardized lat/lon space.
The training cell tracks the best reference-set Haversine across epochs
and saves *that* checkpoint, so a late-epoch divergence does not corrupt
the saved weights.

Spec compliance (Project_submission.pdf §3.1):
  * `Model` and `IMG2GPS` classes instantiable with no arguments.       ✓
  * `get_model()` factory present.                                      ✓
  * `forward(batch)` returns `[lat, lon]` in raw degrees.               ✓
  * `predict(batch)` accepts a list/tensor of inputs.                   ✓
  * Target normalization stats (`_TARGET_MEAN`, `_TARGET_STD`) are
    hard-coded in this file (computed from `Img2GPS/metadata.csv`,
    population mean/std over 89 training photos — same numbers the
    notebook's `LocalGPSImageDataset` derives from the train split).

Checkpoint compatibility:
  * `Img2GPS/model.pt` is expected to be a torchvision-style
    `resnet18` `state_dict()` saved with `torch.save(resnet.state_dict(),
    ...)`. The training notebook saves exactly that.
  * The evaluator (`eval_project_a.py`) introspects `getattr(model,
    "model", None)` to find an inner module to load into. We expose
    `self.model = resnet18` so its state_dict keys (`conv1.weight`,
    `bn1.weight`, ..., `fc.weight`, `fc.bias`) load directly with
    `strict=False`.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import torch
from torch import nn
from torchvision import models


_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")

# Population (np.std with ddof=0) lat/lon mean/std over the 89-row
# Img2GPS/metadata.csv training set. The notebook's LocalGPSImageDataset
# computes these via numpy, so the values here MUST match what training
# saw — otherwise the model's denormalized output is shifted.
_TARGET_MEAN = (39.951564082397, -75.19132408239702)
_TARGET_STD = (0.0002511944016460731, 0.0005488786100232739)


def _resnet18(pretrained: bool = False) -> nn.Module:
    """torchvision resnet18 with both old (`pretrained=True`) and
    new (`weights=...`) APIs. Pretrained is *only* used as a fallback
    starting point; the trained `model.pt` overwrites everything.
    """
    if pretrained:
        try:
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            pass
        try:
            return models.resnet18(pretrained=True)
        except Exception:
            pass
    try:
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=False)


class Model(nn.Module):
    """ResNet-18 regressor for Img2GPS submission.

    The backbone outputs 2 raw values in standardized lat/lon space.
    `forward()` denormalizes them with the hard-coded training stats
    so the returned tensor is `[lat, lon]` in raw degrees, per the
    Project A spec output contract.
    """

    def __init__(self, weights_path: Optional[str] = _DEFAULT_WEIGHTS) -> None:
        super().__init__()

        # We do not download ImageNet weights here: the typical path is
        # to load the trained `model.pt` immediately after construction.
        # If `model.pt` is missing the model still constructs cleanly
        # (Kaiming-initialized, garbage predictions) — this matters for
        # the leaderboard's evaluator which instantiates with a sentinel
        # non-existent weights path and then loads the real checkpoint
        # itself.
        self.model = _resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)

        self.register_buffer("y_mean", torch.tensor(_TARGET_MEAN, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(_TARGET_STD, dtype=torch.float32))

        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)

    def _load_weights(self, weights_path: str) -> None:
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            sd = checkpoint
        else:
            return

        # Two on-disk layouts to support:
        #   (a) raw torchvision state_dict from `torch.save(resnet.state_dict(), ...)`
        #       — keys like `conv1.weight`, `fc.weight`. Loads into `self.model`.
        #   (b) state_dict from `Model(...).state_dict()` — keys prefixed with
        #       `model.` plus our `y_mean`/`y_std` buffers. Loads into `self`.
        sample_key = next(iter(sd.keys()), "")
        if sample_key.startswith("model."):
            self.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(sd, strict=False)

    # ------------------------------------------------------------------
    # Inference path used by the evaluator
    # ------------------------------------------------------------------

    def forward(self, batch):
        x = self._coerce_batch(batch)
        out = self.model(x.float())               # (B, 2) standardized
        return out * self.y_std + self.y_mean      # (B, 2) raw lat/lon degrees

    def predict(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.stack([self._to_tensor(item) for item in batch], dim=0).to(device)
            return self.forward(x).cpu()

    # ------------------------------------------------------------------
    # Backward-compat helpers for callers from the older API
    # ------------------------------------------------------------------

    def predict_standardized(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the raw backbone output (in standardized lat/lon space)
        before denormalization. Mirrors the training-time logits used by
        the loss function."""
        return self.model(x.float())

    def normalize_targets(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

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
