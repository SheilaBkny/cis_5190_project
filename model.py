"""Img2GPS regression model (Project A).

Architecture: ResNet-18 backbone (ImageNet pretrained) with the final linear
layer replaced by a 2-D head that predicts a *standardized* latitude/longitude
pair. The model carries `y_mean` / `y_std` buffers (computed on the training
split during training) so `forward()` always returns predictions in raw
degrees, while training can supervise on the well-conditioned standardized
output.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import torch
from torch import nn
from torchvision import models


def _resnet18(pretrained: bool = True) -> nn.Module:
    """Build ResNet18 while staying compatible with old and new torchvision APIs."""
    if pretrained:
        try:
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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


_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")

# Target normalization stats hard-coded per the submission spec
# ("If you use normalization, hard code the stats in your model.py").
# These are the train-split statistics produced by the iter1 training run
# (Img2GPS/train.py with seed=42 on Img2GPS/metadata.csv).
_TARGET_MEAN = (39.951541900634766, -75.19132232666016)
_TARGET_STD = (0.0002309196861460805, 0.0005374249303713441)


class Model(nn.Module):
    """Img2GPS model used for both training and the official evaluator.

    ``forward(x)`` returns un-normalized latitude/longitude in degrees, shape
    ``(B, 2)``. Internally the backbone outputs a standardized prediction and
    the model un-standardizes it via the registered ``y_mean`` / ``y_std``
    buffers, so a checkpoint round-trips correctly.
    """

    def __init__(self, weights_path: Optional[str] = _DEFAULT_WEIGHTS) -> None:
        super().__init__()
        self.backbone = _resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

        # Hard-coded train-split target stats (see top of file). These are
        # also the values that `model.pt` carries, so loading weights is a
        # no-op for the buffers; if `model.pt` is unavailable the model still
        # returns meaningful degrees.
        self.register_buffer("y_mean", torch.tensor(_TARGET_MEAN, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(_TARGET_STD, dtype=torch.float32))

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def set_target_stats(self, mean: Sequence[float], std: Sequence[float]) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).view(2)
        std_t = torch.as_tensor(std, dtype=torch.float32).view(2)
        std_t = torch.clamp(std_t, min=1e-8)
        self.y_mean.copy_(mean_t)
        self.y_std.copy_(std_t)

    def normalize_targets(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def predict_standardized(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Inference path used by the evaluator
    # ------------------------------------------------------------------

    def forward(self, batch):
        x = self._coerce_batch(batch)
        z = self.backbone(x.float())
        return z * self.y_std + self.y_mean

    # Test-time augmentation: average the model's prediction on the
    # original image and its horizontal flip. H-flip is safe to apply
    # after ImageNet normalization (it's a pure spatial reordering of
    # pixels) and gives a small but reliable improvement on the leaderboard
    # because the regression head's noise is partially uncorrelated across
    # the two views. Doubles inference time (still well under spec budget).
    USE_TTA_HFLIP: bool = True

    def predict(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.stack([self._to_tensor(item) for item in batch], dim=0).to(device).float()
            z = self.backbone(x)
            if self.USE_TTA_HFLIP:
                z_flip = self.backbone(torch.flip(x, dims=[-1]))
                z = (z + z_flip) * 0.5
            preds = z * self.y_std + self.y_mean
            return preds.cpu()

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
    """Alias kept for evaluator compatibility."""


def get_model() -> Model:
    return Model()
