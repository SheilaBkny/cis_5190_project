import os
from typing import Iterable, Optional

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


class Model(nn.Module):
    """
    Baseline Img2GPS regression model.

    The model receives image tensors shaped (B, 3, H, W) and returns raw
    latitude/longitude predictions shaped (B, 2), in degrees.
    """

    def __init__(self, weights_path: Optional[str] = "model.pt") -> None:
        super().__init__()
        self.backbone = _resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.load_state_dict(state_dict, strict=False)

    def forward(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = torch.stack([self._to_tensor(x) for x in batch], dim=0)
        else:
            batch = self._to_tensor(batch)
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
        return self.backbone(batch.float())

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


class IMG2GPS(Model):
    pass


def get_model() -> Model:
    return Model()
