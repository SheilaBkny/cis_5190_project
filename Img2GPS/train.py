"""Train the Img2GPS regression model (Project A).

Highlights
----------
* Targets are standardized using the train-split mean/std and persisted as
  buffers on the model, so ``model.pt`` is self-contained.
* Loss is MSE in *standardized* space, but every epoch we additionally report
  the official metric (mean Haversine distance in meters) on a held-out
  validation split.
* The split is location-grouped: photos that share an exact GPS coordinate
  (the spec describes ~8 photos per spot) live entirely in train *or*
  entirely in val to avoid leakage.
* Adam, lr=1e-3, StepLR(step_size=4, gamma=0.5), 12 epochs by default
  (matches the PDF's suggested 10-15 range).
* Light augmentation suited to walkway photography: hflip, color jitter,
  RandomResizedCrop with scale=(0.85, 1.0).
* The best checkpoint by *val Haversine* is saved to ``--output``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

from model import Model
from preprocess import IMAGENET_MEAN, IMAGENET_STD, load_raw


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset (in-memory tensors so we can shuffle and re-augment cheaply)
# ---------------------------------------------------------------------------


class Img2GPSDataset(Dataset):
    def __init__(self, X_raw: torch.Tensor, y: torch.Tensor, train: bool) -> None:
        self.X_raw = X_raw
        self.y = y
        self.train = train
        self._train_tx = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ]
        )
        self._normalize = transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist())

    def __len__(self) -> int:
        return self.X_raw.shape[0]

    def __getitem__(self, idx: int):
        img = self.X_raw[idx]
        if self.train:
            img = self._train_tx(img)
        img = self._normalize(img)
        return img, self.y[idx]


# ---------------------------------------------------------------------------
# Location-grouped split
# ---------------------------------------------------------------------------


def location_grouped_split(y: torch.Tensor, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    groups: dict[Tuple[float, float], List[int]] = {}
    for i, (lat, lon) in enumerate(y.tolist()):
        key = (round(float(lat), 6), round(float(lon), 6))
        groups.setdefault(key, []).append(i)

    keys = list(groups.keys())
    rng.shuffle(keys)
    target_val = max(1, int(round(val_fraction * len(y))))

    val_idx: List[int] = []
    train_idx: List[int] = []
    for key in keys:
        members = groups[key]
        if len(val_idx) < target_val and len(members) <= max(target_val - len(val_idx), 1):
            val_idx.extend(members)
        else:
            train_idx.extend(members)
    if not val_idx:
        # Fallback when every group is larger than the val budget.
        smallest = min(keys, key=lambda k: len(groups[k]))
        val_idx = list(groups[smallest])
        train_idx = [i for k in keys if k != smallest for i in groups[k]]
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def haversine_meters(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise Haversine distance in meters between (B, 2) lat/lon pairs."""
    radius = 6_371_000.0
    lat1 = torch.deg2rad(pred[:, 0])
    lat2 = torch.deg2rad(target[:, 0])
    dlat = torch.deg2rad(target[:, 0] - pred[:, 0])
    dlon = torch.deg2rad(target[:, 1] - pred[:, 1])
    h = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * radius * torch.asin(torch.clamp(torch.sqrt(h), max=1.0))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _evaluate(model: Model, loader: Iterable, device: torch.device) -> Tuple[float, float]:
    model.eval()
    mse_sum = 0.0
    hav_sum = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            mse = ((preds - targets) ** 2).mean(dim=1)
            hav = haversine_meters(preds, targets)
            batch_size = images.size(0)
            mse_sum += float(mse.sum().item())
            hav_sum += float(hav.sum().item())
            n += batch_size
    return mse_sum / max(n, 1), hav_sum / max(n, 1)


def train(
    csv_path: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
) -> None:
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_raw, y = load_raw(csv_path)
    if len(X_raw) == 0:
        raise RuntimeError(f"No samples loaded from {csv_path}")

    train_idx, val_idx = location_grouped_split(y, val_fraction=val_fraction, seed=seed)
    print(f"data: {len(y)} examples, train={len(train_idx)}, val={len(val_idx)}")

    train_ds = Img2GPSDataset(X_raw[train_idx], y[train_idx], train=True)
    val_ds = Img2GPSDataset(X_raw[val_idx], y[val_idx], train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Model(weights_path=None).to(device)

    train_targets = y[train_idx]
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0, unbiased=False).clamp(min=1e-8)
    model.set_target_stats(target_mean.tolist(), target_std.tolist())
    print(f"target stats: mean={target_mean.tolist()}, std={target_std.tolist()}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_val_hav = math.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            z_targets = model.normalize_targets(targets)

            optimizer.zero_grad()
            z_pred = model.predict_standardized(images)
            loss = criterion(z_pred, z_targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * images.size(0)
            n_seen += images.size(0)

        train_loss = total_loss / max(n_seen, 1)
        val_mse, val_hav = _evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"epoch {epoch:02d}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"train_norm_mse={train_loss:.4f}  val_mse_deg2={val_mse:.6f}  val_haversine_m={val_hav:.2f}"
        )

        if val_hav < best_val_hav:
            best_val_hav = val_hav
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    torch.save(best_state, output_path)
    print(f"best val_haversine_m={best_val_hav:.2f}  saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Img2GPS model.")
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "metadata.csv"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "model.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        args.csv,
        args.output,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.val_fraction,
        args.seed,
    )


if __name__ == "__main__":
    main()
