"""Preprocessing for Project A (Img2GPS).

Public surface used by `Img2GPS/eval_project_a.py`:
    prepare_data(csv_path) -> (X, y)
        X: float tensor (N, 3, 224, 224), ImageNet-normalized.
        y: float tensor (N, 2), raw [latitude, longitude] in degrees.

`load_raw(csv_path)` returns unnormalized [0, 1] tensors and is used by
`train.py` so the training loop can apply augmentation before normalization.

Both calls are cached on disk under `<csv_dir>/.cache/<csv_stem>.<kind>.pt` and
re-used as long as the CSV file mtime has not advanced.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGENET_MEAN_LIST = [0.485, 0.456, 0.406]
IMAGENET_STD_LIST = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Augmentation pipelines used by the released ResNet-18 baseline.
#
# Building these eagerly at import time means notebook cells just do
# ``from preprocess import train_transform, inference_transform`` instead
# of redefining the augmentation recipe inline.
# ---------------------------------------------------------------------------


def _build_transforms():
    """Return ``(train_transform, inference_transform)`` matching the
    course-released baseline notebook exactly:

      * train: RandomResizedCrop(224, scale=(0.7, 1.0)) +
        RandomHorizontalFlip + RandomRotation(15) +
        ColorJitter(0.2, 0.2, 0.2, 0.1) + ImageNet Normalize.
      * inference: ImageNet Normalize only (images already arrive at
        224x224 from ``load_raw``).

    Built lazily so this module stays importable in environments
    without torchvision (e.g. the eval container).
    """
    from torchvision import transforms as T

    train_tx = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.Normalize(mean=IMAGENET_MEAN_LIST, std=IMAGENET_STD_LIST),
        ]
    )
    inference_tx = T.Compose(
        [
            T.Normalize(mean=IMAGENET_MEAN_LIST, std=IMAGENET_STD_LIST),
        ]
    )
    return train_tx, inference_tx


train_transform, inference_transform = _build_transforms()


class LocalGPSImageDataset(Dataset):
    """In-memory drop-in for the released ``GPSImageDataset``.

    Backed by a tensor of pre-resized [0, 1] images (from
    ``load_raw``) plus raw ``[lat, lon]`` labels. Yields
    ``(image, [lat_norm, lon_norm])`` per item, with lat/lon
    standardized using the **training-set** mean/std (computed once
    from the train CSV and reused on the val/reference set, so val
    coordinates are normalized with the same stats as training).

    Attributes ``latitude_mean``, ``latitude_std``, ``longitude_mean``,
    ``longitude_std`` are exposed so callers can denormalize model
    outputs at inference time.
    """

    def __init__(
        self,
        X_raw: torch.Tensor,
        y_raw: torch.Tensor,
        transform=None,
        lat_mean: Optional[float] = None,
        lat_std: Optional[float] = None,
        lon_mean: Optional[float] = None,
        lon_std: Optional[float] = None,
    ) -> None:
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.transform = transform

        lats = y_raw[:, 0].numpy()
        lons = y_raw[:, 1].numpy()
        self.latitude_mean = float(lat_mean) if lat_mean is not None else float(np.mean(lats))
        self.latitude_std = float(lat_std) if lat_std is not None else float(np.std(lats))
        self.longitude_mean = float(lon_mean) if lon_mean is not None else float(np.mean(lons))
        self.longitude_std = float(lon_std) if lon_std is not None else float(np.std(lons))
        self.latitude_std = max(self.latitude_std, 1e-8)
        self.longitude_std = max(self.longitude_std, 1e-8)

    def __len__(self) -> int:
        return self.X_raw.shape[0]

    def __getitem__(self, idx: int):
        image = self.X_raw[idx]  # (3, 224, 224) in [0, 1]
        if self.transform is not None:
            image = self.transform(image)
        lat = (float(self.y_raw[idx, 0]) - self.latitude_mean) / self.latitude_std
        lon = (float(self.y_raw[idx, 1]) - self.longitude_mean) / self.longitude_std
        return image, torch.tensor([lat, lon], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Column / path resolution helpers
# ---------------------------------------------------------------------------


def _resolve_column(columns: List[str], aliases: List[str]) -> str:
    for name in aliases:
        if name in columns:
            return name
    lower_to_original = {name.lower(): name for name in columns}
    for name in aliases:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    raise KeyError(f"Could not find any of columns {aliases} in CSV columns {columns}")


def _candidate_image_paths(csv_path: str, image_value: str) -> List[str]:
    image_value = str(image_value)
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    repo_root = os.path.abspath(os.path.join(csv_dir, os.pardir))
    cwd = os.getcwd()
    bare = os.path.basename(image_value)
    return [
        image_value,
        os.path.join(csv_dir, image_value),
        os.path.join(csv_dir, bare),
        os.path.join(csv_dir, "images", bare),
        os.path.join(cwd, image_value),
        os.path.join(repo_root, image_value),
        os.path.join(repo_root, "data", "images", bare),
        os.path.join(repo_root, "data", "images_converted", bare),
    ]


def _resolve_image_path(csv_path: str, image_value: str) -> str:
    if os.path.isabs(image_value) and os.path.exists(image_value):
        return image_value
    for candidate in _candidate_image_paths(csv_path, image_value):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate image '{image_value}' for CSV '{csv_path}'")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_image_raw(path: str) -> torch.Tensor:
    """Return a float tensor of shape (3, 224, 224) with values in [0, 1]."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std normalization to a (..., 3, H, W) tensor."""
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


# ---------------------------------------------------------------------------
# Cached batch loaders
# ---------------------------------------------------------------------------


def _cache_path(csv_path: str, kind: str) -> Path:
    csv_dir = Path(os.path.abspath(csv_path)).parent
    stem = Path(csv_path).stem
    return csv_dir / ".cache" / f"{stem}.{kind}.pt"


def _load_cache(cache_file: Path, csv_path: str):
    if not cache_file.exists():
        return None
    if cache_file.stat().st_mtime < os.path.getmtime(csv_path):
        return None
    try:
        return torch.load(cache_file, map_location="cpu")
    except Exception:
        return None


def _save_cache(cache_file: Path, payload) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_file)


def _read_csv_records(csv_path: str) -> Tuple[List[str], List[List[float]]]:
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    image_col = _resolve_column(columns, ["image_path", "filepath", "image", "path", "file_name", "file_path", "filename"])
    lat_col = _resolve_column(columns, ["Latitude", "latitude", "lat"])
    lon_col = _resolve_column(columns, ["Longitude", "longitude", "lon", "lng"])
    paths = [_resolve_image_path(csv_path, row[image_col]) for _, row in df.iterrows()]
    labels = [[float(row[lat_col]), float(row[lon_col])] for _, row in df.iterrows()]
    return paths, labels


def load_raw(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (X_raw, y) where X_raw is [0, 1] in shape (N, 3, 224, 224)."""
    cache_file = _cache_path(csv_path, "raw")
    cached = _load_cache(cache_file, csv_path)
    if cached is not None:
        return cached["X"], cached["y"]

    paths, labels = _read_csv_records(csv_path)
    images = [_load_image_raw(p) for p in paths]
    X = torch.stack(images, dim=0) if images else torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty(0, 2)
    _save_cache(cache_file, {"X": X, "y": y})
    return X, y


def prepare_data(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """ImageNet-normalized images and raw lat/lon labels (used by the evaluator)."""
    cache_file = _cache_path(csv_path, "norm")
    cached = _load_cache(cache_file, csv_path)
    if cached is not None:
        return cached["X"], cached["y"]

    X_raw, y = load_raw(csv_path)
    X = normalize_imagenet(X_raw)
    _save_cache(cache_file, {"X": X, "y": y})
    return X, y
