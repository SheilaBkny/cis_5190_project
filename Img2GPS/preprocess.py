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
from typing import List, Tuple

import cv2
import pandas as pd
import torch


IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
    image_col = _resolve_column(columns, ["image_path", "path", "filepath", "file_path", "filename", "file_name"])
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
