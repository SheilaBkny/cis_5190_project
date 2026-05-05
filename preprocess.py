import os
from typing import List, Tuple

import cv2
import pandas as pd
import torch


IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resolve_column(columns: List[str], aliases: List[str]) -> str:
    for name in aliases:
        if name in columns:
            return name
    lower_to_original = {name.lower(): name for name in columns}
    for name in aliases:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    raise KeyError(f"Could not find any of columns {aliases} in CSV columns {columns}")


def _resolve_image_path(csv_path: str, image_value: str) -> str:
    image_value = str(image_value)
    if os.path.isabs(image_value) and os.path.exists(image_value):
        return image_value

    candidates = [
        image_value,
        os.path.join(os.path.dirname(csv_path), image_value),
        os.path.join(os.path.dirname(csv_path), "images", image_value),
        os.path.join(os.getcwd(), image_value),
        os.path.join(os.getcwd(), "data", "images", image_value),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate image '{image_value}' for CSV '{csv_path}'")


def _load_image(path: str) -> torch.Tensor:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def prepare_data(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load Img2GPS examples from a metadata CSV.

    Returns:
        X: float tensor with shape (N, 3, 224, 224), ImageNet-normalized.
        y: float tensor with shape (N, 2), raw [latitude, longitude] degrees.
    """
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    image_col = _resolve_column(columns, ["image_path", "path", "filepath", "file_path", "filename", "file_name"])
    lat_col = _resolve_column(columns, ["Latitude", "latitude", "lat"])
    lon_col = _resolve_column(columns, ["Longitude", "longitude", "lon", "lng"])

    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = _resolve_image_path(csv_path, row[image_col])
        images.append(_load_image(image_path))
        labels.append([float(row[lat_col]), float(row[lon_col])])

    X = torch.stack(images, dim=0) if images else torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty(0, 2)
    return X, y
