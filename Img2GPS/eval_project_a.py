from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


def _dynamic_import(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _instantiate_model(model_module, weights_path_override: str | None = None) -> Any:
    """
    Instantiate the student's model in a way that avoids automatic weight loading
    inside their constructor (so we can control loading ourselves).
    """
    # Prefer explicit class if available so we can pass a sentinel weights path
    if hasattr(model_module, "Model"):
        ModelCls = getattr(model_module, "Model")
        try:
            # Pass a non-existent path so student's loader skips default weights
            sentinel = weights_path_override or "__no_weights__.pth"
            return ModelCls(weights_path=sentinel)
        except TypeError:
            # Constructor may not accept weights_path
            return ModelCls()
        except Exception:
            # Fall back to get_model
            if hasattr(model_module, "get_model") and callable(model_module.get_model):
                return model_module.get_model()
            raise
    # Otherwise, try factory
    if hasattr(model_module, "get_model") and callable(model_module.get_model):
        try:
            return model_module.get_model()
        except Exception:
            # As a last resort, try common class names without args
            for cls_name in ["IMG2GPS", "Model"]:
                if hasattr(model_module, cls_name):
                    try:
                        return getattr(model_module, cls_name)()
                    except Exception:
                        continue
            raise
    # Direct class fallback
    for cls_name in ["Model", "IMG2GPS"]:
        if hasattr(model_module, cls_name):
            cls = getattr(model_module, cls_name)
            return cls()
    raise AttributeError("Model module must expose 'get_model()' or a class named 'Model'/'IMG2GPS'.")


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    normalized = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        while key.startswith("backbone.backbone."):
            key = key.replace("backbone.backbone.", "backbone.", 1)
        normalized[key] = v
    return normalized


def _load_state_into_target(target: Any, sd: dict) -> int:
    """
    Load only intersecting keys (and matching shapes) into the target module.
    Returns number of parameters loaded.
    """
    if target is None or not hasattr(target, "state_dict") or not hasattr(target, "load_state_dict"):
        return 0
    target_sd = target.state_dict()
    filtered = {}
    for k, v in sd.items():
        if k in target_sd and isinstance(v, torch.Tensor) and target_sd[k].shape == v.shape:
            filtered[k] = v
    if not filtered:
        return 0
    missing, unexpected = target.load_state_dict(filtered, strict=False)
    # load_state_dict returns a NamedTuple in newer torch; handle tuple/list fallback
    # We don't use missing/unexpected here beyond validation; count by filtered size.
    return len(filtered)


def _load_checkpoint(model: Any, ckpt_path: str | None) -> Any:
    if not ckpt_path:
        if hasattr(model, "eval"):
            model.eval()
        return model
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # Accept either {"state_dict": ...} or plain state dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = _normalize_state_dict_keys(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        sd = _normalize_state_dict_keys(checkpoint)
    else:
        raise RuntimeError("Checkpoint must be a state_dict or {'state_dict': ...} dictionary.")
    # Try loading into inner model first (common wrapper), then wrapper
    total_loaded = 0
    inner = getattr(model, "model", None)
    total_loaded += _load_state_into_target(inner, sd)
    total_loaded += _load_state_into_target(model, sd)
    if total_loaded == 0:
        # Provide actionable debug info
        sample_keys = list(sd.keys())[:10]
        raise RuntimeError(
            "Failed to load any parameters from checkpoint into model. "
            f"Example checkpoint keys after normalization: {sample_keys}"
        )
    if hasattr(model, "eval"):
        model.eval()
    return model


def _predict_in_batches(model: Any, X: List[Any], batch_size: int = 32) -> Tuple[List[Any], float, float]:
    preds: List[Any] = []
    total_s = 0.0
    total_examples = 0
    has_predict = hasattr(model, "predict") and callable(getattr(model, "predict"))
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        start = time.perf_counter()
        if has_predict:
            batch_preds = model.predict(batch)
        else:
            with torch.no_grad():
                outputs = model(batch)  # type: ignore
                if isinstance(outputs, torch.Tensor):
                    batch_preds = outputs.cpu().tolist()
                else:
                    batch_preds = outputs
        end = time.perf_counter()
        infer_time = end - start
        total_s += infer_time
        total_examples += len(batch)
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu().tolist()
        preds.extend(list(batch_preds))
    avg_ms = (total_s / max(total_examples, 1)) * 1000.0
    return preds, total_s, avg_ms


def _resolve_column(columns: List[str], aliases: List[str]) -> str:
    for name in aliases:
        if name in columns:
            return name
    raise KeyError(f"Could not find any of the columns {aliases} in {columns}")


def _load_raw_lat_lon(csv_path: str) -> List[List[float]]:
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    lat_col = _resolve_column(cols, ["Latitude", "latitude", "lat"])
    lon_col = _resolve_column(cols, ["Longitude", "longitude", "lon"])
    labels: List[List[float]] = []
    for _, row in df.iterrows():
        labels.append([float(row[lat_col]), float(row[lon_col])])
    return labels


def _ensure_pairs(arr: List[Any]) -> np.ndarray:
    pairs: List[List[float]] = []
    for item in arr:
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        item_np = np.asarray(item, dtype=np.float64)
        if item_np.shape == (2,):
            pairs.append([float(item_np[0]), float(item_np[1])])
        elif item_np.ndim == 1 and item_np.size == 2:
            pairs.append([float(item_np[0]), float(item_np[1])])
        else:
            raise ValueError(f"Expected 2-length pair, got shape {item_np.shape}")
    return np.asarray(pairs, dtype=np.float64)


def _haversine_m(a: Iterable[float], b: Iterable[float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def compute_metrics(preds: List[Any], targets_raw: List[Any]) -> dict:
    preds_np = _ensure_pairs(preds)
    t_np = _ensure_pairs(targets_raw)
    n = min(len(preds_np), len(t_np))
    preds_np = preds_np[:n]
    t_np = t_np[:n]
    diffs = preds_np - t_np
    mae = float(np.abs(diffs).mean())
    rmse = float(np.sqrt((diffs ** 2).mean()))
    distances = [_haversine_m(p, t) for p, t in zip(preds_np, t_np)]
    avg_distance_m = float(np.mean(distances)) if distances else float("nan")
    return {"mae": mae, "rmse": rmse, "avg_distance_m": avg_distance_m, "num_examples": n}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local evaluator for Project A (img2gps).")
    p.add_argument("--model", required=True, help="Path to student's model.py")
    p.add_argument("--preprocess", required=True, help="Path to student's preprocess.py")
    p.add_argument("--weights", default=None, help="Optional path to model checkpoint (e.g., model.pt)")
    p.add_argument("--csv", required=True, help="Path to validation CSV (e.g., ./val/metadata.csv)")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_mod = _dynamic_import(args.model, "student_model_a")
    preproc_mod = _dynamic_import(args.preprocess, "student_preproc_a")
    # Instantiate while preventing any default weight load from student's constructor
    model = _instantiate_model(model_mod, weights_path_override="__no_weights__.pth")
    model = _load_checkpoint(model, args.weights)

    X, _ = preproc_mod.prepare_data(args.csv)
    if isinstance(X, torch.Tensor):
        inputs = list(X)
    elif isinstance(X, np.ndarray):
        inputs = list(X)
    else:
        inputs = list(X)

    preds, total_s, avg_ms = _predict_in_batches(model, inputs, batch_size=args.batch_size)
    targets_raw = _load_raw_lat_lon(args.csv)
    metrics = compute_metrics(preds, targets_raw)

    print(f"num_examples: {metrics['num_examples']}")
    print(f"avg_infer_ms: {avg_ms:.3f}")
    print(f"total_infer_s: {total_s:.3f}")
    print(f"mae (deg): {metrics['mae']:.6f}")
    print(f"rmse (deg): {metrics['rmse']:.6f}")
    print(f"avg_distance_m: {metrics['avg_distance_m']:.3f}")


if __name__ == "__main__":
    main()


