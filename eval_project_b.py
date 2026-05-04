from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from typing import Any, List, Tuple, Dict

import numpy as np
import torch


def _dynamic_import(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _instantiate_model(model_module) -> Any:
    # Prefer explicit class instantiation to avoid unintended default weight loading
    for cls_name in ["Model", "NewsClassifier"]:
        if hasattr(model_module, cls_name):
            cls = getattr(model_module, cls_name)
            try:
                return cls(weights_path="__no_weights__.pth")
            except TypeError:
                return cls()
    if hasattr(model_module, "get_model") and callable(model_module.get_model):
        return model_module.get_model()
    raise AttributeError("Model module must expose 'get_model()' or a class named 'Model'/'NewsClassifier'.")

def _normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        normalized[key] = v
    return normalized


def _load_state_into_target(target: Any, sd: Dict[str, Any]) -> int:
    if target is None or not hasattr(target, "state_dict") or not hasattr(target, "load_state_dict"):
        return 0
    target_sd = target.state_dict()
    filtered: Dict[str, Any] = {}
    for k, v in sd.items():
        if k in target_sd and isinstance(v, torch.Tensor) and target_sd[k].shape == v.shape:
            filtered[k] = v
    if not filtered:
        return 0
    target.load_state_dict(filtered, strict=False)
    return len(filtered)


def _load_checkpoint(model: Any, ckpt_path: str | None) -> Any:
    if not ckpt_path:
        if hasattr(model, "eval"):
            model.eval()
        return model
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = _normalize_state_dict_keys(checkpoint["state_dict"])  # type: ignore[index]
    elif isinstance(checkpoint, dict):
        sd = _normalize_state_dict_keys(checkpoint)  # type: ignore[arg-type]
    else:
        raise RuntimeError("Checkpoint must be a state_dict or {'state_dict': ...} dictionary.")
    total_loaded = 0
    inner = getattr(model, "model", None)
    total_loaded += _load_state_into_target(inner, sd)
    total_loaded += _load_state_into_target(model, sd)
    if total_loaded == 0:
        sample = list(sd.keys())[:10]
        raise RuntimeError(f"Failed to load any parameters from checkpoint. Example keys: {sample}")
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
                outputs = model(batch)
                if hasattr(outputs, "argmax"):
                    batch_preds = outputs.argmax(dim=-1).cpu().tolist()
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


def _coerce_to_str_list(values: List[Any]) -> List[str]:
    return [str(v) for v in values]


def accuracy_robust(preds: List[Any], targets: List[Any]) -> float:
    if len(preds) == 0 or len(targets) == 0:
        return 0.0
    if all(isinstance(p, str) for p in preds) and all(isinstance(t, str) for t in targets):
        return float(sum(int(p == t) for p, t in zip(preds, targets)) / max(1, len(targets)))
    if all(isinstance(p, (int, np.integer)) for p in preds) and all(isinstance(t, (int, np.integer)) for t in targets):
        return float(sum(int(p == t) for p, t in zip(preds, targets)) / max(1, len(targets)))
    preds_list = list(preds)
    targets_list = list(targets)
    preds_list = [int(p) if isinstance(p, (np.integer,)) else p for p in preds_list]
    targets_list = [int(t) if isinstance(t, (np.integer,)) else t for t in targets_list]
    if all(isinstance(p, int) for p in preds_list) and not all(isinstance(t, int) for t in targets_list):
        unique_targets = list(dict.fromkeys(_coerce_to_str_list(targets_list)))
        if len(unique_targets) == 2:
            a, b = unique_targets[0], unique_targets[1]
            map1 = {a: 0, b: 1}
            acc1 = sum(int(p == map1[_t]) for p, _t in zip(preds_list, _coerce_to_str_list(targets_list)))
            map2 = {a: 1, b: 0}
            acc2 = sum(int(p == map2[_t]) for p, _t in zip(preds_list, _coerce_to_str_list(targets_list)))
            return float(max(acc1, acc2) / max(1, len(targets_list)))
    if not all(isinstance(p, int) for p in preds_list) and all(isinstance(t, int) for t in targets_list):
        unique_preds = list(dict.fromkeys(_coerce_to_str_list(preds_list)))
        if len(unique_preds) == 2:
            a, b = unique_preds[0], unique_preds[1]
            map1 = {a: 0, b: 1}
            acc1 = sum(int(map1[_p] == t) for _p, t in zip(_coerce_to_str_list(preds_list), targets_list))
            map2 = {a: 1, b: 0}
            acc2 = sum(int(map2[_p] == t) for _p, t in zip(_coerce_to_str_list(preds_list), targets_list))
            return float(max(acc1, acc2) / max(1, len(targets_list)))
    preds_str = _coerce_to_str_list(preds_list)
    targets_str = _coerce_to_str_list(targets_list)
    return float(sum(int(p == t) for p, t in zip(preds_str, targets_str)) / max(1, len(targets_str)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local evaluator for Project B.")
    p.add_argument("--model", required=True, help="Path to student's model.py")
    p.add_argument("--preprocess", required=True, help="Path to student's preprocess.py")
    p.add_argument("--csv", required=True, help="Path to validation CSV (e.g., url_val.csv)")
    p.add_argument("--weights", default=None, help="Optional path to model checkpoint (e.g., model.pt)")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_mod = _dynamic_import(args.model, "student_model_b")
    preproc_mod = _dynamic_import(args.preprocess, "student_preproc_b")
    model = _instantiate_model(model_mod)
    model = _load_checkpoint(model, args.weights)

    X, y = preproc_mod.prepare_data(args.csv)

    if isinstance(X, torch.Tensor):
        inputs = list(X)
    elif isinstance(X, np.ndarray):
        inputs = list(X)
    else:
        inputs = list(X)
    targets = list(y)

    preds, total_s, avg_ms = _predict_in_batches(model, inputs, batch_size=args.batch_size)
    acc = accuracy_robust(preds, targets)

    print(f"num_examples: {len(targets)}")
    print(f"avg_infer_ms: {avg_ms:.3f}")
    print(f"total_infer_s: {total_s:.3f}")
    print(f"accuracy: {acc:.6f}")


if __name__ == "__main__":
    main()


