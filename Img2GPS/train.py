"""Build the Img2GPS submission checkpoint -- DINOv2 retrieval gallery.

This is **not** a gradient-fine-tuning script. The encoder (DINOv2 ViT-S/14)
stays frozen at its self-supervised ImageNet weights; "training" here
means:

    1. forward every collected photo through the encoder (CPU/GPU, ~1s
       each on CPU; ~100ms on GPU)
    2. take the CLS token, L2-normalize, store into the gallery
    3. fit a single scalar (the softmax temperature) on a held-out
       location-grouped val split, minimizing Haversine on val

That's it. The output ``model.pt`` is a single dict:

    {
        "encoder_state_dict" : DinoVitS14.state_dict(),
        "gallery_emb"        : (N_train, 384) L2-normalized,
        "gallery_gps"        : (N_train, 2)   raw [lat, lon] degrees,
        "temperature"        : float scalar,
        "version"            : "dino_retrieval_v1",
    }

Three split modes (mutually exclusive, same shape as the prior train.py
so notebooks keep working):

    * default               -> single location-grouped train/val split.
                                Tunes temperature on val. Saves the train
                                gallery + tuned T.
    * --bootstrap-rounds B  -> B independent location-level bootstrap
                                cycles; reports mean +/- std OOB Haversine.
                                Saves the *best round*'s gallery + T.
    * --use-all-data        -> use every example as gallery, no holdout.
                                Temperature stays at default. Best for
                                the final model AFTER picking T.

Loss / metric: when fitting the softmax **temperature** T, we minimize
**mean Haversine (meters)** on a location-grouped holdout — not MSE.
(The legacy ResNet notebook still uses MSE; this script does not.)

Augmentation: NONE at gallery-build time (we want canonical embeddings
per photo). Light test-time augmentation can be added later in model.py.

``--use-all-data``: the **gallery** uses every photo; T is still fit on
a **location holdout** (same ``--val-fraction``) so the submit checkpoint
gets a data-driven T without dropping rows from the gallery.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import DinoVitS14, Model, load_official_dinov2_vits14
from preprocess import IMAGENET_MEAN, IMAGENET_STD, load_raw


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Location-grouped splits
# ---------------------------------------------------------------------------


def _group_by_location(y: torch.Tensor) -> "dict[Tuple[float, float], List[int]]":
    groups: dict[Tuple[float, float], List[int]] = {}
    for i, (lat, lon) in enumerate(y.tolist()):
        key = (round(float(lat), 6), round(float(lon), 6))
        groups.setdefault(key, []).append(i)
    return groups


def location_grouped_split(
    y: torch.Tensor, val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Single train/val split where photos sharing a (rounded) GPS land
    entirely on one side -- no per-spot leakage."""
    rng = random.Random(seed)
    groups = _group_by_location(y)
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
        smallest = min(keys, key=lambda k: len(groups[k]))
        val_idx = list(groups[smallest])
        train_idx = [i for k in keys if k != smallest for i in groups[k]]
    return train_idx, val_idx


def bootstrap_location_split(
    y: torch.Tensor, seed: int
) -> Tuple[List[int], List[int]]:
    """Location-level bootstrap: sample locations WITH replacement; all
    photos at unsampled locations form the OOB val set."""
    rng = random.Random(seed)
    groups = _group_by_location(y)
    keys = list(groups.keys())
    sampled_keys = [rng.choice(keys) for _ in range(len(keys))]
    sampled_set = set(sampled_keys)
    train_idx: List[int] = []
    for key in sampled_keys:
        train_idx.extend(groups[key])
    oob_idx: List[int] = []
    for key in keys:
        if key not in sampled_set:
            oob_idx.extend(groups[key])
    return train_idx, oob_idx


# ---------------------------------------------------------------------------
# Differentiable Haversine (meters) -- used as both training loss and metric
# ---------------------------------------------------------------------------


def haversine_meters(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise Haversine in meters between (B, 2) lat/lon degree pairs.

    Differentiable in both inputs; safe near zero (square root is
    bounded away from a singular gradient by the asin's clamp at 1.0).
    """
    radius = 6_371_000.0
    lat1 = torch.deg2rad(pred[:, 0])
    lat2 = torch.deg2rad(target[:, 0])
    dlat = torch.deg2rad(target[:, 0] - pred[:, 0])
    dlon = torch.deg2rad(target[:, 1] - pred[:, 1])
    h = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * radius * torch.asin(torch.clamp(torch.sqrt(h.clamp_min(0.0)), max=1.0))


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------


def _build_encoder(device: torch.device, weights_path: str | None = None) -> DinoVitS14:
    encoder = load_official_dinov2_vits14(weights_path=weights_path)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def _embed_all(
    encoder: DinoVitS14,
    X_raw: torch.Tensor,
    indices: List[int],
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Forward each image once through the encoder, return L2-normalized
    (n, 384) tensor of CLS embeddings on CPU.
    """
    encoder.eval()
    embs: List[torch.Tensor] = []
    for i in range(0, len(indices), batch_size):
        chunk = indices[i : i + batch_size]
        x = X_raw[chunk].to(device)
        x = _imagenet_normalize(x)
        emb = encoder(x)
        emb = F.normalize(emb, dim=-1).cpu()
        embs.append(emb)
    return torch.cat(embs, dim=0) if embs else torch.zeros(0, DinoVitS14.EMBED_DIM)


# ---------------------------------------------------------------------------
# Temperature tuning + evaluation
# ---------------------------------------------------------------------------


def _tune_temperature(
    train_emb: torch.Tensor,         # (N_t, D)
    train_gps: torch.Tensor,         # (N_t, 2)
    val_emb: torch.Tensor,           # (N_v, D)
    val_gps: torch.Tensor,           # (N_v, 2)
    init_T: float = 20.0,
    steps: int = 800,
    lr: float = 1.0,
    log_prefix: str = "",
) -> Tuple[float, float]:
    """Fit the softmax temperature by minimizing Haversine on val.

    Returns (best_T, best_val_haversine_m).
    """
    if val_emb.numel() == 0:
        return init_T, float("nan")

    log_T_min, log_T_max = math.log(1e-2), math.log(200.0)
    log_T = torch.tensor(math.log(init_T), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([log_T], lr=lr * 0.05)  # log-space updates: small lr is plenty

    sim = val_emb @ train_emb.t()                      # (N_v, N_t), constant
    best_T = float(init_T)
    best_loss = math.inf
    for step in range(steps):
        T = torch.exp(log_T)
        weights = torch.softmax(sim * T, dim=-1)
        pred = weights @ train_gps
        loss = haversine_meters(pred, val_gps).mean()
        if not torch.isfinite(loss):
            break
        cur = float(loss.item())
        if cur < best_loss:
            best_loss = cur
            best_T = float(T.item())                   # T at the value that produced this loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_T.clamp_(min=log_T_min, max=log_T_max)
    print(f"{log_prefix}temperature tune: T={best_T:.2f}  val_haversine={best_loss:.2f}m")
    return best_T, best_loss


def _evaluate_with_gallery(
    train_emb: torch.Tensor,
    train_gps: torch.Tensor,
    val_emb: torch.Tensor,
    val_gps: torch.Tensor,
    temperature: float,
) -> float:
    """Run the soft-retrieval prediction and return mean Haversine (m)."""
    if val_emb.numel() == 0:
        return float("nan")
    sim = val_emb @ train_emb.t()
    weights = torch.softmax(sim * float(temperature), dim=-1)
    pred = weights @ train_gps
    return float(haversine_meters(pred, val_gps).mean().item())


# ---------------------------------------------------------------------------
# One training cycle (single split or one bootstrap round)
# ---------------------------------------------------------------------------


def _build_gallery_and_tune(
    *,
    encoder: DinoVitS14,
    X_raw: torch.Tensor,
    y: torch.Tensor,
    train_idx: List[int],
    val_idx: List[int],
    device: torch.device,
    log_prefix: str = "",
    temp_steps: int = 800,
) -> Tuple[float, dict]:
    """Returns (val_haversine, payload_for_model_pt).

    The payload is a single flat ``state_dict`` so the staff evaluator's
    shape-filtered ``load_state_dict`` flow accepts it directly.
    """
    train_emb = _embed_all(encoder, X_raw, train_idx, device)
    train_gps = y[train_idx].clone()

    if val_idx:
        val_emb = _embed_all(encoder, X_raw, val_idx, device)
        val_gps = y[val_idx].clone()
        T, val_hav = _tune_temperature(
            train_emb,
            train_gps,
            val_emb,
            val_gps,
            steps=temp_steps,
            log_prefix=log_prefix,
        )
    else:
        T, val_hav = 20.0, float("nan")
        print(f"{log_prefix}no val: temperature stays at default T={T:.2f}")

    payload = _payload_from_gallery(encoder, train_emb, train_gps, T)
    return val_hav, payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _payload_from_gallery(
    encoder: DinoVitS14,
    gallery_emb: torch.Tensor,
    gallery_gps: torch.Tensor,
    temperature: float,
) -> dict:
    flat = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    flat["gallery_emb"] = gallery_emb.to(torch.float32)
    flat["gallery_gps"] = gallery_gps.to(torch.float32)
    flat["temperature"] = torch.tensor(float(temperature), dtype=torch.float32)
    return {"state_dict": flat, "version": "dino_retrieval_v1"}


def train(
    csv_path: str,
    output_path: str,
    val_fraction: float,
    seed: int,
    bootstrap_rounds: int = 0,
    use_all_data: bool = False,
    dinov2_weights: str | None = None,
    temp_steps: int = 800,
) -> float:
    if use_all_data and bootstrap_rounds > 0:
        raise ValueError(
            "--use-all-data and --bootstrap-rounds are mutually exclusive."
        )

    _seed_everything(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    X_raw, y = load_raw(csv_path)
    if len(X_raw) == 0:
        raise RuntimeError(f"No samples loaded from {csv_path}")
    print(f"loaded {len(X_raw)} photos at {len(_group_by_location(y))} unique locations")

    encoder = _build_encoder(device, weights_path=dinov2_weights)
    print(f"encoder: DINOv2 ViT-S/14 (frozen), {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M params")

    if use_all_data:
        all_idx = list(range(len(y)))
        tune_tr, tune_va = location_grouped_split(
            y, val_fraction=val_fraction, seed=seed
        )
        T = 20.0
        tune_hav = float("nan")
        if tune_va:
            tr_e = _embed_all(encoder, X_raw, tune_tr, device)
            va_e = _embed_all(encoder, X_raw, tune_va, device)
            T, tune_hav = _tune_temperature(
                tr_e,
                y[tune_tr],
                va_e,
                y[tune_va],
                steps=temp_steps,
                log_prefix="[all-data T] ",
            )
            print(
                f"data: {len(y)} examples, gallery={len(all_idx)} (all photos); "
                f"T fit on {len(tune_tr)} train locs / {len(tune_va)} val photos "
                f"(holdout Haversine {tune_hav:.2f} m)"
            )
        else:
            print(
                f"data: {len(y)} examples, gallery={len(all_idx)} (all photos); "
                f"no loc-holdout for T -> default T={T:.2f}"
            )
        full_emb = _embed_all(encoder, X_raw, all_idx, device)
        full_gps = y[all_idx].clone()
        payload = _payload_from_gallery(encoder, full_emb, full_gps, T)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        torch.save(payload, output_path)
        print(f"saved gallery checkpoint to {output_path}")
        print(
            "RESULT mode=all_data "
            f"seed={seed} bootstrap_rounds=0 "
            f"val_haversine_m={tune_hav:.4f} "
            f"val_haversine_m_mean={tune_hav:.4f} "
            f"val_haversine_m_std=0.0000"
        )
        return tune_hav

    if bootstrap_rounds > 0:
        oob_havs: List[float] = []
        best_round_hav = math.inf
        best_round_payload: dict = {}
        best_round_idx = -1
        for r in range(bootstrap_rounds):
            round_seed = seed + r
            train_idx, val_idx = bootstrap_location_split(y, seed=round_seed)
            if not val_idx:
                print(f"[r{r+1}] empty OOB; skipping")
                continue
            print(
                f"[r{r+1}/{bootstrap_rounds}] gallery={len(train_idx)} (with dupes), oob={len(val_idx)}"
            )
            val_hav, payload = _build_gallery_and_tune(
                encoder=encoder, X_raw=X_raw, y=y,
                train_idx=train_idx, val_idx=val_idx, device=device,
                log_prefix=f"[r{r+1}] ",
                temp_steps=temp_steps,
            )
            oob_havs.append(val_hav)
            if val_hav < best_round_hav:
                best_round_hav = val_hav
                best_round_payload = payload
                best_round_idx = r

        if not oob_havs:
            raise RuntimeError("All bootstrap rounds produced empty OOB sets.")

        mean_hav = float(np.mean(oob_havs))
        std_hav = float(np.std(oob_havs))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        torch.save(best_round_payload, output_path)
        print(
            f"\nbootstrap done: B={len(oob_havs)} rounds, "
            f"mean_oob_haversine_m={mean_hav:.2f} +/- {std_hav:.2f}, "
            f"best round={best_round_idx + 1} ({best_round_hav:.2f}m) -> {output_path}"
        )
        print(
            "RESULT mode=bootstrap "
            f"seed={seed} bootstrap_rounds={len(oob_havs)} "
            f"val_haversine_m={mean_hav:.4f} "
            f"val_haversine_m_mean={mean_hav:.4f} "
            f"val_haversine_m_std={std_hav:.4f}"
        )
        return mean_hav

    # Default: single location-grouped split, tune T on val.
    train_idx, val_idx = location_grouped_split(y, val_fraction=val_fraction, seed=seed)
    print(f"data: {len(y)} examples, gallery={len(train_idx)}, val={len(val_idx)}")
    val_hav, payload = _build_gallery_and_tune(
        encoder=encoder, X_raw=X_raw, y=y,
        train_idx=train_idx, val_idx=val_idx, device=device,
        temp_steps=temp_steps,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    torch.save(payload, output_path)
    print(f"saved gallery checkpoint (val_haversine_m={val_hav:.2f}) to {output_path}")
    print(
        "RESULT mode=split "
        f"seed={seed} bootstrap_rounds=0 "
        f"val_haversine_m={val_hav:.4f} "
        f"val_haversine_m_mean={val_hav:.4f} "
        f"val_haversine_m_std=0.0000"
    )
    return val_hav


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Img2GPS DINOv2 retrieval checkpoint.")
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "metadata.csv"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "model.pt"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=0,
        help=(
            "If > 0, run B independent location-level bootstrap rounds. "
            "Each samples locations with replacement to form the gallery; "
            "OOB locations form the val set. The RESULT line reports "
            "mean +/- std OOB Haversine across rounds."
        ),
    )
    parser.add_argument(
        "--use-all-data",
        action="store_true",
        help=(
            "Gallery = every photo. Temperature T is still fit by Haversine on a "
            "location-grouped holdout (see --val-fraction); holdout photos remain "
            "in the gallery."
        ),
    )
    parser.add_argument(
        "--temp-steps",
        type=int,
        default=800,
        help="Adam steps for Haversine minimization over softmax temperature T.",
    )
    parser.add_argument(
        "--dinov2-weights",
        default=None,
        help=(
            "Optional path to a local DINOv2 ViT-S/14 .pth checkpoint. If "
            "omitted, downloads from the official URL via torch.hub cache."
        ),
    )
    args = parser.parse_args()
    train(
        csv_path=args.csv,
        output_path=args.output,
        val_fraction=args.val_fraction,
        seed=args.seed,
        bootstrap_rounds=args.bootstrap_rounds,
        use_all_data=args.use_all_data,
        dinov2_weights=args.dinov2_weights,
        temp_steps=args.temp_steps,
    )


if __name__ == "__main__":
    main()
