"""Train the Img2GPS submission model (ResNet-18 + 2-D regression).

Architecture: see ``Img2GPS/model.py``. Training recipe matches the
course-released baseline notebook:

    * ImageNet ResNet-18, full fine-tune
    * MSE loss in standardized lat/lon space
    * Adam(lr=1e-3, weight_decay=1e-4), StepLR(step_size=5, gamma=0.1)
    * Augmentation: RandomResizedCrop(224, scale=(0.7, 1.0)) +
      RandomHorizontalFlip + RandomRotation(15) + ColorJitter +
      ImageNet Normalize
    * Per-batch gradient-norm clip at 1.0 + per-batch NaN guard
    * Best-by-val-Haversine state restored before saving (so a late-
      epoch divergence does not corrupt the saved checkpoint)

Three split modes (mutually exclusive, same shape as the previous
cluster-classifier ``train.py`` so notebooks keep working):

    * default            -> single location-grouped train/val split.
                            Saves best-by-val-Haversine state.
    * --bootstrap-rounds -> B independent location-level bootstrap
                            cycles. RESULT reports mean ± std OOB
                            Haversine. Best for HP search on tiny data.
    * --use-all-data     -> train on every example, no holdout. Best
                            for the final model AFTER picking HPs.

Output: a torchvision-style ``resnet18.state_dict()`` saved to
``--output`` (default ``Img2GPS/model.pt``). ``Img2GPS/model.py``'s
``Model`` constructor auto-loads this file at construction time, and
``Img2GPS/eval_project_a.py`` loads it via the inner ``self.model``.

The reference set (``Img2GPS/reference/metadata.csv``) is NOT used by
this script. It is only for post-hoc held-out evaluation via the
notebook or ``eval_project_a.py``.
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
# Dataset
# ---------------------------------------------------------------------------


class Img2GPSDataset(Dataset):
    """In-memory dataset for the ResNet-18 baseline.

    Yields ``(image, y_standardized, y_raw)``:

      * ``image``           - augmented + ImageNet-normalized (3,224,224)
      * ``y_standardized``  - (y_raw - y_mean) / y_std, the MSE target
      * ``y_raw``           - raw [lat, lon] degrees, for Haversine

    Standardization stats are passed in (population stats over the
    *training split*, computed once by the caller). The model's
    ``Model._TARGET_MEAN`` / ``_TARGET_STD`` are also seeded from the
    full ``metadata.csv``, so train-time and inference-time stats match
    exactly when ``--csv`` is the default file.
    """

    def __init__(
        self,
        X_raw: torch.Tensor,
        y_raw: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        train: bool,
    ) -> None:
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.y_mean = y_mean
        self.y_std = y_std
        self.train = train
        self._train_tx = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        )
        self._normalize = transforms.Normalize(
            mean=IMAGENET_MEAN.flatten().tolist(),
            std=IMAGENET_STD.flatten().tolist(),
        )

    def __len__(self) -> int:
        return self.X_raw.shape[0]

    def __getitem__(self, idx: int):
        img = self.X_raw[idx]
        if self.train:
            img = self._train_tx(img)
        img = self._normalize(img)
        y_raw = self.y_raw[idx]
        y_std = (y_raw - self.y_mean) / self.y_std
        return img, y_std, y_raw


# ---------------------------------------------------------------------------
# Location-grouped train/val split (and bootstrap)
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
    """Single train/val split where photos sharing a GPS location are
    kept entirely on one side of the split (no per-photo leakage).
    """
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
    """Location-level bootstrap: sample ``len(unique_locations)`` keys
    WITH REPLACEMENT (a location drawn 3x contributes its images 3x to
    the gradient); all images at *unsampled* locations form the OOB val
    set. ~37% of locations are OOB on average per round.
    """
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


def _evaluate(
    model: Model, loader: Iterable, device: torch.device
) -> Tuple[float, float]:
    """Return (mse_in_standardized_space, haversine_meters_mean)."""
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    mse_sum = 0.0
    hav_sum = 0.0
    n = 0
    with torch.no_grad():
        for images, y_std, y_raw in loader:
            images = images.to(device)
            y_std = y_std.to(device)
            y_raw = y_raw.to(device)
            preds_std = model.predict_standardized(images)
            mse_sum += float(mse_loss(preds_std, y_std).item())
            preds_raw = model(images)
            hav_sum += float(haversine_meters(preds_raw, y_raw).sum().item())
            n += images.size(0)
    return mse_sum / max(n, 1), hav_sum / max(n, 1)


def _train_one_cycle(
    *,
    X_raw: torch.Tensor,
    y_raw: torch.Tensor,
    train_idx: List[int],
    val_idx: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    log_prefix: str = "",
) -> Tuple[float, dict]:
    """Run ``epochs`` of training. Returns (best_val_haversine_m,
    best_state_dict_cpu). With ``val_idx == []`` the metric is the
    final-epoch train Haversine and the saved state is the final-epoch
    state.
    """
    train_y = y_raw[train_idx]
    y_mean = train_y.mean(dim=0)
    y_std = train_y.std(dim=0, unbiased=False).clamp_min(1e-8)

    train_ds = Img2GPSDataset(
        X_raw[train_idx], y_raw[train_idx], y_mean, y_std, train=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    has_val = len(val_idx) > 0
    if has_val:
        val_ds = Img2GPSDataset(
            X_raw[val_idx], y_raw[val_idx], y_mean, y_std, train=False
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None

    model = Model(weights_path=None).to(device)
    # Seed the wrapped resnet from torchvision's ImageNet weights when
    # we can; full fine-tuning from scratch on 89 images doesn't work.
    try:
        from torchvision import models as tvm
        pretrained = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        in_features = pretrained.fc.in_features
        pretrained.fc = nn.Linear(in_features, 2)
        model.model.load_state_dict(pretrained.state_dict(), strict=True)
    except Exception as exc:  # offline runner / weights download blocked
        print(f"{log_prefix}warning: could not load ImageNet weights ({exc}); "
              "training the regression head from scratch initialization")

    # Overwrite the model's hard-coded (y_mean, y_std) buffers so its
    # forward() denormalizes with the *training-split* stats. The
    # train-then-restore loop is identical to the notebook's recipe.
    model.y_mean.copy_(y_mean.to(model.y_mean.device))
    model.y_std.copy_(y_std.to(model.y_std.device))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_metric = math.inf
    best_state: dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running_mse = 0.0
        running_hav = 0.0
        n_seen = 0
        n_skipped = 0
        for images, y_std_b, y_raw_b in train_loader:
            images = images.to(device)
            y_std_b = y_std_b.to(device)
            y_raw_b = y_raw_b.to(device)

            optimizer.zero_grad()
            preds_std = model.predict_standardized(images)
            loss = criterion(preds_std, y_std_b)
            if not torch.isfinite(loss):
                n_skipped += 1
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                preds_raw = preds_std * model.y_std + model.y_mean
                hav = haversine_meters(preds_raw, y_raw_b)
            running_mse += float(loss.item()) * images.size(0)
            running_hav += float(hav.sum().item())
            n_seen += images.size(0)

        scheduler.step()
        train_mse = running_mse / max(n_seen, 1)
        train_hav = running_hav / max(n_seen, 1)

        if has_val:
            val_mse, val_hav = _evaluate(model, val_loader, device)
            skip_msg = f"  [skipped {n_skipped} non-finite batches]" if n_skipped else ""
            print(
                f"{log_prefix}epoch {epoch:02d}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"train_mse={train_mse:.4f}  train_hav_m={train_hav:.1f}  "
                f"val_mse={val_mse:.4f}  val_hav_m={val_hav:.1f}{skip_msg}"
            )
            metric = val_hav
        else:
            skip_msg = f"  [skipped {n_skipped} non-finite batches]" if n_skipped else ""
            print(
                f"{log_prefix}epoch {epoch:02d}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"train_mse={train_mse:.4f}  train_hav_m={train_hav:.1f}  "
                f"(no val: training on all data){skip_msg}"
            )
            metric = train_hav

        if math.isfinite(metric) and metric < best_metric:
            best_metric = metric
            # Save only the wrapped resnet's state_dict so the file is
            # the canonical torchvision format that eval_project_a.py
            # loads via getattr(model, "model", None).
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()
            }

    if not best_state:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()
        }

    return best_metric, best_state


def train(
    csv_path: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
    bootstrap_rounds: int = 0,
    use_all_data: bool = False,
) -> float:
    if use_all_data and bootstrap_rounds > 0:
        raise ValueError(
            "--use-all-data and --bootstrap-rounds are mutually exclusive."
        )

    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_raw, y = load_raw(csv_path)
    if len(X_raw) == 0:
        raise RuntimeError(f"No samples loaded from {csv_path}")

    common_kwargs = dict(
        X_raw=X_raw,
        y_raw=y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )

    if use_all_data:
        train_idx = list(range(len(y)))
        print(f"data: {len(y)} examples, train={len(train_idx)}, val=0 (use-all-data)")
        best_train_hav, best_state = _train_one_cycle(
            train_idx=train_idx, val_idx=[], **common_kwargs
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        torch.save(best_state, output_path)
        print(f"final train_haversine_m={best_train_hav:.2f}  saved to {output_path}")
        print(
            "RESULT "
            f"mode=all_data "
            f"lr={lr:g} epochs={epochs} batch_size={batch_size} weight_decay={weight_decay:g} "
            f"seed={seed} bootstrap_rounds=0 "
            f"val_haversine_m={best_train_hav:.4f} "
            f"val_haversine_m_mean={best_train_hav:.4f} "
            f"val_haversine_m_std=0.0000"
        )
        return best_train_hav

    if bootstrap_rounds > 0:
        oob_havs: List[float] = []
        best_round_hav = math.inf
        best_round_state: dict = {}
        best_round_idx = -1
        for r in range(bootstrap_rounds):
            round_seed = seed + r
            train_idx, val_idx = bootstrap_location_split(y, seed=round_seed)
            if not val_idx:
                print(f"[round {r+1}/{bootstrap_rounds}] empty OOB; skipping")
                continue
            print(
                f"[round {r+1}/{bootstrap_rounds}] data: {len(y)} examples, "
                f"train={len(train_idx)} (with dupes), oob={len(val_idx)}"
            )
            best_hav, state = _train_one_cycle(
                train_idx=train_idx,
                val_idx=val_idx,
                log_prefix=f"[r{r+1}] ",
                **common_kwargs,
            )
            oob_havs.append(best_hav)
            if best_hav < best_round_hav:
                best_round_hav = best_hav
                best_round_state = state
                best_round_idx = r
            print(
                f"[round {r+1}/{bootstrap_rounds}] best_oob_haversine_m={best_hav:.2f}"
            )

        if not oob_havs:
            raise RuntimeError("All bootstrap rounds produced empty OOB sets.")

        mean_hav = float(np.mean(oob_havs))
        std_hav = float(np.std(oob_havs))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        torch.save(best_round_state, output_path)
        print(
            f"bootstrap done: B={len(oob_havs)} rounds, "
            f"mean_oob_haversine_m={mean_hav:.2f} \u00b1 {std_hav:.2f}, "
            f"best round={best_round_idx + 1} ({best_round_hav:.2f}m) saved to {output_path}"
        )
        print(
            "RESULT "
            f"mode=bootstrap "
            f"lr={lr:g} epochs={epochs} batch_size={batch_size} weight_decay={weight_decay:g} "
            f"seed={seed} bootstrap_rounds={len(oob_havs)} "
            f"val_haversine_m={mean_hav:.4f} "
            f"val_haversine_m_mean={mean_hav:.4f} "
            f"val_haversine_m_std={std_hav:.4f}"
        )
        return mean_hav

    train_idx, val_idx = location_grouped_split(y, val_fraction=val_fraction, seed=seed)
    print(f"data: {len(y)} examples, train={len(train_idx)}, val={len(val_idx)}")
    best_val_hav, best_state = _train_one_cycle(
        train_idx=train_idx, val_idx=val_idx, **common_kwargs
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    torch.save(best_state, output_path)
    print(f"best val_haversine_m={best_val_hav:.2f}  saved to {output_path}")
    print(
        "RESULT "
        f"mode=split "
        f"lr={lr:g} epochs={epochs} batch_size={batch_size} weight_decay={weight_decay:g} "
        f"seed={seed} bootstrap_rounds=0 "
        f"val_haversine_m={best_val_hav:.4f} "
        f"val_haversine_m_mean={best_val_hav:.4f} "
        f"val_haversine_m_std=0.0000"
    )
    return best_val_hav


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Img2GPS ResNet-18 baseline.")
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "metadata.csv"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "model.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=0,
        help=(
            "If > 0, run B independent training cycles, each with a "
            "location-level bootstrap sample (locations drawn with "
            "replacement; OOB locations form the validation set). The "
            "RESULT line reports mean ± std OOB Haversine across "
            "rounds. Use this for hyperparameter selection on small "
            "datasets — every example is validated across rounds, so "
            "no data is permanently held out."
        ),
    )
    parser.add_argument(
        "--use-all-data",
        action="store_true",
        help=(
            "Train on every example with no val split (mutually "
            "exclusive with --bootstrap-rounds). Use for the final "
            "model after picking HPs via bootstrap."
        ),
    )
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
        bootstrap_rounds=args.bootstrap_rounds,
        use_all_data=args.use_all_data,
    )


if __name__ == "__main__":
    main()
