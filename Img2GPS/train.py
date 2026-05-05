"""Train the Img2GPS soft-cluster classifier (Project A).

Pipeline rewrite (vs. the prior ResNet-18 + MSE-on-standardized-coords
baseline). The ResNet-18 regressor was getting ~80 m on the leaderboard
— *worse* than the 48.7 m constant-mean baseline — because:

1. With only ~89 phone images, MSE in standardized space has a strong
   "predict the centroid" attractor. Inspection of held-out predictions
   showed every output collapsing to the training mean.
2. ``RandomHorizontalFlip`` and ``RandomRotation`` destroy the bearing
   cues (which side of the walkway you're on, what's at the horizon)
   that disambiguate GPS coordinates within a 120 x 270 m rectangle.

This rewrite addresses both:

* **K-means cluster head**: the model predicts ``K`` logits
  (``--num-clusters`` flag, default 80), softmaxes them, and outputs
  the weighted sum of ``K`` cluster centers in raw degrees.
  Mean-collapse becomes architecturally impossible — outputs are
  restricted to the convex hull of the cluster centers, and CE loss
  pushes the model to commit to one specific cluster per image. Soft
  mixing across clusters provides sub-cluster precision.

* **Augmentations are photometric only**: ``ColorJitter``, mild
  ``RandomResizedCrop(scale=(0.92, 1.0))``, and a small Gaussian blur.
  No flip, no rotation. Keeps every spatial cue intact.

* **Phone-only training**: the ``--csv`` flag points at
  ``Img2GPS/metadata.csv`` (the spec-collected phone images).

* **Backbone freeze**: only the last MobileNetV3-Small block + the
  classifier head get trained. With ~89 images this prevents the
  pretrained features from being washed away.

* **Hybrid loss**: cross-entropy on the closest-cluster index gives
  stable gradients; a small ``haversine_meters / 1000`` term aligns
  the soft mixture with the eval metric.

Three split modes (only one applies per invocation):

* **default** — single location-grouped train/val split. Saves the
  best-by-val-Haversine checkpoint. Fast; useful for smoke tests.
* **``--bootstrap-rounds B``** — run B independent training cycles,
  each on a location-level bootstrap sample (locations drawn with
  replacement; out-of-bag locations validate). RESULT reports mean ±
  std OOB Haversine. Best for hyperparameter selection on a small
  dataset: every example is validated *across* rounds, so no data is
  permanently held out.
* **``--use-all-data``** — train on every example, no val. Best for
  the final model after picking the best config with bootstrap.

The checkpoint at ``--output`` is the best-OOB-fold model in bootstrap
mode, the final-epoch model in ``--use-all-data`` mode, and the
best-by-val-Haversine model in the default single-split mode.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import Iterable, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

from model import Model, _NUM_CLUSTERS
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
    """In-memory dataset.

    ``targets`` are *cluster indices* (LongTensor of shape (N,)). The raw
    lat/lon labels are kept on ``y`` so the eval loop can compute
    Haversine distance against ground truth.
    """

    def __init__(
        self,
        X_raw: torch.Tensor,
        y: torch.Tensor,
        cluster_idx: torch.Tensor,
        train: bool,
    ) -> None:
        self.X_raw = X_raw
        self.y = y
        self.cluster_idx = cluster_idx
        self.train = train
        # No horizontal flip / rotation: those break left-right and
        # horizon cues that GPS prediction relies on.
        self._train_tx = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.92, 1.0), antialias=True),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=0.2,
                ),
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
        return img, self.cluster_idx[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Location-grouped train/val split
# ---------------------------------------------------------------------------


def _group_by_location(y: torch.Tensor) -> "dict[Tuple[float, float], List[int]]":
    """Group dataset indices by their (rounded) GPS location."""
    groups: dict[Tuple[float, float], List[int]] = {}
    for i, (lat, lon) in enumerate(y.tolist()):
        key = (round(float(lat), 6), round(float(lon), 6))
        groups.setdefault(key, []).append(i)
    return groups


def location_grouped_split(
    y: torch.Tensor, val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Single train/val split where photos sharing a GPS location are
    kept entirely on one side of the split.
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
    """Location-level bootstrap: sample ``len(unique_locations)`` location
    keys WITH REPLACEMENT, take all images at the sampled locations as
    train (with duplicates preserved — a location drawn 3x contributes
    its images 3x to the gradient), and all images at the *unsampled*
    locations as the out-of-bag (OOB) validation set.

    With ~48 unique GPS locations on this dataset, ~37% of locations
    are OOB on average, giving ~15-18 OOB images per round — enough
    signal to estimate val Haversine without permanently holding any
    location out of training across rounds.
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
# K-means clustering on raw GPS labels
# ---------------------------------------------------------------------------


def fit_cluster_centers(
    train_y: torch.Tensor, n_clusters: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run K-means on training (lat, lon) coordinates.

    Returns:
        centers: (K, 2) float tensor of cluster centroids in degrees.
        labels:  (N,) long tensor of cluster assignments for each train sample.

    K-means in raw degrees is fine here because the bbox is tiny
    (~120 x 270 m), so 1 deg lat ≈ 1 deg lon as a Euclidean proxy at
    this latitude. Switching to a meter-scaled space changes centroids
    by <1 m at this scale.

    If we have fewer unique training locations than the requested K,
    sklearn caps the number of distinct clusters and we pad the
    centers tensor by jittering existing centroids by ~1 m so the
    buffer shape (K, 2) still matches the model's classifier head.
    """
    target_k = int(n_clusters)
    coords = train_y.numpy().astype(np.float64)
    fit_k = min(target_k, len(coords))
    km = KMeans(n_clusters=fit_k, random_state=seed, n_init=10)
    labels = km.fit_predict(coords)
    centers = km.cluster_centers_

    if fit_k < target_k:
        pad_count = target_k - fit_k
        rng = np.random.default_rng(seed)
        jitter = rng.normal(scale=1.0e-5, size=(pad_count, 2))
        pad_seed = centers[rng.integers(0, fit_k, size=pad_count)]
        centers = np.vstack([centers, pad_seed + jitter])

    return (
        torch.tensor(centers, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def assign_clusters(y: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Assign each (lat, lon) to its nearest cluster center (Euclidean in deg)."""
    diff = y.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, 2)
    dists = (diff ** 2).sum(dim=-1)
    return dists.argmin(dim=-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _evaluate(
    model: Model,
    loader: Iterable,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    ce_sum = 0.0
    hav_sum = 0.0
    correct = 0
    n = 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, cluster_idx, y_true in loader:
            images = images.to(device)
            cluster_idx = cluster_idx.to(device)
            y_true = y_true.to(device)
            logits = model.predict_logits(images)
            preds = model(images)
            ce_sum += float(ce(logits, cluster_idx).item())
            hav_sum += float(haversine_meters(preds, y_true).sum().item())
            correct += int((logits.argmax(dim=-1) == cluster_idx).sum().item())
            n += images.size(0)
    return ce_sum / max(n, 1), hav_sum / max(n, 1), correct / max(n, 1)


def _freeze_backbone_partial(model: Model) -> None:
    """Train only the last MobileNetV3-Small block + classifier head.

    With ~70 train images this is essential — full fine-tune of even a
    small backbone overfits in 1-2 epochs.
    """
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Unfreeze the last feature block.
    last_block = model.backbone.features[-1]
    for p in last_block.parameters():
        p.requires_grad = True
    for p in model.backbone.classifier.parameters():
        p.requires_grad = True


def _train_one_cycle(
    *,
    X_raw: torch.Tensor,
    y: torch.Tensor,
    train_idx: List[int],
    val_idx: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    haversine_loss_weight: float,
    num_clusters: int,
    seed: int,
    device: torch.device,
    log_prefix: str = "",
) -> Tuple[float, dict]:
    """Run ``epochs`` of training on the given indices.

    Returns the best (lowest) val Haversine in meters and the
    corresponding state_dict (CPU). When ``val_idx`` is empty the
    "best" state is just the final-epoch state and the returned
    Haversine is the training Haversine of the last epoch (used as a
    progress proxy when no validation data is available).
    """
    centers, train_cluster_labels = fit_cluster_centers(
        y[train_idx], n_clusters=num_clusters, seed=seed
    )
    print(
        f"{log_prefix}K-means: {centers.shape[0]} clusters fit on "
        f"{len(train_idx)} train coords"
    )

    train_ds = Img2GPSDataset(
        X_raw[train_idx], y[train_idx], train_cluster_labels, train=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    has_val = len(val_idx) > 0
    if has_val:
        val_cluster_labels = assign_clusters(y[val_idx], centers)
        val_ds = Img2GPSDataset(
            X_raw[val_idx], y[val_idx], val_cluster_labels, train=False
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None

    model = Model(weights_path=None, num_clusters=num_clusters).to(device)
    model.set_cluster_centers(centers.tolist())
    _freeze_backbone_partial(model)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_metric = math.inf
    best_state: dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_ce = 0.0
        total_hav = 0.0
        correct = 0
        n_seen = 0
        for images, cluster_idx, y_true in train_loader:
            images = images.to(device)
            cluster_idx = cluster_idx.to(device)
            y_true = y_true.to(device)

            optimizer.zero_grad()
            logits = model.predict_logits(images)
            ce = ce_loss(logits, cluster_idx)
            # Soft prediction is the *exact* inference path (softmax @ centers),
            # so optimizing Haversine on it directly aligns training with eval.
            soft_pred = torch.softmax(logits, dim=-1) @ model.cluster_centers
            hav = haversine_meters(soft_pred, y_true).mean()
            loss = ce + haversine_loss_weight * (hav / 1000.0)
            loss.backward()
            optimizer.step()

            total_ce += float(ce.item()) * images.size(0)
            total_hav += float(hav.item()) * images.size(0)
            correct += int((logits.argmax(dim=-1) == cluster_idx).sum().item())
            n_seen += images.size(0)

        scheduler.step()

        train_ce = total_ce / max(n_seen, 1)
        train_hav = total_hav / max(n_seen, 1)
        train_acc = correct / max(n_seen, 1)

        if has_val:
            val_ce, val_hav, val_acc = _evaluate(model, val_loader, device)
            print(
                f"{log_prefix}epoch {epoch:02d}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"train_ce={train_ce:.4f}  train_acc={train_acc:.3f}  train_hav_m={train_hav:.1f}  "
                f"val_ce={val_ce:.4f}  val_acc={val_acc:.3f}  val_hav_m={val_hav:.1f}"
            )
            metric = val_hav
        else:
            print(
                f"{log_prefix}epoch {epoch:02d}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"train_ce={train_ce:.4f}  train_acc={train_acc:.3f}  train_hav_m={train_hav:.1f}  "
                f"(no val: training on all data)"
            )
            metric = train_hav

        if metric < best_metric:
            best_metric = metric
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if not best_state:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
    haversine_loss_weight: float,
    num_clusters: int = _NUM_CLUSTERS,
    bootstrap_rounds: int = 0,
    use_all_data: bool = False,
) -> float:
    """Top-level dispatcher across split modes.

    * ``use_all_data=True``      -> one cycle on every example, no val
    * ``bootstrap_rounds > 0``   -> B cycles, location-level bootstrap,
                                     OOB validation; saves the model
                                     trained on the best-OOB-fold round
    * default                    -> one cycle with location-grouped
                                     held-out val (current behavior)
    """
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
        y=y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        haversine_loss_weight=haversine_loss_weight,
        num_clusters=num_clusters,
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
        # No val available — leaderboard val_haversine_m field is left
        # at the train Haversine so callers don't have to special-case
        # the missing field. The companion ``mode`` field disambiguates.
        print(
            "RESULT "
            f"mode=all_data "
            f"k={num_clusters} lr={lr:g} hav_w={haversine_loss_weight:g} "
            f"epochs={epochs} batch_size={batch_size} weight_decay={weight_decay:g} "
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
                # Extremely unlikely with ~48 unique locations: every key
                # was sampled at least once. Skip this round to avoid a
                # zero-OOB metric that would bias the mean downward.
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
            f"mean_oob_haversine_m={mean_hav:.2f} ± {std_hav:.2f}, "
            f"best round={best_round_idx + 1} ({best_round_hav:.2f}m) saved to {output_path}"
        )
        print(
            "RESULT "
            f"mode=bootstrap "
            f"k={num_clusters} lr={lr:g} hav_w={haversine_loss_weight:g} "
            f"epochs={epochs} batch_size={batch_size} weight_decay={weight_decay:g} "
            f"seed={seed} bootstrap_rounds={len(oob_havs)} "
            f"val_haversine_m={mean_hav:.4f} "
            f"val_haversine_m_mean={mean_hav:.4f} "
            f"val_haversine_m_std={std_hav:.4f}"
        )
        return mean_hav

    # Default: single location-grouped split.
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
        f"k={num_clusters} lr={lr:g} hav_w={haversine_loss_weight:g} "
        f"epochs={epochs} batch_size={batch_size} weight_decay={weight_decay:g} "
        f"seed={seed} bootstrap_rounds=0 "
        f"val_haversine_m={best_val_hav:.4f} "
        f"val_haversine_m_mean={best_val_hav:.4f} "
        f"val_haversine_m_std=0.0000"
    )
    return best_val_hav


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Img2GPS cluster classifier.")
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "metadata.csv"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "model.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=_NUM_CLUSTERS,
        help=(
            "K, the number of K-means location clusters. The classifier "
            "head is sized to K, and ``cluster_centers`` is a (K, 2) "
            "buffer overwritten with the K-means centroids of the "
            "training split."
        ),
    )
    parser.add_argument(
        "--haversine-loss-weight",
        type=float,
        default=0.5,
        help=(
            "Weight on the (haversine_m / 1000) auxiliary loss term. "
            "Larger K (instance-retrieval regime) makes CE per-class "
            "signal noisier and benefits from a larger Haversine term."
        ),
    )
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
            "model after picking the best config via bootstrap."
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
        args.haversine_loss_weight,
        args.num_clusters,
        bootstrap_rounds=args.bootstrap_rounds,
        use_all_data=args.use_all_data,
    )


if __name__ == "__main__":
    main()
