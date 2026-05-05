# Project A — Img2GPS

Predict GPS coordinates from a camera image taken on Penn's campus
(spec test region: 33rd & Walnut → 34th & Spruce). Metric: average
Haversine distance in meters (lower is better).

## Layout

```
Img2GPS/
├── README.md
├── eval_project_a.py     # course-style evaluator (do not edit)
├── extract_exif.py       # builds metadata.csv from data/images/*.HEIC
├── metadata.csv          # image_path, latitude, longitude
├── model.pt              # trained checkpoint (state_dict)
├── model.py              # ResNet-18 with target-normalization buffers
├── preprocess.py         # 224x224 + ImageNet norm, tensor cache
├── train.py              # training loop (lr=1e-3, StepLR, location split, aug)
├── notebooks/
│   └── img2gps.ipynb     # walkthrough: data → train → eval → vis
├── reference/            # course-provided sanity-check examples
└── scripts/
    └── fetch_mapillary.py  # optional: pull extra training images from Mapillary
```

Source images live under `../data/images` (HEIC) and
`../data/images_converted` (PNG produced by `extract_exif.py`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

## End-to-end commands

```bash
python Img2GPS/extract_exif.py                               # build metadata.csv
python Img2GPS/train.py --epochs 12 --lr 1e-3                # train -> Img2GPS/model.pt
python Img2GPS/eval_project_a.py \
    --model      Img2GPS/model.py \
    --preprocess Img2GPS/preprocess.py \
    --weights    Img2GPS/model.pt \
    --csv        Img2GPS/metadata.csv
```

The training script prints per-epoch `train_norm_mse`, `val_mse_deg2`, and
`val_haversine_m`, and saves the best-by-Haversine state dict.

## Implementation notes

* **Targets are standardized** during training (per spec §2.3) using the
  train-split mean/std. The mean/std are saved as buffers on the model so
  `model.pt` is self-contained — `forward()` always returns degrees.
* **Location-grouped split**: photos sharing exact GPS coordinates (the
  spec describes ~8 photos per spot) are kept entirely on one side of
  the train/val split.
* **Augmentation** at training time: `RandomResizedCrop(scale=(0.85,1.0))`,
  horizontal flip, mild color jitter.
* **Optimizer**: Adam, `lr=1e-3`, weight decay `1e-4`. **Scheduler**:
  `StepLR(step_size=4, gamma=0.5)`. **Epochs**: 12 (within the spec's
  10–15 range).
* **Caching**: `preprocess.py` writes `<csv_dir>/.cache/<csv_stem>.{raw,norm}.pt`
  and reuses them while the CSV file mtime hasn't advanced.

## Extra training data (optional)

Walking the test region by hand is the cleanest source, but the spec
test set lives in a tiny ~150 × 110 m rectangle — `scripts/fetch_mapillary.py`
can pull additional georeferenced street-level images of the same area
from [Mapillary](https://www.mapillary.com/) (CC-BY-SA, ML use is
explicitly permitted by their ToU §12). **Do not** use Google Street
View: §3(c)(vii) of the Maps Platform terms forbids using its content
to "train, test, validate or fine-tune" ML models.

```bash
export MAPILLARY_TOKEN="MLY|<your client token>"
python Img2GPS/scripts/fetch_mapillary.py probe \
    --bbox=-75.19297,39.95026,-75.18949,39.95291
python Img2GPS/scripts/fetch_mapillary.py download \
    --bbox=-75.19297,39.95026,-75.18949,39.95291 \
    --out=data/mapillary --limit=200 --skip-pano
```

The download writes a `metadata_mapillary.csv` in the same
`image_path,latitude,longitude` schema as `metadata.csv`, so you can
either point `train.py --csv` at it directly or concatenate the two.
Caveat: Mapillary photos are typically captured from car/bike rigs and
are a noticeable domain shift from a phone held upright on walkways —
mix them with your own photos rather than replacing them.
