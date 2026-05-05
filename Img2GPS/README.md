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
├── metadata.csv          # image_path, latitude, longitude (phone captures)
├── model.pt              # trained checkpoint (state_dict)
├── model.py              # MobileNetV3-Small + soft K-cluster head
├── preprocess.py         # 224x224 + ImageNet norm, tensor cache
├── train.py              # training loop (CE + Haversine, cosine LR, location split, aug)
├── notebooks/
│   └── img2gps.ipynb     # walkthrough: data → train → eval → vis
└── reference/            # course-provided sanity-check examples
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

## Training data

Training uses only the phone-captured images listed in
`Img2GPS/metadata.csv`. To rebuild that CSV from scratch (e.g. after
adding new HEIC files under `data/images/`), run
`python Img2GPS/extract_exif.py`. The course-staged test set lives in
the same ~150 × 110 m Penn rectangle, so the spec sampling guidance
already produces in-domain training data — additional sources tend to
hurt more than help on a region this small.
