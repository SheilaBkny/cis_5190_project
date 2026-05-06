# Project A — Img2GPS

Predict GPS coordinates from a campus image. Metric: **mean Haversine distance (meters)** — lower is better.

## Layout

```
Img2GPS/
├── README.md
├── eval_project_a.py      # course-style local evaluator
├── dinov2_vit.py          # vendored DINOv2 ViT-S/14 (checkpoint-compatible)
├── extract_exif.py        # optional: build metadata.csv from camera files
├── metadata.csv           # image_path, latitude, longitude
├── model.pt               # frozen DINOv2 + gallery embeddings + GPS + temperature
├── model.py               # frozen DINOv2 + softmax retrieval head
├── preprocess.py        # 224×224, ImageNet norm, tensor cache
├── train.py               # build gallery, Haversine-tune temperature T
└── reference/             # course sanity-check CSV + images
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Append more photos (e.g. `data/sheila/*.HEIC`)

On **macOS**, HEIC GPS is read via Spotlight (`mdls`) when Pillow cannot; HEIC is converted to `data/images_converted/<stem>.png` with `qlmanage`.

```bash
python Img2GPS/extract_exif.py --ingest data/sheila
```

Merges into `Img2GPS/metadata.csv` (dedupes by `image_path`). Use `--ingest-only` to replace the entire CSV with only that folder.

## Train (submission checkpoint)

```bash
python Img2GPS/train.py --csv Img2GPS/metadata.csv --use-all-data --output Img2GPS/model.pt
```

Optional CV: `--bootstrap-rounds 25`. Temperature **T** is optimized with **mean Haversine** on a location-grouped holdout (not MSE). With `--use-all-data`, every photo stays in the gallery; **T** still uses a loc-holdout slice (see `--val-fraction`). Flag `--temp-steps` (default **800**) controls Adam steps for **T**.

First run may download DINOv2 ViT-S/14 weights (~85 MB) via `torch.hub`.

## Eval

```bash
python Img2GPS/eval_project_a.py \
    --model Img2GPS/model.py \
    --preprocess Img2GPS/preprocess.py \
    --weights Img2GPS/model.pt \
    --csv Img2GPS/reference/metadata.csv
```

## Notes

* **No target MSE** in `train.py`: the only learned scalar is **T**, via differentiable Haversine.
* **Location-grouped splits** keep all photos at the same rounded GPS on one side of a holdout.
* **Cache**: `preprocess.py` uses `<csv_dir>/.cache/<stem>.{raw,norm}.pt` gated on CSV mtime.
