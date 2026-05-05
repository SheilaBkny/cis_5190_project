# Project A — Img2GPS submission

This branch is the **leaderboard submission package** for Project A. It
intentionally contains only the three files that the Hugging Face
backend needs:

| File | Purpose |
|---|---|
| `model.py` | ResNet-18 regressor, exposes `Model`, `IMG2GPS`, `get_model()`; target normalization stats are hard-coded. |
| `preprocess.py` | `prepare_data(csv_path) -> (X, y)` returning ImageNet-normalized images and raw lat/lon degrees. |
| `model.pt` | Trained state_dict (best-by-val-Haversine snapshot from the iter1 training run). |

For the development pipeline (data, training script, notebook,
augmentation, metric tracking, etc.) check out the [`iter1`](
https://github.com/SheilaBkny/cis_5190_project/tree/iter1) branch.

## Backend contract this submission satisfies

- `prepare_data(csv_path) -> (X, y)` with `y` in **raw degrees**.
- Image-path column aliases: `image_path`, `filepath`, `image`, `path`,
  `file_name`.
- Lat aliases: `Latitude`, `latitude`, `lat`. Lon aliases: `Longitude`,
  `longitude`, `lon`.
- `Model()` and `IMG2GPS()` instantiate with no arguments and return a
  `torch.nn.Module`. `get_model()` is also provided.
- `model.predict(batch)` and `model(batch)` both return `(B, 2)`
  `[lat, lon]` pairs in raw degrees.
- `model.pt` keys match the model parameters and load via
  `torch.load(..., map_location="cpu")` + `load_state_dict(strict=False)`.

## Local sanity check

```bash
# from the iter1 branch (which still has the evaluator + reference set)
python Img2GPS/eval_project_a.py \
    --model      <path-to-this-branch>/model.py \
    --preprocess <path-to-this-branch>/preprocess.py \
    --weights    <path-to-this-branch>/model.pt \
    --csv        Img2GPS/reference/metadata.csv
```
