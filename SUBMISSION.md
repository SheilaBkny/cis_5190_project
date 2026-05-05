# CIS 5190 Final Project — Submission Guidelines

Distilled from `Project_submission.pdf` (course-released April 9, 2026)
and the staff resources bundle (`project-resources.zip`). Use this as the
single source of truth when packaging a leaderboard submission.

---

## 1. Submission overview

- The backend lives on Hugging Face. Two leaderboards:
  - **Project A — Img2GPS**
  - **Project B — News Headline Classifier**
- Each submission must include:
  - `model.py` — model implementation, importable and instantiable
  - `preprocess.py` — preprocessing function(s)
  - `model.pt` _(optional, but required if the model needs trained weights)_
- Each submission also requires a Group ID and Alias string.
- A 5-page report and a public Hugging Face Dataset for the collected data
  are due **May 6, 2026** on Gradescope (links must be in the report).

## 2. Backend environment

The backend has these libraries available:

```
numpy, pandas, torch==2.9.1, torchvision, scikit-learn, opencv-python
```

If you need anything else, post on Ed first. PyTorch is strongly preferred.

## 3. Data contracts

### 3.1 Project A — Img2GPS

**Input CSV** (similar to `Img2GPS/reference/metadata.csv`):

- Image path column — one of: `image_path`, `filepath`, `image`, `path`, `file_name`
- Latitude column — one of: `Latitude`, `latitude`, `lat`
- Longitude column — one of: `Longitude`, `longitude`, `lon`

**`preprocess.py` must expose:**

```python
def prepare_data(csv_path: str) -> (X, y)
```

- `X` — sequence/array/tensor of inputs that will be fed to
  `model.predict(batch)` or `model(batch)`.
- `y` — sequence/array/tensor of `[lat, lon]` pairs in **raw degrees**
  (not normalized).
- If the model uses target normalization, the stats **must be hard-coded
  in `model.py`** — not derived from the CSV at runtime.

**`model.py` must expose either:**

- `get_model() -> model_instance`, **or**
- a class named `Model` or `IMG2GPS` that is instantiable with no
  arguments.

At inference the backend calls `model.predict(batch)` if it exists,
otherwise `model(batch)`. Outputs must be `[lat, lon]` in **raw degrees**.

**`model.pt` (optional):**

- Loaded via `torch.load(..., map_location="cpu")` and a robust
  `load_state_dict` routine — checkpoint keys must match the model's
  parameter names (or normalize to them).

### 3.2 Project B — News Headline Classifier

**Input CSV** similar to `url_data_only.csv`.

**`preprocess.py` must expose:**

```python
def prepare_data(csv_path: str) -> (X, y)
```

- `X` — sequence/array/tensor of inputs suitable for the model.
- `y` — sequence of labels (strings or integer class ids).

**`model.py` must expose either:**

- `get_model() -> model_instance`, **or**
- a class named `Model` or `NewsClassifier` instantiable with no arguments.

Backend calls `model.predict(batch)` when available; otherwise
`model(batch)` and falls back to `argmax` over the final dimension if a
logits tensor is returned.

## 4. Evaluation metrics

### 4.1 Project A

**Average Haversine distance in meters** (lower is better):

\[
 d(a, b) = 2R \arcsin\!\sqrt{\sin^2\!\tfrac{\Delta\phi}{2} + \cos\phi_1 \cos\phi_2 \sin^2\!\tfrac{\Delta\lambda}{2}}
\]

with \(R = 6\,371\,000\) m. The leaderboard reports
\(\frac{1}{N} \sum_i d(y_i, \hat y_i)\) over the hidden test set.

### 4.2 Project B

**Accuracy.** If predictions are integer ids and labels are strings (or
vice versa), the backend applies a robust 2-class mapping; otherwise it
compares as strings.

## 5. End-to-end backend flow (per project)

1. Read CSV (e.g. `test/metadata.csv`).
2. `preprocess.prepare_data(csv)` → `(X, y)`.
3. Instantiate `model` from `model.py`.
4. Load `model.pt` if provided.
5. Run batched inference via `model.predict(batch)` or `model(batch)`.
6. Compare predictions to ground truth (raw lat/lon for A, labels for B).
7. Write JSON results and update the leaderboard.

## 6. Local sanity checks (strongly recommended)

Run the staff evaluators before every submission:

```bash
# Project A
python Img2GPS/eval_project_a.py \
    --model      Img2GPS/model.py \
    --preprocess Img2GPS/preprocess.py \
    --weights    Img2GPS/model.pt \
    --csv        Img2GPS/reference/metadata.csv

# Project B (when on the Project B branch)
python eval_project_b.py \
    --model      model.py \
    --preprocess preprocess.py \
    --weights    model.pt \
    --csv        url_data_only.csv
```

## 7. Packaging & gotchas (from the spec)

- Stick to the dependency list in §2 — extra libraries require Ed approval.
- `prepare_data` and the model entry points must match §3 contracts
  exactly.
- If you ship `model.pt`, ensure its keys match your model parameters.
- **Project A outputs must be in degrees** and will be compared against
  raw labels in the CSV.
- Include the Hugging Face Dataset link in the report by the deadline.

## 8. Compliance status — `iter1` (Project A)

| Spec requirement | Status in this branch |
|---|---|
| `prepare_data(csv_path) -> (X, y)` | ✅ `Img2GPS/preprocess.py` |
| Image-path column aliases (`image_path`, `filepath`, `image`, `path`, `file_name`) | ✅ all accepted (`_resolve_column`) |
| Lat / lon column aliases | ✅ all accepted |
| `y` returned in raw degrees | ✅ |
| `Model` / `IMG2GPS` class instantiable with no args | ✅ both present |
| `get_model()` factory | ✅ |
| `model.predict(batch)` returns `[lat, lon]` degrees | ✅ |
| `model(batch)` returns `[lat, lon]` degrees | ✅ |
| Target normalization stats hard-coded in `model.py` | ✅ `_TARGET_MEAN`, `_TARGET_STD` literals |
| `model.pt` loadable via `torch.load` + `load_state_dict` (strict=False) | ✅ keys match |
| Backend dependencies (`torch`, `torchvision`, `numpy`, `pandas`, `opencv-python`, `scikit-learn`) | ✅ pinned in `requirements.txt` |
| Runs cleanly through `Img2GPS/eval_project_a.py` | ✅ |

**Files to send for the Project A submission:**

```
Img2GPS/model.py
Img2GPS/preprocess.py
Img2GPS/model.pt
```
