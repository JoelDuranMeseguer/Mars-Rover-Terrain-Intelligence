# Mars Rover Terrain Intelligence

Mars Rover Terrain Intelligence is a computer vision project built around a simple but realistic idea:

**rover image -> terrain segmentation -> traversability cost map -> local path recommendation**

The goal is not to claim full autonomous navigation from a single image.  
The goal is to build a clean, defendable MVP for **local terrain-aware decision support** from rover imagery.

## Current MVP scope

The current version focuses on a local planning pipeline:

1. **semantic segmentation** of rover terrain,
2. **class-to-cost mapping** to turn terrain labels into traversability costs,
3. **local A\*** path planning on the resulting 2D cost grid.

### What this MVP does

- predicts terrain classes from rover images,
- converts those classes into an interpretable cost map,
- proposes a local low-cost path on that map.

### What this MVP does not do

- global navigation,
- physically grounded 3D planning,
- BEV projection,
- dynamic obstacle handling,
- closed-loop rover control.

That limitation is intentional.  
This project is currently framed as **local route recommendation**, not full autonomy.

## Current pipeline

```text
RGB rover image
   ↓
Segmentation model (CNN / U-Net baseline)
   ↓
Predicted terrain classes
   ↓
Class-to-cost mapping
   ↓
Local traversability cost map
   ↓
A* path recommendation on a local 2D grid
```

## Repository layout

```text
mars-rover-terrain-intelligence/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── Makefile
├── requirements.txt
├── .pre-commit-config.yaml
├── configs/
├── data/
├── notebooks/
├── models/
├── reports/
├── app/
├── src/mrti/
├── scripts/
└── tests/
```

## Quick start

From the repository root:

```bash
python -m pip install -e .
pytest -q
```

If you want to launch the app:

```bash
streamlit run app/Home.py
```

## Demo outputs

After training and saving a best checkpoint, you can generate a few qualitative outputs from the pipeline.

### 1) Image | ground-truth mask | predicted mask

```bash
python scripts/inspect_predictions.py \
  --checkpoint checkpoints/baseline_best.pt \
  --model unet \
  --num-samples 4
```

### 2) Image | predicted mask | terrain cost map

```bash
python scripts/build_cost_map.py \
  --checkpoint checkpoints/baseline_best.pt \
  --model unet \
  --num-samples 4
```

### 3) Image | local cost map | local A* path recommendation

```bash
python scripts/plan_path.py \
  --checkpoint checkpoints/baseline_best.pt \
  --model unet \
  --sample-idx 0
```

Default output folders:

- `outputs/predictions/`
- `outputs/cost_maps/`
- `outputs/path_plans/`

### 4) README-friendly end-to-end panel (recommended for GitHub)

```bash
python scripts/make_demo_panel.py \
  --checkpoint checkpoints/baseline_c3_mid_final.pt \
  --model unet \
  --sample-idx 0 \
  --class3-threshold 0.6 \
  --path-thickness 5 \
  --safety-radius 1 \
  --readme-mode \
  --output-dir docs/assets
```

Recommended filename for GitHub embedding: `docs/assets/readme_demo_panel.png`.

Use this reference in `README.md`:

```markdown
![End-to-end demo: rover image, segmentation, cost map, and local A* path](docs/assets/readme_demo_panel.png)
```

This remains a **local path recommendation** demo, not autonomous navigation.

## Current limitations

This is still an MVP, so a few limitations matter:

- Planning is **local** and **image-plane based**.
- The cost map currently uses **fixed hand-defined class costs**.
- Path quality depends heavily on segmentation quality.
- Start/goal placement and the local planning mask are still heuristic.
- The system does **not** yet use explicit depth or geometric grounding.

## Why this scope is still useful

Even with those limitations, the current pipeline already demonstrates a real robotics pattern:

- perception from vision,
- semantic interpretation of terrain,
- conversion to traversability cost,
- classical path planning on top of that representation.

That makes it a good intermediate step between pure computer vision and full rover autonomy.

## Next steps

Planned improvements include:

- stronger segmentation baselines and better validation,
- more robust traversability estimation,
- geometric grounding with depth or pseudo-BEV,
- smoother and more realistic local planning,
- eventual simulation-oriented evaluation.

## Data placement

- Put downloaded datasets under `data/raw/`  
  for example: `data/raw/AI4Mars/`
- Do **not** commit the full dataset to GitHub.
- Keep only lightweight metadata or manifests in the repo when useful.

## AI4Mars audit and preparation

### 1) Audit the raw dataset tree and export manifests

```bash
python scripts/audit_ai4mars.py \
  --root data/raw/AI4Mars \
  --out data/processed/manifests
```

This generates `msl_ncam_v1.csv` plus inventory and summary files.

### 2) Build the clean training-ready subset (`msl_ncam_v1`)

```bash
python scripts/prepare_data.py \
  --raw-root data/raw/AI4Mars \
  --manifest data/processed/manifests/msl_ncam_v1.csv \
  --out-dir data/processed/msl_ncam_v1 \
  --seed 42
```

Expected output layout:

```text
data/processed/msl_ncam_v1/
├── images/
├── masks/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── index.csv
```

`prepare_data.py` validates image/mask pairs, keeps only valid samples, renames files by sample id, creates train/val/test splits, and writes `index.csv`.

## PyTorch dataset layer

Use `AI4MarsSegmentationDataset` to load one split from the prepared dataset folder:

```python
from mrti.data import AI4MarsSegmentationDataset

train_ds = AI4MarsSegmentationDataset(
    dataset_root="data/processed/msl_ncam_v1",
    split="train",
)

sample = train_ds[0]
print(sample["id"], sample["image"].shape, sample["mask"].shape)
```

Quick CLI sanity check:

```bash
python scripts/check_dataset.py \
  --data-root data/processed/msl_ncam_v1 \
  --split train \
  --num-samples 3
```

## Troubleshooting: `No module named "mrti"`

This repo uses a `src/` layout, so the package lives under `src/mrti`.

If you run scripts without installing the project first, Python may not find `mrti`.

Recommended steps:

1. Create and activate a virtual environment from the repo root.
2. Install the project in editable mode:

   ```bash
   python -m pip install -e .
   ```

3. Verify the import:

   ```bash
   python -c "import mrti; print(mrti.__file__)"
   ```

For `scripts/check_dataset.py`, there is also a small fallback that adds `src/` to `sys.path` when the script is run directly.

## Project status

Current status: **working MVP**

The project already supports:

- dataset preparation and sanity checks,
- segmentation training and checkpointing,
- qualitative prediction inspection,
- terrain cost map generation,
- local A* path recommendation from image-derived terrain understanding.

The next stage is to make the interpretation and planning blocks more realistic, without overclaiming what a single-image system can do.
