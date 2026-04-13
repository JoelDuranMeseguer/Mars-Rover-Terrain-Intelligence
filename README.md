# Mars Rover Terrain Intelligence

A clean project template for terrain segmentation, traversability analysis, and path planning.

## Problem and current MVP scope

This project explores how to go from a single rover image to a **local route recommendation**:

1. semantic segmentation (terrain understanding),
2. class-to-cost mapping (traversability interpretation),
3. local A* path suggestion on the resulting grid.

Current scope is intentionally limited to **decision support**, not full autonomy.

- ✅ What this MVP does:
  - produce segmentation masks,
  - build an interpretable terrain cost map,
  - generate a local 2D A* path recommendation.
- ❌ What this MVP does not do (yet):
  - global navigation,
  - 3D geometry / BEV projection,
  - dynamic obstacle handling,
  - rover control/actuation in closed loop.

## Current pipeline (perception -> interpretation -> decision support)

```text
RGB image
   ↓
Segmentation model (CNN / UNet baseline)
   ↓
Predicted terrain classes
   ↓
Class-cost mapping
   ↓
Local cost map
   ↓
A* on 2D grid (local recommendation)
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

```bash
python -m pip install -e .
pytest -q
streamlit run app/Home.py
```

## Demo outputs (MVP)

After training and saving a best checkpoint, you can generate qualitative outputs:

```bash
# 1) Image | GT mask | Predicted mask
python scripts/inspect_predictions.py \
  --checkpoint checkpoints/baseline_best.pt \
  --model unet \
  --num-samples 4

# 2) Image | Predicted mask | Cost map
python scripts/build_cost_map.py \
  --checkpoint checkpoints/baseline_best.pt \
  --model unet \
  --num-samples 4

# 3) Image | Cost map | Cost map + local A* path
python scripts/plan_path.py \
  --checkpoint checkpoints/baseline_best.pt \
  --model unet \
  --sample-idx 0
```

Default output folders:

- `outputs/predictions/`
- `outputs/cost_maps/`
- `outputs/path_plans/`

## Limitations (honest status)

- Planning is image-plane and local; it is not physically grounded 3D navigation.
- Cost mapping currently uses fixed, hand-defined class costs.
- Path quality depends directly on segmentation quality and may degrade under distribution shift.
- Start/goal and local planning mask are heuristic choices for MVP demonstration.

## Next steps

- Improve segmentation robustness (more data, stronger validation, better calibration).
- Replace fixed class costs with data-informed or learned traversability estimates.
- Add geometric grounding (depth/BEV) before making stronger navigation claims.
- Move from static recommendation to closed-loop evaluation in simulation.

## Data placement

- Put downloaded datasets under `data/raw/` (for example `data/raw/AI4Mars/`).
- Do not commit the full dataset to GitHub: it is large and usually has licensing/distribution constraints.
- Keep only lightweight metadata/manifests in the repo when needed.

## AI4Mars audit and preparation

### 1) Audit raw tree and export manifests

```bash
python scripts/audit_ai4mars.py \
  --root data/raw/AI4Mars \
  --out data/processed/manifests
```

This generates `msl_ncam_v1.csv` plus inventory and summary files.

### 2) Build clean training-ready subset (`msl_ncam_v1`)

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

`prepare_data.py` validates image/mask existence, copies only valid pairs, renames by sample id, creates 80/10/10 splits, and writes `index.csv`.

## PyTorch dataset layer (index.csv + splits)

Use `AI4MarsSegmentationDataset` to load one split from the prepared folder:

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
python scripts/check_dataset.py --data-root data/processed/msl_ncam_v1 --split train --num-samples 3
```


## Error común: `No module named "mrti"`

Este repo usa estructura `src/`, así que `mrti` vive dentro de `src/mrti`.
Si ejecutas scripts sin instalar el proyecto, Python no siempre encuentra ese paquete.

Pasos recomendados:

1. Desde la raíz del repo, crea/activa tu entorno virtual.
2. Instala el proyecto en editable:
   ```bash
   python -m pip install -e .
   ```
3. Verifica import:
   ```bash
   python -c "import mrti; print(mrti.__file__)"
   ```

Para el script `scripts/check_dataset.py` también se añadió un fallback pequeño que agrega `src/` al `sys.path` cuando lo ejecutas directamente.
