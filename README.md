# Mars Rover Terrain Intelligence

A clean project template for terrain segmentation, traversability analysis, and path planning.

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

