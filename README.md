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
