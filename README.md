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

Put downloaded datasets in `data/raw/` (for example `data/raw/ai4mars-dataset-merged-0.6/`).
