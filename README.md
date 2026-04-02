# Mars Rover Terrain Intelligence

Mars Rover Terrain Intelligence is a lightweight starter repository for building terrain understanding and autonomous navigation workflows from rover imagery.

## Project goals
- Load and manage rover image/label datasets.
- Train terrain segmentation models.
- Run inference on new terrain images.
- Estimate traversability scores from model outputs.
- Plan safe paths across segmented terrain.
- Visualize intermediate and final results.

## Project structure
```text
src/
  data/            # Dataset loading and dataset utilities
  features/        # Preprocessing and feature engineering
  models/          # Training, inference, and evaluation logic
  navigation/      # Traversability and path-planning components
  visualization/   # Plotting and rendering helpers
  utils/           # Shared utilities
notebooks/         # EDA and experimentation notebooks
app/               # Streamlit app scaffold
tests/             # Placeholder unit/integration tests
configs/           # YAML config files
data/
  raw/             # Raw source data
  interim/         # Intermediate outputs
  processed/       # Model-ready processed datasets
models/            # Saved model artifacts
reports/figures/   # Generated figures and plots
```

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run placeholder tests:
   ```bash
   pytest -q
   ```
4. Launch the Streamlit scaffold:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Typical workflow
- Update `configs/base.yaml` for paths and training settings.
- Explore data in `notebooks/01_eda.ipynb`.
- Train model with `python scripts/train_segmentation.py --config configs/base.yaml`.
- Run inference with `python scripts/run_inference.py --config configs/base.yaml --image path/to/image.png`.

## Notes
This scaffold intentionally keeps implementations simple so you can extend each module iteratively.
