PYTHON ?= python
CONFIG ?= configs/base.yaml

.PHONY: install test prepare train eval infer demo

install:
	$(PYTHON) -m pip install -e .

test:
	pytest -q

prepare:
	$(PYTHON) scripts/prepare_data.py --raw-root data/raw/AI4Mars --manifest data/processed/manifests/msl_ncam_v1.csv --out-dir data/processed/msl_ncam_v1 --seed 42

train:
	$(PYTHON) scripts/train.py --config configs/train_deeplab.yaml

eval:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

infer:
	$(PYTHON) scripts/run_inference.py --config $(CONFIG) --image data/raw/example.png

demo:
	$(PYTHON) scripts/launch_demo.py
