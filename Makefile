PYTHON ?= python
CONFIG ?= configs/base.yaml

.PHONY: install test train infer app

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	pytest -q

train:
	$(PYTHON) scripts/train_segmentation.py --config $(CONFIG)

infer:
	$(PYTHON) scripts/run_inference.py --config $(CONFIG) --image data/raw/example.png

app:
	streamlit run app/streamlit_app.py
