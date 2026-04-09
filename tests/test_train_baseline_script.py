"""Smoke test for scripts/train_baseline.py."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")


def _write_pair(root: Path, sample_id: str, value: int) -> tuple[str, str]:
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    mask = np.full((8, 8), value, dtype=np.uint8)

    Image.fromarray(image).save(images_dir / f"{sample_id}.png")
    Image.fromarray(mask).save(masks_dir / f"{sample_id}.png")
    return f"images/{sample_id}.png", f"masks/{sample_id}.png"


def _build_dataset(root: Path) -> None:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        ("A",) + _write_pair(root, "A", 0),
        ("B",) + _write_pair(root, "B", 1),
        ("C",) + _write_pair(root, "C", 1),
    ]

    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "mask_path"])
        writer.writeheader()
        for sample_id, image_path, mask_path in rows:
            writer.writerow({"id": sample_id, "image_path": image_path, "mask_path": mask_path})

    (split_dir / "train.txt").write_text("A\nB\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("C\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("A\n", encoding="utf-8")


def test_train_baseline_runs_one_step(tmp_path: Path) -> None:
    dataset_root = tmp_path / "msl_ncam_v1"
    _build_dataset(dataset_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline.py",
            "--dataset-root",
            str(dataset_root),
            "--num-classes",
            "2",
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--max-train-batches",
            "1",
            "--max-val-batches",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "train batch 1" in result.stdout
    assert "val batch 1" in result.stdout
    assert "summary" in result.stdout
