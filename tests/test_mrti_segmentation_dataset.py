"""Tests for AI4MarsSegmentationDataset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mrti.data.dataset import AI4MarsSegmentationDataset


torch = pytest.importorskip("torch")


def _write_dummy_pair(images_dir: Path, masks_dir: Path, sample_id: str) -> tuple[str, str]:
    image_name = f"{sample_id}.png"
    mask_name = f"{sample_id}.png"

    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 64
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 2

    Image.fromarray(image).save(images_dir / image_name)
    Image.fromarray(mask).save(masks_dir / mask_name)

    return f"images/{image_name}", f"masks/{mask_name}"


def test_dataset_reads_split_and_returns_tensors(tmp_path: Path) -> None:
    root = tmp_path / "msl_ncam_v1"
    images_dir = root / "images"
    masks_dir = root / "masks"
    splits_dir = root / "splits"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    row_a = _write_dummy_pair(images_dir, masks_dir, "A")
    row_b = _write_dummy_pair(images_dir, masks_dir, "B")

    with (root / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerow({"id": "A", "image_path": row_a[0], "mask_path": row_a[1]})
        writer.writerow({"id": "B", "image_path": row_b[0], "mask_path": row_b[1]})

    (splits_dir / "train.txt").write_text("B\n", encoding="utf-8")

    ds = AI4MarsSegmentationDataset(dataset_root=root, split="train")
    sample = ds[0]

    assert len(ds) == 1
    assert sample["id"] == "B"
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["image"].shape == (3, 8, 10)
    assert sample["mask"].shape == (8, 10)
    assert sample["mask"].dtype == torch.int64
