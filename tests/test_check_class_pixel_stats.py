"""Tests for scripts/check_class_pixel_stats.py."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _write_pair(root: Path, sample_id: str, mask: np.ndarray) -> tuple[str, str]:
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    Image.fromarray(image).save(images_dir / f"{sample_id}.png")
    Image.fromarray(mask.astype(np.uint8)).save(masks_dir / f"{sample_id}.png")

    return f"images/{sample_id}.png", f"masks/{sample_id}.png"


def _prepare_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "msl_ncam_v1"
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append(("A",) + _write_pair(root, "A", np.array([[0, 0], [1, 1]], dtype=np.uint8)))
    rows.append(("B",) + _write_pair(root, "B", np.array([[1, 2], [2, 2]], dtype=np.uint8)))
    rows.append(("C",) + _write_pair(root, "C", np.array([[2, 2], [2, 3]], dtype=np.uint8)))

    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "mask_path"])
        writer.writeheader()
        for sample_id, image_path, mask_path in rows:
            writer.writerow({"id": sample_id, "image_path": image_path, "mask_path": mask_path})

    (split_dir / "train.txt").write_text("A\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("B\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("C\n", encoding="utf-8")

    return root


def test_script_writes_expected_csv(tmp_path: Path) -> None:
    root = _prepare_dataset(tmp_path)
    output_csv = tmp_path / "counts.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_class_pixel_stats.py",
            "--dataset-root",
            str(root),
            "--inspect-samples",
            "1",
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "observed unique ids" in result.stdout

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_id = {int(row["class_id"]): row for row in rows}

    assert by_id[0]["train"] == "2"
    assert by_id[1]["train"] == "2"
    assert by_id[1]["val"] == "1"
    assert by_id[2]["val"] == "3"
    assert by_id[2]["test"] == "3"
    assert by_id[3]["test"] == "1"
