"""Pixel-class sanity check for processed segmentation datasets.

Recorre train/val/test, cuenta píxeles por clase y muestra un resumen.
Opcionalmente guarda un CSV con los resultados.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count mask pixels per class for each split")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/processed/msl_ncam_v1"),
        help="Dataset folder with index.csv, masks/, splits/",
    )
    parser.add_argument(
        "--inspect-samples",
        type=int,
        default=5,
        help="How many masks per split to inspect first for unique class ids",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path (example: outputs/class_pixel_stats.csv)",
    )
    return parser.parse_args()


def load_index(index_path: Path) -> dict[str, str]:
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping: dict[str, str] = {}
        for row in reader:
            sample_id = row["id"].strip()
            if sample_id:
                mapping[sample_id] = row["mask_path"]
    return mapping


def read_split_ids(split_path: Path) -> list[str]:
    ids: list[str] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_id = line.strip()
            if sample_id:
                ids.append(sample_id)
    return ids


def read_mask_ids(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


def inspect_unique_ids(dataset_root: Path, ids: list[str], index_map: dict[str, str], limit: int) -> set[int]:
    observed: set[int] = set()
    n = min(limit, len(ids))

    for sample_id in ids[:n]:
        rel = index_map.get(sample_id)
        if rel is None:
            continue
        mask_ids = read_mask_ids(dataset_root / rel)
        observed.update(int(v) for v in np.unique(mask_ids))

    return observed


def count_pixels_for_split(dataset_root: Path, ids: list[str], index_map: dict[str, str]) -> Counter[int]:
    counts: Counter[int] = Counter()

    for sample_id in ids:
        rel = index_map.get(sample_id)
        if rel is None:
            continue
        mask_ids = read_mask_ids(dataset_root / rel)
        unique_ids, unique_counts = np.unique(mask_ids, return_counts=True)
        for class_id, class_pixels in zip(unique_ids.tolist(), unique_counts.tolist()):
            counts[int(class_id)] += int(class_pixels)

    return counts


def print_summary(split_counts: dict[str, Counter[int]]) -> None:
    class_ids = sorted({cid for counts in split_counts.values() for cid in counts})

    print("\n=== Pixel counts by split ===")
    header = ["class_id", "train", "val", "test", "total"]
    print(" | ".join(header))
    print("-" * 54)

    for class_id in class_ids:
        train = split_counts["train"].get(class_id, 0)
        val = split_counts["val"].get(class_id, 0)
        test = split_counts["test"].get(class_id, 0)
        total = train + val + test
        print(f"{class_id} | {train} | {val} | {test} | {total}")


def save_csv(output_path: Path, split_counts: dict[str, Counter[int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_ids = sorted({cid for counts in split_counts.values() for cid in counts})

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "train", "val", "test", "total"])
        writer.writeheader()

        for class_id in class_ids:
            train = split_counts["train"].get(class_id, 0)
            val = split_counts["val"].get(class_id, 0)
            test = split_counts["test"].get(class_id, 0)
            writer.writerow(
                {
                    "class_id": class_id,
                    "train": train,
                    "val": val,
                    "test": test,
                    "total": train + val + test,
                }
            )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root

    index_path = dataset_root / "index.csv"
    split_dir = dataset_root / "splits"

    index_map = load_index(index_path)

    split_ids = {
        "train": read_split_ids(split_dir / "train.txt"),
        "val": read_split_ids(split_dir / "val.txt"),
        "test": read_split_ids(split_dir / "test.txt"),
    }

    print("Dataset root:", dataset_root)
    print("Inspecting unique ids from first", args.inspect_samples, "masks per split...")
    for split_name, ids in split_ids.items():
        observed = inspect_unique_ids(dataset_root, ids, index_map, args.inspect_samples)
        print(f"- {split_name}: observed unique ids (sampled) = {sorted(observed)}")

    split_counts = {
        split_name: count_pixels_for_split(dataset_root, ids, index_map)
        for split_name, ids in split_ids.items()
    }

    print_summary(split_counts)

    if args.output_csv is not None:
        save_csv(args.output_csv, split_counts)
        print(f"\nCSV saved: {args.output_csv}")


if __name__ == "__main__":
    main()
