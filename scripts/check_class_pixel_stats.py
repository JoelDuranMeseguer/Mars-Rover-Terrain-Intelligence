"""Pixel-class sanity check for processed segmentation datasets.

Recorre train/val/test, cuenta píxeles por clase y muestra un resumen.
Opcionalmente guarda un CSV con los resultados.
"""

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


def read_mask_ids(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root

    # 1) Cargar index.csv (id -> mask_path)
    index_map: dict[str, str] = {}
    with (dataset_root / "index.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row["id"].strip()
            if sample_id:
                index_map[sample_id] = row["mask_path"]

    # 2) Cargar ids por split
    split_ids: dict[str, list[str]] = {}
    for split_name in ["train", "val", "test"]:
        ids: list[str] = []
        split_path = dataset_root / "splits" / f"{split_name}.txt"
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                sample_id = line.strip()
                if sample_id:
                    ids.append(sample_id)
        split_ids[split_name] = ids

    print("Dataset root:", dataset_root)
    print("Inspecting unique ids from first", args.inspect_samples, "masks per split...")

    # 3) Inspección rápida de ids únicos (sin asumir clases)
    for split_name in ["train", "val", "test"]:
        observed: set[int] = set()
        ids = split_ids[split_name]
        n = min(args.inspect_samples, len(ids))
        for sample_id in ids[:n]:
            rel = index_map.get(sample_id)
            if rel is None:
                continue
            observed.update(int(v) for v in np.unique(read_mask_ids(dataset_root / rel)))
        print(f"- {split_name}: observed unique ids (sampled) = {sorted(observed)}")

    # 4) Conteo total por split
    split_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    for split_name in ["train", "val", "test"]:
        for sample_id in split_ids[split_name]:
            rel = index_map.get(sample_id)
            if rel is None:
                continue
            mask_ids = read_mask_ids(dataset_root / rel)
            unique_ids, unique_counts = np.unique(mask_ids, return_counts=True)
            for class_id, class_pixels in zip(unique_ids.tolist(), unique_counts.tolist()):
                split_counts[split_name][int(class_id)] += int(class_pixels)

    # 5) Resumen por consola
    class_ids = sorted({cid for counts in split_counts.values() for cid in counts})
    print("\n=== Pixel counts by split ===")
    print("class_id | train | val | test | total")
    print("-" * 54)
    for class_id in class_ids:
        train = split_counts["train"].get(class_id, 0)
        val = split_counts["val"].get(class_id, 0)
        test = split_counts["test"].get(class_id, 0)
        total = train + val + test
        print(f"{class_id} | {train} | {val} | {test} | {total}")

    # 6) CSV opcional
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
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
        print(f"\nCSV saved: {args.output_csv}")


if __name__ == "__main__":
    main()
