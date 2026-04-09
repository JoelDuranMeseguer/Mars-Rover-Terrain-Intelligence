"""Prepare a clean AI4Mars subset for PyTorch training."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    required = {"id", "image_relpath", "label_relpath"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {missing_str}")

    return rows


def split_ids(sample_ids: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = sample_ids[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val : n_train + n_val + n_test]
    return train_ids, val_ids, test_ids


def write_split(path: Path, split_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id in split_ids:
            handle.write(f"{sample_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare msl_ncam_v1 data for training")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/AI4Mars"),
        help="Root folder for the raw AI4Mars dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifests/msl_ncam_v1.csv"),
        help="Manifest CSV with id, image_relpath, label_relpath",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/msl_ncam_v1"),
        help="Output folder for clean images, masks, splits and index.csv",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    splits_dir = out_dir / "splits"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest_rows(manifest_path)

    kept_rows: list[dict[str, str]] = []
    missing_count = 0

    for row in rows:
        sample_id = row["id"].strip()
        image_src = raw_root / row["image_relpath"]
        mask_src = raw_root / row["label_relpath"]

        if not image_src.exists() or not mask_src.exists():
            missing_count += 1
            continue

        image_dst_name = f"{sample_id}{image_src.suffix.lower()}"
        mask_dst_name = f"{sample_id}{mask_src.suffix.lower()}"
        image_dst = images_dir / image_dst_name
        mask_dst = masks_dir / mask_dst_name

        shutil.copy2(image_src, image_dst)
        shutil.copy2(mask_src, mask_dst)

        kept_rows.append(
            {
                "id": sample_id,
                "image_path": str(Path("images") / image_dst_name),
                "mask_path": str(Path("masks") / mask_dst_name),
            }
        )

    kept_rows.sort(key=lambda item: item["id"])
    sample_ids = [row["id"] for row in kept_rows]
    train_ids, val_ids, test_ids = split_ids(sample_ids, seed=args.seed)

    write_split(splits_dir / "train.txt", train_ids)
    write_split(splits_dir / "val.txt", val_ids)
    write_split(splits_dir / "test.txt", test_ids)

    index_path = out_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerows(kept_rows)

    print("AI4Mars data preparation complete")
    print(f"- manifest rows: {len(rows)}")
    print(f"- valid pairs copied: {len(kept_rows)}")
    print(f"- skipped (missing image/mask): {missing_count}")
    print(f"- train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"- output dir: {out_dir}")


if __name__ == "__main__":
    main()
