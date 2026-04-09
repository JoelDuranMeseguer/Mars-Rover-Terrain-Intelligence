"""Quick sanity check for AI4Mars PyTorch dataset loading."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check AI4MarsSegmentationDataset")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/msl_ncam_v1"),
        help="Folder containing index.csv, splits/, images/ and masks/",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from mrti.data import AI4MarsSegmentationDataset

    dataset = AI4MarsSegmentationDataset(dataset_root=args.data_root, split=args.split)

    print(f"Dataset root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")

    n = min(args.num_samples, len(dataset))
    for idx in range(n):
        sample = dataset[idx]
        image = sample["image"]
        mask = sample["mask"]
        sample_id = sample["id"]
        print(
            f"[{idx}] id={sample_id} image_shape={tuple(image.shape)} image_dtype={image.dtype} "
            f"mask_shape={tuple(mask.shape)} mask_dtype={mask.dtype}"
        )


if __name__ == "__main__":
    main()
