"""Simple visual sanity check for AI4Mars dataset samples.

Example:
    python scripts/check_dataset.py \
        --dataset-root data/processed/msl_ncam_v1 \
        --split train \
        --num-samples 4
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mrti.data.dataset import AI4MarsSegmentationDataset


DEFAULT_COLORS = np.array(
    [
        [0, 0, 0],        # 0 background / unknown
        [0, 180, 0],      # 1 terrain
        [220, 0, 0],      # 2 obstacle
        [0, 120, 220],    # 3 other class
        [220, 180, 0],    # 4 other class
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual sanity check for AI4Mars dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/processed/msl_ncam_v1"),
        help="Path to msl_ncam_v1 processed dataset",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to preview")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_preview"),
        help="Where preview PNG files are written",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="If set, only image+mask are saved side by side",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split=args.split)

    print("Dataset root:", args.dataset_root)
    print("Split:", args.split)
    print("Samples:", len(dataset))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        sample_id = str(sample["id"])

        # Imagen tensor (C,H,W, 0..1) -> uint8 (H,W,C)
        image_uint8 = sample["image"].permute(1, 2, 0).cpu().numpy()
        image_uint8 = np.clip(image_uint8 * 255.0, 0, 255).astype(np.uint8)

        # Máscara -> ids enteros
        mask_ids = sample["mask"].cpu().numpy().astype(np.int64)

        # Si aparece un id mayor que la paleta base, repetimos el último color.
        max_value = int(mask_ids.max())
        if max_value >= len(DEFAULT_COLORS):
            extra = max_value + 1 - len(DEFAULT_COLORS)
            repeated = np.tile(DEFAULT_COLORS[-1], (extra, 1))
            colors = np.vstack([DEFAULT_COLORS, repeated])
        else:
            colors = DEFAULT_COLORS

        mask_rgb = colors[mask_ids]

        panels = [image_uint8, mask_rgb]
        if not args.no_overlay:
            overlay = ((1.0 - 0.4) * image_uint8 + 0.4 * mask_rgb).astype(np.uint8)
            panels.append(overlay)

        preview = np.concatenate(panels, axis=1)
        output_name = f"{i:02d}_{sample_id}.png"
        output_path = args.output_dir / output_name
        Image.fromarray(preview).save(output_path)

        print(
            f"[{i}] id={sample_id} "
            f"image_shape={tuple(sample['image'].shape)} "
            f"mask_shape={tuple(sample['mask'].shape)} "
            f"saved={output_path}"
        )


if __name__ == "__main__":
    main()
