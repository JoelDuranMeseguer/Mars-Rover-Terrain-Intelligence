"""Build simple terrain cost maps from baseline segmentation predictions."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mrti.data.dataset import AI4MarsSegmentationDataset
from train_baseline import build_model


# Explicit and simple class-to-cost mapping for interpretation.
CLASS_COSTS = {
    0: 1,    # low cost
    1: 3,    # medium cost
    2: 7,    # high cost
    3: 10,   # very high cost
    255: 255,  # invalid / non-traversable
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build terrain cost maps from segmentation predictions")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cost_maps"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def colorize_pred_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],       # class 0
            [255, 0, 0],     # class 1
            [0, 255, 0],     # class 2
            [0, 0, 255],     # class 3
            [255, 0, 255],   # invalid / unknown
        ],
        dtype=np.uint8,
    )
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask == 0] = palette[0]
    out[mask == 1] = palette[1]
    out[mask == 2] = palette[2]
    out[mask == 3] = palette[3]
    out[(mask < 0) | (mask > 3) | (mask == 255)] = palette[4]
    return out


def mask_to_cost_map(mask: np.ndarray) -> np.ndarray:
    cost_map = np.full(mask.shape, CLASS_COSTS[255], dtype=np.uint8)
    for cls_id in (0, 1, 2, 3):
        cost_map[mask == cls_id] = CLASS_COSTS[cls_id]
    return cost_map


def colorize_cost_map(cost_map: np.ndarray) -> np.ndarray:
    # Grayscale visualization: low cost -> dark, high cost -> bright.
    max_cost = max(CLASS_COSTS.values())
    vis = (cost_map.astype(np.float32) / max_cost * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([vis, vis, vis], axis=-1)


def make_triptych(image: np.ndarray, pred_mask_rgb: np.ndarray, cost_map_rgb: np.ndarray) -> np.ndarray:
    separator = np.full((image.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image, separator, pred_mask_rgb, separator, cost_map_rgb], axis=1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Class costs:", CLASS_COSTS)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_model_name = checkpoint["model_name"]
    if args.model != checkpoint_model_name:
        raise ValueError(
            f"--model ({args.model}) does not match checkpoint model_name ({checkpoint_model_name})."
        )

    model = build_model(model_name=checkpoint_model_name, num_classes=args.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_ds = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split="val")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = min(args.num_samples, len(val_ds))
    print(f"Saving {n_samples} cost-map previews to: {args.output_dir}")
    with torch.no_grad():
        for idx in range(n_samples):
            sample = val_ds[idx]
            image = sample["image"]
            sample_id = sample["id"]

            logits = model(image.unsqueeze(0).to(device))
            pred_mask = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)

            cost_map = mask_to_cost_map(pred_mask)

            image_rgb = to_uint8_image(image)
            pred_rgb = colorize_pred_mask(pred_mask)
            cost_rgb = colorize_cost_map(cost_map)
            panel = make_triptych(image_rgb, pred_rgb, cost_rgb)

            output_path = args.output_dir / f"{idx:03d}_{sample_id}_cost_map.png"
            Image.fromarray(panel).save(output_path)
            print(f"saved={output_path}")


if __name__ == "__main__":
    main()
