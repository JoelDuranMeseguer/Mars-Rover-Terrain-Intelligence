"""Visual inspection tool for baseline segmentation predictions on val split."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mrti.data.dataset import AI4MarsSegmentationDataset
from train_baseline import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect baseline predictions on validation samples")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],        # class 0
            [255, 0, 0],      # class 1
            [0, 255, 0],      # class 2
            [0, 0, 255],      # class 3
            [255, 0, 255],    # ignore (255)
        ],
        dtype=np.uint8,
    )
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in range(4):
        colored[mask == cls] = palette[cls]
    colored[mask == 255] = palette[4]
    return colored


def to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def make_triptych(image: np.ndarray, target_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    target_rgb = colorize_mask(target_mask)
    pred_rgb = colorize_mask(pred_mask)
    separator = np.full((image.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image, separator, target_rgb, separator, pred_rgb], axis=1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_model_name = checkpoint.get("model_name")
    model_name = checkpoint_model_name if checkpoint_model_name is not None else args.model
    print("Model:", model_name)

    model = build_model(model_name=model_name, num_classes=args.num_classes).to(device)
    model_state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()

    val_ds = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split="val")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = min(args.num_samples, len(val_ds))
    print(f"Saving {n_samples} prediction previews to: {args.output_dir}")
    with torch.no_grad():
        for idx in range(n_samples):
            sample = val_ds[idx]
            image = sample["image"]
            target = sample["mask"]
            sample_id = sample["id"]

            logits = model(image.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            image_np = to_uint8_image(image)
            target_np = target.detach().cpu().numpy().astype(np.uint8)
            canvas = make_triptych(image_np, target_np, pred)
            output_path = args.output_dir / f"{idx:03d}_{sample_id}.png"
            Image.fromarray(canvas).save(output_path)
            print(f"saved={output_path}")


if __name__ == "__main__":
    main()
