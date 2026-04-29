"""Generate a single end-to-end demo panel for README/demo usage."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mrti.data.dataset import AI4MarsSegmentationDataset
from train_baseline import build_model
from build_cost_map import (
    CLASS_COSTS,
    calibrate_class_prediction,
    colorize_cost_map,
    colorize_pred_mask,
    mask_to_cost_map,
    to_uint8_image,
)
from plan_path import (
    LOCAL_PLANNING_MIN_ROW_RATIO,
    astar,
    draw_path_on_cost_map,
    find_nearest_traversable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one horizontal demo panel: image|prediction|cost|path")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--class3-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_panels"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_panel(
    image: np.ndarray,
    pred_rgb: np.ndarray,
    cost_rgb: np.ndarray,
    path_rgb: np.ndarray,
) -> np.ndarray:
    separator = np.full((image.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image, separator, pred_rgb, separator, cost_rgb, separator, path_rgb], axis=1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_model_name = checkpoint["model_name"]
    if args.model != checkpoint_model_name:
        raise ValueError(
            f"--model ({args.model}) does not match checkpoint model_name ({checkpoint_model_name})."
        )

    model = build_model(model_name=checkpoint_model_name, num_classes=args.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split="val")
    sample_idx = max(0, min(args.sample_idx, len(dataset) - 1))
    sample = dataset[sample_idx]

    with torch.no_grad():
        logits = model(sample["image"].unsqueeze(0).to(device))
    pred_mask = calibrate_class_prediction(
        logits=logits,
        class_index=3,
        threshold=args.class3_threshold,
    ).squeeze(0).detach().cpu().numpy().astype(np.int64)

    cost_map = mask_to_cost_map(pred_mask)
    h, w = cost_map.shape
    min_plan_row = int(h * LOCAL_PLANNING_MIN_ROW_RATIO)
    masked_cost_map = cost_map.copy()
    masked_cost_map[:min_plan_row, :] = CLASS_COSTS[255]

    raw_start = (max(h - 5, 0), w // 2)
    raw_goal = (min(min_plan_row + 5, h - 1), w // 2)
    start = find_nearest_traversable(masked_cost_map, raw_start)
    goal = find_nearest_traversable(masked_cost_map, raw_goal)
    if start is None or goal is None:
        raise RuntimeError("Could not find traversable start/goal points in this sample.")

    path = astar(masked_cost_map, start, goal)
    if path is None:
        print("No path found.")
    else:
        print(f"Path found with {len(path)} waypoints.")

    image_rgb = to_uint8_image(sample["image"])
    pred_rgb = colorize_pred_mask(pred_mask)
    cost_rgb = colorize_cost_map(masked_cost_map)
    path_rgb = draw_path_on_cost_map(cost_rgb, path, start, goal)
    panel = make_panel(image_rgb, pred_rgb, cost_rgb, path_rgb)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{sample_idx:03d}_{sample['id']}_demo_panel.png"
    Image.fromarray(panel).save(output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
