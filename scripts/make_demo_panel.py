"""Generate end-to-end demo panels for README/demo usage."""

import argparse
import csv
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
    inflate_obstacles_square,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one horizontal demo panel: image|prediction|cost|path")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None)
    parser.add_argument("--class3-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_panels"))
    parser.add_argument("--path-thickness", type=int, default=3)
    parser.add_argument("--safety-radius", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def make_panel(
    image: np.ndarray,
    pred_rgb: np.ndarray,
    cost_rgb: np.ndarray,
    path_rgb: np.ndarray,
) -> np.ndarray:
    separator = np.full((image.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image, separator, pred_rgb, separator, cost_rgb, separator, path_rgb], axis=1)




def apply_planning_mask_visual(planning_rgb: np.ndarray, min_plan_row: int) -> np.ndarray:
    out = planning_rgb.copy()
    out[:min_plan_row, :] = np.array([45, 45, 45], dtype=np.uint8)
    return out

def build_sample_indices(sample_idx: int, sample_indices: list[int] | None, dataset_len: int) -> list[int]:
    raw_indices = sample_indices if sample_indices else [sample_idx]
    return [max(0, min(idx, dataset_len - 1)) for idx in raw_indices]


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
    selected_indices = build_sample_indices(args.sample_idx, args.sample_indices, len(dataset))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    manifest_rows = []

    with torch.no_grad():
        for sample_idx in selected_indices:
            sample = dataset[sample_idx]
            logits = model(sample["image"].unsqueeze(0).to(device))
            pred_mask = calibrate_class_prediction(
                logits=logits,
                class_index=3,
                threshold=args.class3_threshold,
            ).squeeze(0).detach().cpu().numpy().astype(np.int64)

            raw_cost_map = mask_to_cost_map(pred_mask)
            h, w = raw_cost_map.shape
            min_plan_row = int(h * LOCAL_PLANNING_MIN_ROW_RATIO)
            masked_cost_map = raw_cost_map.copy()
            masked_cost_map[:min_plan_row, :] = CLASS_COSTS[255]
            planning_cost_map = inflate_obstacles_square(masked_cost_map, args.safety_radius)

            raw_start = (max(h - 5, 0), w // 2)
            raw_goal = (min(min_plan_row + 5, h - 1), w // 2)
            start = find_nearest_traversable(planning_cost_map, raw_start)
            goal = find_nearest_traversable(planning_cost_map, raw_goal)
            if start is None or goal is None:
                print(f"skipped sample_idx={sample_idx}: no traversable start/goal")
                continue

            path = astar(planning_cost_map, start, goal)

            image_rgb = to_uint8_image(sample["image"])
            pred_rgb = colorize_pred_mask(pred_mask)
            cost_rgb = colorize_cost_map(raw_cost_map)
            planning_rgb = colorize_cost_map(planning_cost_map)
            planning_rgb = apply_planning_mask_visual(planning_rgb, min_plan_row=min_plan_row)
            path_rgb = draw_path_on_cost_map(planning_rgb, path, start, goal, path_thickness=args.path_thickness)
            panel = make_panel(image_rgb, pred_rgb, cost_rgb, path_rgb)

            output_path = args.output_dir / f"{sample_idx:03d}_{sample['id']}_demo_panel.png"
            Image.fromarray(panel).save(output_path)
            saved_count += 1
            manifest_rows.append({"sample_idx": sample_idx, "sample_id": sample["id"], "path_found": path is not None, "output_path": str(output_path)})
            print(f"saved={output_path}")

    if args.write_manifest:
        manifest_path = args.output_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_idx", "sample_id", "path_found", "output_path"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"manifest={manifest_path}")

    print("-" * 72)
    print("Demo panel generation summary")
    print(f"saved_panels: {saved_count}")
    print(f"output_dir: {args.output_dir}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"class3_threshold: {args.class3_threshold}")
    print(f"path_thickness: {args.path_thickness}")
    print(f"safety_radius: {args.safety_radius}")


if __name__ == "__main__":
    main()
