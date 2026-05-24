"""Evaluate local planning metrics over threshold/safety-radius sweeps."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from mrti.data.dataset import AI4MarsSegmentationDataset
from train_baseline import build_model
from build_cost_map import CLASS_COSTS, calibrate_class_prediction, mask_to_cost_map
from plan_path import LOCAL_PLANNING_MIN_ROW_RATIO, astar, find_nearest_traversable, inflate_obstacles_square


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep planning metrics on validation split")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.6])
    parser.add_argument("--safety-radii", type=int, nargs="*", default=[0])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def path_cost(cost_map: np.ndarray, path: list[tuple[int, int]] | None) -> float | None:
    if path is None or len(path) <= 1:
        return None
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        step = 1.41421356237 if (r0 != r1 and c0 != c1) else 1.0
        total += float(cost_map[r1, c1]) * step
    return total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_model_name = checkpoint["model_name"]
    if args.model != checkpoint_model_name:
        raise ValueError(f"--model ({args.model}) does not match checkpoint model_name ({checkpoint_model_name}).")

    model = build_model(model_name=checkpoint_model_name, num_classes=args.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split="val")
    n_samples = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)

    print("=" * 88)
    print("Planning sweep summary")
    print("=" * 88)
    print(f"checkpoint: {args.checkpoint}")
    print(f"model: {args.model}")
    print(f"num_samples: {n_samples}")

    with torch.no_grad():
        for threshold, safety_radius in itertools.product(args.thresholds, args.safety_radii):
            found = 0
            attempted = 0
            lengths = []
            costs = []

            for idx in range(n_samples):
                sample = dataset[idx]
                logits = model(sample["image"].unsqueeze(0).to(device))
                pred_mask = calibrate_class_prediction(
                    logits=logits,
                    class_index=3,
                    threshold=threshold,
                ).squeeze(0).detach().cpu().numpy().astype(np.int64)

                raw_cost_map = mask_to_cost_map(pred_mask)
                h, w = raw_cost_map.shape
                min_plan_row = int(h * LOCAL_PLANNING_MIN_ROW_RATIO)
                masked_cost_map = raw_cost_map.copy()
                masked_cost_map[:min_plan_row, :] = CLASS_COSTS[255]
                planning_cost_map = inflate_obstacles_square(masked_cost_map, safety_radius)

                raw_start = (max(h - 5, 0), w // 2)
                raw_goal = (min(min_plan_row + 5, h - 1), w // 2)
                start = find_nearest_traversable(planning_cost_map, raw_start)
                goal = find_nearest_traversable(planning_cost_map, raw_goal)
                if start is None or goal is None:
                    continue

                attempted += 1
                path = astar(planning_cost_map, start, goal)
                if path is None:
                    continue

                found += 1
                lengths.append(len(path))
                c = path_cost(planning_cost_map, path)
                if c is not None:
                    costs.append(c)

            success = (found / attempted) if attempted > 0 else 0.0
            mean_len = float(np.mean(lengths)) if lengths else 0.0
            mean_cost = float(np.mean(costs)) if costs else 0.0

            print("-" * 88)
            print(f"threshold={threshold:.3f} | safety_radius={safety_radius}")
            print(f"attempted: {attempted}")
            print(f"path_found: {found}")
            print(f"success_rate: {success:.6f}")
            print(f"mean_path_length: {mean_len:.6f}")
            print(f"mean_path_cost: {mean_cost:.6f}")

    print("=" * 88)


if __name__ == "__main__":
    main()
