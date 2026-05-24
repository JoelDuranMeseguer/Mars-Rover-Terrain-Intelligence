"""Simple A* path planning on a segmentation-derived terrain cost map."""

import argparse
import heapq
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
    mask_to_cost_map,
    to_uint8_image,
)

LOCAL_PLANNING_MIN_ROW_RATIO = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a simple path on a predicted terrain cost map")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/path_plans"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class3-threshold", type=float, default=None)
    parser.add_argument("--path-thickness", type=int, default=3)
    parser.add_argument("--safety-radius", type=int, default=0)
    return parser.parse_args()


def neighbors_8(r: int, c: int, h: int, w: int) -> list[tuple[int, int, float]]:
    out = []
    for dr, dc in [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            step = 1.41421356237 if (dr != 0 and dc != 0) else 1.0
            out.append((nr, nc, step))
    return out


def find_nearest_traversable(cost_map: np.ndarray, point: tuple[int, int]) -> tuple[int, int] | None:
    h, w = cost_map.shape
    pr, pc = point
    if 0 <= pr < h and 0 <= pc < w and cost_map[pr, pc] != CLASS_COSTS[255]:
        return point

    max_radius = max(h, w)
    for radius in range(1, max_radius):
        r0, r1 = max(0, pr - radius), min(h - 1, pr + radius)
        c0, c1 = max(0, pc - radius), min(w - 1, pc + radius)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if cost_map[rr, cc] != CLASS_COSTS[255]:
                    return (rr, cc)
    return None




def inflate_obstacles_square(cost_map: np.ndarray, safety_radius: int) -> np.ndarray:
    if safety_radius <= 0:
        return cost_map

    inflated = cost_map.copy()
    blocked = cost_map == CLASS_COSTS[255]
    rows, cols = np.where(blocked)
    h, w = cost_map.shape
    for r, c in zip(rows, cols):
        r0 = max(0, r - safety_radius)
        r1 = min(h, r + safety_radius + 1)
        c0 = max(0, c - safety_radius)
        c1 = min(w, c + safety_radius + 1)
        inflated[r0:r1, c0:c1] = CLASS_COSTS[255]
    return inflated

def astar(cost_map: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = cost_map.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if cost_map[start] == CLASS_COSTS[255] or cost_map[goal] == CLASS_COSTS[255]:
        return None

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    g_score = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    while open_heap:
        _, g_curr, curr = heapq.heappop(open_heap)
        if curr == goal:
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return path

        if g_curr > g_score.get(curr, float("inf")):
            continue

        for nr, nc, step_cost in neighbors_8(curr[0], curr[1], h, w):
            if cost_map[nr, nc] == CLASS_COSTS[255]:
                continue
            move_cost = float(cost_map[nr, nc]) * step_cost
            tentative_g = g_curr + move_cost
            n = (nr, nc)
            if tentative_g < g_score.get(n, float("inf")):
                came_from[n] = curr
                g_score[n] = tentative_g
                f = tentative_g + heuristic(n, goal)
                heapq.heappush(open_heap, (f, tentative_g, n))
    return None


def draw_path_on_cost_map(
    cost_map_rgb: np.ndarray,
    path: list[tuple[int, int]] | None,
    start: tuple[int, int],
    goal: tuple[int, int],
    path_thickness: int = 1,
) -> np.ndarray:
    out = cost_map_rgb.copy()
    radius = max(0, (path_thickness - 1) // 2)
    if path is not None:
        for r, c in path:
            if radius == 0:
                out[r, c] = np.array([0, 255, 255], dtype=np.uint8)  # cyan path
            else:
                r0 = max(0, r - radius)
                r1 = min(out.shape[0], r + radius + 1)
                c0 = max(0, c - radius)
                c1 = min(out.shape[1], c + radius + 1)
                out[r0:r1, c0:c1] = np.array([0, 255, 255], dtype=np.uint8)
    out[start[0], start[1]] = np.array([0, 255, 0], dtype=np.uint8)   # green start
    out[goal[0], goal[1]] = np.array([255, 0, 0], dtype=np.uint8)      # red goal
    return out


def make_triptych(image: np.ndarray, cost_map_rgb: np.ndarray, path_overlay: np.ndarray) -> np.ndarray:
    separator = np.full((image.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image, separator, cost_map_rgb, separator, path_overlay], axis=1)


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

    val_ds = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split="val")
    sample_idx = max(0, min(args.sample_idx, len(val_ds) - 1))
    sample = val_ds[sample_idx]

    with torch.no_grad():
        logits = model(sample["image"].unsqueeze(0).to(device))
    pred_mask = calibrate_class_prediction(
        logits=logits,
        class_index=3,
        threshold=args.class3_threshold,
    ).squeeze(0).detach().cpu().numpy().astype(np.int64)
    cost_map = mask_to_cost_map(pred_mask)

    # Local planning mask: allow planning only in lower part of the image.
    h, w = cost_map.shape
    min_plan_row = int(h * LOCAL_PLANNING_MIN_ROW_RATIO)
    masked_cost_map = cost_map.copy()
    masked_cost_map[:min_plan_row, :] = CLASS_COSTS[255]
    planning_cost_map = inflate_obstacles_square(masked_cost_map, args.safety_radius)

    raw_start = (max(h - 5, 0), w // 2)
    raw_goal = (min(min_plan_row + 5, h - 1), w // 2)
    start = find_nearest_traversable(planning_cost_map, raw_start)
    goal = find_nearest_traversable(planning_cost_map, raw_goal)

    if start is None or goal is None:
        raise RuntimeError("Could not find traversable start/goal points in this sample.")

    path = astar(planning_cost_map, start, goal)
    if path is None:
        print("No path found.")
    else:
        print(f"Path found with {len(path)} waypoints.")

    image_rgb = to_uint8_image(sample["image"])
    cost_rgb = colorize_cost_map(planning_cost_map)
    path_rgb = draw_path_on_cost_map(cost_rgb, path, start, goal, path_thickness=args.path_thickness)
    panel = make_triptych(image_rgb, cost_rgb, path_rgb)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{sample_idx:03d}_{sample['id']}_path_plan.png"
    Image.fromarray(panel).save(output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
