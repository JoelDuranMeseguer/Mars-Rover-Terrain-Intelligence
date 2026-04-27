"""Analyze class-3 segmentation errors and save visual panels."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mrti.data.dataset import AI4MarsSegmentationDataset
from train_baseline import build_model

IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze class-3 prediction errors")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--class-index", type=int, default=3)
    parser.add_argument("--max-samples-per-group", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/class3_error_analysis"),
    )
    return parser.parse_args()


def to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


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
    colored[mask == IGNORE_INDEX] = palette[4]
    return colored


def build_class_error_overlay(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    class_index: int,
) -> tuple[np.ndarray, int, int, int]:
    valid = target_mask != IGNORE_INDEX
    pred_cls = (pred_mask == class_index) & valid
    target_cls = (target_mask == class_index) & valid

    tp_mask = pred_cls & target_cls
    fp_mask = pred_cls & (~target_cls)
    fn_mask = (~pred_cls) & target_cls

    overlay = np.zeros((target_mask.shape[0], target_mask.shape[1], 3), dtype=np.uint8)
    overlay[tp_mask] = [0, 255, 0]      # TP -> green
    overlay[fp_mask] = [255, 0, 0]      # FP -> red
    overlay[fn_mask] = [0, 0, 255]      # FN -> blue
    return overlay, int(tp_mask.sum()), int(fp_mask.sum()), int(fn_mask.sum())


def make_panel(
    image: np.ndarray,
    target_mask: np.ndarray,
    pred_mask: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    target_rgb = colorize_mask(target_mask)
    pred_rgb = colorize_mask(pred_mask)
    separator = np.full((image.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image, separator, target_rgb, separator, pred_rgb, separator, overlay], axis=1)


def safe_metric(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint_model_name = checkpoint.get("model_name") or checkpoint.get("model") or args.model
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        checkpoint_model_name = args.model
        state_dict = checkpoint

    dataset = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split=args.split)
    model = build_model(model_name=str(checkpoint_model_name), num_classes=args.num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Device:", device)
    print("Split:", args.split)
    print("Checkpoint:", args.checkpoint)
    print("Class index:", args.class_index)
    print("Num samples:", len(dataset))

    sample_stats: list[dict] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_target = 0
    total_pred = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample["image"]
            target = sample["mask"]
            sample_id = sample["id"]

            logits = model(image.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            target_np = target.detach().cpu().numpy().astype(np.uint8)

            overlay, tp, fp, fn = build_class_error_overlay(
                pred_mask=pred,
                target_mask=target_np,
                class_index=args.class_index,
            )
            valid = target_np != IGNORE_INDEX
            target_pixels = int(((target_np == args.class_index) & valid).sum())
            pred_pixels = int(((pred == args.class_index) & valid).sum())
            precision = safe_metric(tp, tp + fp)
            recall = safe_metric(tp, tp + fn)
            iou = safe_metric(tp, tp + fp + fn)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_target += target_pixels
            total_pred += pred_pixels

            sample_stats.append(
                {
                    "dataset_index": idx,
                    "sample_id": sample_id,
                    "image_np": to_uint8_image(image),
                    "target_np": target_np,
                    "pred_np": pred,
                    "overlay_np": overlay,
                    "target_pixels": target_pixels,
                    "pred_pixels": pred_pixels,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "iou": iou,
                }
            )

    top_fp = [x for x in sample_stats if x["fp"] > 0]
    top_fp.sort(key=lambda x: x["fp"], reverse=True)
    top_fp = top_fp[: args.max_samples_per_group]

    top_fn = [x for x in sample_stats if x["fn"] > 0]
    top_fn.sort(key=lambda x: x["fn"], reverse=True)
    top_fn = top_fn[: args.max_samples_per_group]

    best_cases = [x for x in sample_stats if x["target_pixels"] > 0 and x["iou"] is not None]
    best_cases.sort(key=lambda x: (x["iou"], x["tp"]), reverse=True)
    best_cases = best_cases[: args.max_samples_per_group]

    saved_paths: dict[str, list[Path]] = {"top_fp": [], "top_fn": [], "best_cases": []}

    for group_name, items in [("top_fp", top_fp), ("top_fn", top_fn), ("best_cases", best_cases)]:
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for rank, item in enumerate(items, start=1):
            panel = make_panel(
                image=item["image_np"],
                target_mask=item["target_np"],
                pred_mask=item["pred_np"],
                overlay=item["overlay_np"],
            )
            filename = (
                f"{rank:03d}_{item['sample_id']}_"
                f"tp{item['tp']}_fp{item['fp']}_fn{item['fn']}.png"
            )
            output_path = group_dir / filename
            Image.fromarray(panel).save(output_path)
            saved_paths[group_name].append(output_path)

    agg_precision = safe_metric(total_tp, total_tp + total_fp)
    agg_recall = safe_metric(total_tp, total_tp + total_fn)
    agg_iou = safe_metric(total_tp, total_tp + total_fp + total_fn)

    print("=" * 72)
    print("Class error analysis summary")
    print("=" * 72)
    print(f"output_dir: {args.output_dir}")
    print(f"saved_top_fp: {len(saved_paths['top_fp'])} -> {args.output_dir / 'top_fp'}")
    print(f"saved_top_fn: {len(saved_paths['top_fn'])} -> {args.output_dir / 'top_fn'}")
    print(f"saved_best_cases: {len(saved_paths['best_cases'])} -> {args.output_dir / 'best_cases'}")
    print("-" * 72)
    print(f"class_index: {args.class_index}")
    print(f"total_target_pixels: {total_target}")
    print(f"total_predicted_pixels: {total_pred}")
    print(f"total_tp: {total_tp}")
    print(f"total_fp: {total_fp}")
    print(f"total_fn: {total_fn}")
    print(f"agg_precision: {agg_precision if agg_precision is not None else 'n/a'}")
    print(f"agg_recall: {agg_recall if agg_recall is not None else 'n/a'}")
    print(f"agg_iou: {agg_iou if agg_iou is not None else 'n/a'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
