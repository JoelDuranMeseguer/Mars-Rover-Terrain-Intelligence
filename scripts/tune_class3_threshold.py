"""Sweep post-inference class-threshold calibration for segmentation checkpoints."""

import argparse
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mrti.data.dataset import AI4MarsSegmentationDataset
from train_baseline import build_model

IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune post-inference threshold/margin for one class")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-index", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.5])
    parser.add_argument("--margins", type=float, nargs="*", default=[0.0])
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    default_model_name: str,
    default_num_classes: int,
) -> tuple[dict[str, torch.Tensor], str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        model_name = str(checkpoint.get("model_name") or checkpoint.get("model") or default_model_name)
        num_classes = int(checkpoint.get("num_classes", default_num_classes))
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        model_name = default_model_name
        num_classes = default_num_classes
        state_dict = checkpoint
    return state_dict, model_name, num_classes


def build_configs(thresholds: list[float], margins: list[float]) -> list[dict[str, float | None]]:
    configs: list[dict[str, float | None]] = [{"threshold": None, "margin": None}]
    for threshold, margin in itertools.product(thresholds, margins):
        configs.append({"threshold": float(threshold), "margin": float(margin)})
    return configs


def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if not valid.any():
        return

    preds = preds[valid].to(torch.int64)
    targets = targets[valid].to(torch.int64)
    encoded = targets * num_classes + preds
    confmat += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def apply_class_calibration(
    base_preds: torch.Tensor,
    best_probs: torch.Tensor,
    best_indices: torch.Tensor,
    class_index: int,
    threshold: float | None,
    margin: float | None,
) -> torch.Tensor:
    if threshold is None and margin is None:
        return base_preds

    calibrated = base_preds.clone()
    pred_is_class = base_preds == class_index
    if not pred_is_class.any():
        return calibrated

    class_prob = best_probs[..., 0]
    second_prob = best_probs[..., 1]
    second_class = best_indices[..., 1]

    should_replace = torch.zeros_like(pred_is_class)
    if threshold is not None:
        should_replace |= class_prob < threshold
    if margin is not None:
        should_replace |= (class_prob - second_prob) < margin

    replace_mask = pred_is_class & should_replace
    calibrated[replace_mask] = second_class[replace_mask]
    return calibrated


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def summarize_config(
    confmat: torch.Tensor,
    correct_pixels: int,
    total_pixels: int,
    class_index: int,
) -> dict[str, float | int | None]:
    num_classes = confmat.shape[0]

    iou_values: list[float] = []
    for cls in range(num_classes):
        tp = int(confmat[cls, cls].item())
        fp = int(confmat[:, cls].sum().item() - tp)
        fn = int(confmat[cls, :].sum().item() - tp)
        iou = safe_div(tp, tp + fp + fn)
        if iou is not None:
            iou_values.append(iou)

    mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0.0
    pixel_accuracy = safe_div(correct_pixels, total_pixels)

    tp_c3 = int(confmat[class_index, class_index].item())
    fp_c3 = int(confmat[:, class_index].sum().item() - tp_c3)
    fn_c3 = int(confmat[class_index, :].sum().item() - tp_c3)

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "precision_class": safe_div(tp_c3, tp_c3 + fp_c3),
        "recall_class": safe_div(tp_c3, tp_c3 + fn_c3),
        "iou_class": safe_div(tp_c3, tp_c3 + fp_c3 + fn_c3),
        "target_pixels_class": int(confmat[class_index, :].sum().item()),
        "predicted_pixels_class": int(confmat[:, class_index].sum().item()),
        "tp": tp_c3,
        "fp": fp_c3,
        "fn": fn_c3,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict, model_name, num_classes = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        default_model_name=args.model,
        default_num_classes=args.num_classes,
    )

    if args.class_index < 0 or args.class_index >= num_classes:
        raise ValueError(f"--class-index must be in [0, {num_classes - 1}]")

    dataset = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    configs = build_configs(args.thresholds, args.margins)
    stats = []
    for config in configs:
        stats.append(
            {
                "threshold": config["threshold"],
                "margin": config["margin"],
                "confmat": torch.zeros((num_classes, num_classes), dtype=torch.int64),
                "correct_pixels": 0,
                "total_pixels": 0,
            }
        )

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            best_probs, best_indices = probs.topk(k=2, dim=1)
            base_preds = best_indices[:, 0]

            valid = (targets != IGNORE_INDEX) & (targets >= 0) & (targets < num_classes)
            for config_stats in stats:
                calibrated_preds = apply_class_calibration(
                    base_preds=base_preds,
                    best_probs=best_probs,
                    best_indices=best_indices,
                    class_index=args.class_index,
                    threshold=config_stats["threshold"],
                    margin=config_stats["margin"],
                )
                config_stats["correct_pixels"] += (calibrated_preds[valid] == targets[valid]).sum().item()
                config_stats["total_pixels"] += valid.sum().item()
                update_confusion_matrix(
                    confmat=config_stats["confmat"],
                    preds=calibrated_preds.detach().cpu(),
                    targets=targets.detach().cpu(),
                    num_classes=num_classes,
                )

    print("=" * 80)
    print("Class calibration sweep summary")
    print("=" * 80)
    print(f"dataset_root: {args.dataset_root}")
    print(f"split: {args.split}")
    print(f"num_samples: {len(dataset)}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"model: {model_name}")
    print(f"num_classes: {num_classes}")
    print(f"class_index: {args.class_index}")
    print(f"device: {device}")

    for idx, config_stats in enumerate(stats):
        metrics = summarize_config(
            confmat=config_stats["confmat"],
            correct_pixels=int(config_stats["correct_pixels"]),
            total_pixels=int(config_stats["total_pixels"]),
            class_index=args.class_index,
        )
        threshold = config_stats["threshold"]
        margin = config_stats["margin"]

        print("-" * 80)
        print(f"config_{idx}:")
        print(f"  threshold: {threshold if threshold is not None else 'none'}")
        print(f"  margin: {margin if margin is not None else 'none'}")
        print(f"  pixel_accuracy: {format_metric(metrics['pixel_accuracy'])}")
        print(f"  mean_iou: {format_metric(metrics['mean_iou'])}")
        print(f"  precision_class_{args.class_index}: {format_metric(metrics['precision_class'])}")
        print(f"  recall_class_{args.class_index}: {format_metric(metrics['recall_class'])}")
        print(f"  iou_class_{args.class_index}: {format_metric(metrics['iou_class'])}")
        print(f"  target_pixels_class_{args.class_index}: {metrics['target_pixels_class']}")
        print(f"  predicted_pixels_class_{args.class_index}: {metrics['predicted_pixels_class']}")
        print(f"  tp_class_{args.class_index}: {metrics['tp']}")
        print(f"  fp_class_{args.class_index}: {metrics['fp']}")
        print(f"  fn_class_{args.class_index}: {metrics['fn']}")

    print("=" * 80)


if __name__ == "__main__":
    main()
