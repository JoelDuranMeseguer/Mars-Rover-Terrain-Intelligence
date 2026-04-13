"""Evaluate a trained segmentation checkpoint on AI4Mars splits."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mrti.data.dataset import AI4MarsSegmentationDataset
from scripts.train_baseline import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt/.pth)")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/processed/msl_ncam_v1"),
        help="Processed dataset root",
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ignore-index", type=int, default=255)
    return parser.parse_args()


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    valid = targets != ignore_index
    preds = preds[valid].view(-1)
    targets = targets[valid].view(-1)

    if preds.numel() == 0:
        return

    indices = targets * num_classes + preds
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion_matrix += counts.view(num_classes, num_classes)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict):
        model_name = checkpoint.get("model", args.model)
        num_classes = int(checkpoint.get("num_classes", args.num_classes))
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    else:
        model_name = args.model
        num_classes = args.num_classes
        state_dict = checkpoint

    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    if state_dict is None:
        raise ValueError(
            "No state_dict found in checkpoint. Expected key 'model_state_dict' or 'state_dict'."
        )
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            preds = logits.argmax(dim=1)
            update_confusion_matrix(
                confusion_matrix=confusion_matrix,
                preds=preds.cpu(),
                targets=masks.cpu(),
                num_classes=num_classes,
                ignore_index=args.ignore_index,
            )

            total_loss += loss.item()
            total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)

    tp = confusion_matrix.diag().to(torch.float64)
    fp = confusion_matrix.sum(dim=0).to(torch.float64) - tp
    fn = confusion_matrix.sum(dim=1).to(torch.float64) - tp

    iou_per_class = []
    for cls in range(num_classes):
        denom = tp[cls] + fp[cls] + fn[cls]
        iou = (tp[cls] / denom).item() if denom > 0 else 0.0
        iou_per_class.append(iou)

    mean_iou = sum(iou_per_class) / max(len(iou_per_class), 1)

    total_pixels = confusion_matrix.sum().item()
    pixel_acc = (tp.sum().item() / total_pixels) if total_pixels > 0 else 0.0

    print("=" * 70)
    print("Segmentation evaluation")
    print("-" * 70)
    print(f"checkpoint   : {args.checkpoint}")
    print(f"dataset root : {args.dataset_root}")
    print(f"split        : {args.split}")
    print(f"model        : {model_name}")
    print(f"num classes  : {num_classes}")
    print("-" * 70)
    print(f"mean loss    : {mean_loss:.6f}")
    print(f"pixel acc    : {pixel_acc:.6f}")
    for cls, iou in enumerate(iou_per_class):
        print(f"iou class {cls:<2}: {iou:.6f}")
    print(f"mean IoU     : {mean_iou:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
