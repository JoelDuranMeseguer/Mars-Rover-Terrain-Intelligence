"""Evaluate a trained segmentation checkpoint on AI4Mars splits."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mrti.data.dataset import AI4MarsSegmentationDataset
from train_baseline import FIXED_CLASS_WEIGHTS, build_model


IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation checkpoint")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-class-weights", action="store_true")
    return parser.parse_args()


def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if valid.sum() == 0:
        return

    preds = preds[valid].to(torch.int64)
    targets = targets[valid].to(torch.int64)

    encoded = targets * num_classes + preds
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    confmat += counts.reshape(num_classes, num_classes)


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
    checkpoint_model_name = checkpoint.get("model_name", args.model)

    if checkpoint_model_name != args.model:
        raise ValueError(
            f"--model ({args.model}) does not match checkpoint model_name ({checkpoint_model_name})."
        )

    model = build_model(model_name=args.model, num_classes=args.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_weights = None
    if args.use_class_weights:
        class_weights = torch.tensor(FIXED_CLASS_WEIGHTS, dtype=torch.float32, device=device)
        if len(class_weights) != args.num_classes:
            raise ValueError(
                f"FIXED_CLASS_WEIGHTS expects {len(class_weights)} classes but got "
                f"--num-classes={args.num_classes}"
            )

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    total_loss = 0.0
    num_batches = 0
    correct_pixels = 0
    total_pixels = 0
    confmat = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            num_batches += 1

            valid = (masks != IGNORE_INDEX) & (masks >= 0) & (masks < args.num_classes)
            correct_pixels += (preds[valid] == masks[valid]).sum().item()
            total_pixels += valid.sum().item()

            update_confusion_matrix(
                confmat=confmat,
                preds=preds.detach().cpu(),
                targets=masks.detach().cpu(),
                num_classes=args.num_classes,
            )

    mean_loss = total_loss / max(num_batches, 1)
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    iou_per_class: list[float | None] = []
    for cls in range(args.num_classes):
        tp = confmat[cls, cls].item()
        fp = confmat[:, cls].sum().item() - tp
        fn = confmat[cls, :].sum().item() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else None
        iou_per_class.append(iou)

    valid_ious = [iou for iou in iou_per_class if iou is not None]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    print("=" * 60)
    print("Segmentation evaluation summary")
    print("=" * 60)
    print(f"dataset_root: {args.dataset_root}")
    print(f"split: {args.split}")
    print(f"num_samples: {len(dataset)}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"mean_loss: {mean_loss:.6f}")
    print(f"pixel_accuracy: {pixel_acc:.6f}")
    for cls, iou in enumerate(iou_per_class):
        if iou is None:
            print(f"iou_class_{cls}: n/a")
        else:
            print(f"iou_class_{cls}: {iou:.6f}")
    print(f"mean_iou: {mean_iou:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
