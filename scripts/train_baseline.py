"""Baseline mínimo de entrenamiento para segmentación en AI4Mars.

Conecta dataset + dataloader + modelo + loss en un flujo sencillo.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mrti.data.dataset import AI4MarsSegmentationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple baseline training for segmentation")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-batches", type=int, default=1)
    parser.add_argument("--max-val-batches", type=int, default=1)
    return parser.parse_args()


def build_model(num_classes: int) -> nn.Module:
    # CNN pequeña: logits por pixel con la misma resolución de entrada.
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, num_classes, kernel_size=1),
    )


def update_iou_stats(
    preds: torch.Tensor,
    targets: torch.Tensor,
    intersections: torch.Tensor,
    unions: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> None:
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersections[cls] += (pred_cls & target_cls).sum().item()
        unions[cls] += (pred_cls | target_cls).sum().item()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split="train")
    val_ds = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = build_model(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            print(
                f"[epoch {epoch + 1}] train batch {batch_idx + 1} "
                f"loss={loss.item():.4f}"
            )

            if (batch_idx + 1) >= args.max_train_batches:
                break

        model.eval()
        val_losses = []
        iou_intersections = torch.zeros(args.num_classes, dtype=torch.float64)
        iou_unions = torch.zeros(args.num_classes, dtype=torch.float64)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                update_iou_stats(
                    preds=preds,
                    targets=masks,
                    intersections=iou_intersections,
                    unions=iou_unions,
                    num_classes=args.num_classes,
                )

                print(
                    f"[epoch {epoch + 1}] val batch {batch_idx + 1} "
                    f"loss={loss.item():.4f}"
                )

                if (batch_idx + 1) >= args.max_val_batches:
                    break

        train_mean = sum(train_losses) / max(len(train_losses), 1)
        val_mean = sum(val_losses) / max(len(val_losses), 1)
        iou_per_class = []
        for cls in range(args.num_classes):
            union = iou_unions[cls].item()
            iou = iou_intersections[cls].item() / union if union > 0 else 0.0
            iou_per_class.append(iou)
        mean_iou = sum(iou_per_class) / max(len(iou_per_class), 1)
        iou_text = ", ".join(
            [f"class_{cls}_iou={iou_per_class[cls]:.4f}" for cls in range(args.num_classes)]
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} summary -> "
            f"train_loss={train_mean:.4f}, val_loss={val_mean:.4f}, "
            f"{iou_text}, mean_iou={mean_iou:.4f}"
        )


if __name__ == "__main__":
    main()
