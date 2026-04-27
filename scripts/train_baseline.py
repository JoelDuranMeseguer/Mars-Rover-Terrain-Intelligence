"""Baseline mínimo de entrenamiento para segmentación en AI4Mars.

Conecta dataset + dataloader + modelo + loss en un flujo sencillo.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from mrti.data.dataset import AI4MarsSegmentationDataset

FIXED_CLASS_WEIGHTS = (0.668995, 0.504918, 2.050571, 27.003855)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple baseline training for segmentation")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/msl_ncam_v1"))
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap on train batches per epoch. Default: use all batches.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap on val batches per epoch. Default: use all batches.",
    )
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], default="unet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help='Optional comma-separated class weights, e.g. "0.66,0.50,2.05,27.0"',
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints") / "baseline_best.pt",
        help="Path to save the best checkpoint.",
    )
    parser.add_argument("--use-train-augmentations", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print batch loss every N batches (train/val).",
    )
    return parser.parse_args()


class FlatCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.bottleneck = ConvBlock(128, 256)

        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))

        up3 = F.interpolate(bottleneck, size=enc3.shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = F.interpolate(dec3, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.classifier(dec1)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "cnn":
        return FlatCNN(num_classes=num_classes)
    return SmallUNet(num_classes=num_classes)


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
    if args.class_weights is not None and not args.use_class_weights:
        raise ValueError("--class-weights requires --use-class-weights")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print("Seed:", args.seed)
    print("Model:", args.model)
    print("Use class weights:", args.use_class_weights)
    print("Use train augmentations:", args.use_train_augmentations)
    print("Max train batches:", args.max_train_batches)
    print("Max val batches:", args.max_val_batches)
    print("Num workers:", args.num_workers)
    print("Log every:", args.log_every)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = AI4MarsSegmentationDataset(
        dataset_root=args.dataset_root,
        split="train",
        use_augmentations=args.use_train_augmentations,
    )
    val_ds = AI4MarsSegmentationDataset(dataset_root=args.dataset_root, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = build_model(args.model, args.num_classes).to(device)
    class_weights = None
    if args.use_class_weights:
        if args.class_weights is not None:
            try:
                class_weights_values = [float(x.strip()) for x in args.class_weights.split(",")]
            except ValueError as exc:
                raise ValueError(
                    "--class-weights must be a comma-separated list of numeric values, "
                    'for example: "0.66,0.50,2.05,27.0"'
                ) from exc
            weights_source = "CLI (--class-weights)"
        else:
            class_weights_values = list(FIXED_CLASS_WEIGHTS)
            weights_source = "FIXED_CLASS_WEIGHTS"

        if len(class_weights_values) != args.num_classes:
            raise ValueError(
                f"Expected {args.num_classes} class weights but got "
                f"{len(class_weights_values)} from {weights_source}"
            )
        class_weights = torch.tensor(class_weights_values, dtype=torch.float32, device=device)
        class_weights_list = [round(x, 6) for x in class_weights.detach().cpu().tolist()]
        print("Class weights mode: enabled")
        print(f"Using class weights from {weights_source}:", class_weights_list)
    else:
        print("Class weights mode: disabled (uniform CE)")
    if args.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_epoch = -1
    best_mean_iou = -1.0
    best_iou_per_class = [0.0 for _ in range(args.num_classes)]
    best_checkpoint_path = Path(str(args.checkpoint_path).replace("\\", "/"))
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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
            if args.log_every > 0 and ((batch_idx + 1) % args.log_every == 0 or batch_idx == 0):
                print(
                    f"[epoch {epoch + 1}] train batch {batch_idx + 1} "
                    f"loss={loss.item():.4f}"
                )

            if args.max_train_batches is not None and (batch_idx + 1) >= args.max_train_batches:
                break

        model.eval()
        val_losses = []
        iou_intersections = torch.zeros(args.num_classes, dtype=torch.float64)
        iou_unions = torch.zeros(args.num_classes, dtype=torch.float64)
        target_pixel_counts = torch.zeros(args.num_classes, dtype=torch.float64)
        pred_pixel_counts = torch.zeros(args.num_classes, dtype=torch.float64)
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
                valid = masks != 255
                valid_targets = masks[valid]
                valid_preds = preds[valid]
                for cls in range(args.num_classes):
                    target_pixel_counts[cls] += (valid_targets == cls).sum().item()
                    pred_pixel_counts[cls] += (valid_preds == cls).sum().item()

                if args.log_every > 0 and ((batch_idx + 1) % args.log_every == 0 or batch_idx == 0):
                    print(
                        f"[epoch {epoch + 1}] val batch {batch_idx + 1} "
                        f"loss={loss.item():.4f}"
                    )

                if args.max_val_batches is not None and (batch_idx + 1) >= args.max_val_batches:
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
        if mean_iou > best_mean_iou:
            best_epoch = epoch + 1
            best_mean_iou = mean_iou
            best_iou_per_class = iou_per_class.copy()
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "best_mean_iou": best_mean_iou,
                    "model_name": args.model,
                    "use_class_weights": args.use_class_weights,
                },
                best_checkpoint_path,
            )
            print(
                f"Saved new best model checkpoint to {best_checkpoint_path} "
                f"(epoch={best_epoch}, mean_iou={best_mean_iou:.4f})"
            )
        for cls in range(args.num_classes):
            print(
                f"class_{cls} target_pixels={int(target_pixel_counts[cls].item())} "
                f"predicted_pixels={int(pred_pixel_counts[cls].item())}"
            )

    best_iou_text = ", ".join(
        [f"class_{cls}_iou={best_iou_per_class[cls]:.4f}" for cls in range(args.num_classes)]
    )
    print(
        f"Best validation epoch -> best_epoch={best_epoch}, "
        f"best_mean_iou={best_mean_iou:.4f}, {best_iou_text}"
    )
    print(f"Best checkpoint path: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
