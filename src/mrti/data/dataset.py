"""PyTorch dataset for AI4Mars semantic segmentation."""

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class AI4MarsSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        use_augmentations: bool = False,
    ) -> None:

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")
                
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.use_augmentations = use_augmentations and split == "train"

        self.index_path = self.dataset_root / "index.csv"
        self.split_path = self.dataset_root / "splits" / f"{split}.txt"

        if not self.index_path.exists():
            raise FileNotFoundError(f"index.csv not found: {self.index_path}")

        if not self.split_path.exists():
            raise FileNotFoundError(f"split file not found: {self.split_path}")

        split_ids = self._read_split_ids()
        rows_by_id = self._read_index_rows()

        self.samples = []
        missing_ids = []

        for sample_id in split_ids:
            row = rows_by_id.get(sample_id)

            if row is None:
                missing_ids.append(sample_id)
                continue

            image_path = self.dataset_root / row["image_path"]
            mask_path = self.dataset_root / row["mask_path"]

            self.samples.append(
                {
                    "id": sample_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                f"{len(missing_ids)} ids from split '{split}' were not found in index.csv. "
                f"Examples: {preview}"
            )

    def _read_split_ids(self) -> list[str]:
        ids = []

        with self.split_path.open("r", encoding="utf-8") as f:
            for line in f:
                sample_id = line.strip()
                if sample_id:
                    ids.append(sample_id)

        return ids

    def _read_index_rows(self) -> dict[str, dict[str, str]]:
        with self.index_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            required = {"id", "image_path", "mask_path"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"index.csv missing required columns: {', '.join(sorted(missing))}"
                )

            rows_by_id = {}

            for row in reader:
                sample_id = row["id"].strip()
                if sample_id:
                    rows_by_id[sample_id] = row

        return rows_by_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]

        sample_id = sample["id"]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for id={sample_id}: {image_path}")

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for id={sample_id}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.use_augmentations:
            image, mask = self._apply_train_augmentations(image, mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = self._image_to_tensor(image)
            mask = self._mask_to_tensor(mask)

        return {
            "image": image,
            "mask": mask,
            "id": sample_id,
        }

    def _apply_train_augmentations(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if torch.rand(1).item() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return image

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        mask = np.array(mask)

        # Some masks may be stored with 3 channels even though they encode class ids.
        # In that case, keep just one channel.
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = torch.from_numpy(mask.astype(np.int64)).contiguous()
        return mask
