"""Dataset utilities for AI4Mars segmentation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - handled at runtime in environments without torch
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback base class when torch is not installed."""


def list_images(root: str, suffix: str = ".png") -> list[Path]:
    """Recursively list image files."""
    return sorted(Path(root).rglob(f"*{suffix}"))


class AI4MarsSegmentationDataset(Dataset):
    """Simple PyTorch dataset reading index.csv + split files.

    Expected structure inside ``dataset_root``:

    - index.csv (columns: id, image_path, mask_path)
    - splits/train.txt, splits/val.txt, splits/test.txt
    - images/...
    - masks/...
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "AI4MarsSegmentationDataset needs PyTorch. Install torch first, e.g. `pip install torch`."
            )

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform

        self.index_path = self.dataset_root / "index.csv"
        self.split_path = self.dataset_root / "splits" / f"{split}.txt"

        if not self.index_path.exists():
            raise FileNotFoundError(f"index.csv not found: {self.index_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"split file not found: {self.split_path}")

        split_ids = self._read_split_ids(self.split_path)
        rows_by_id = self._read_index_rows(self.index_path)

        self.samples: list[dict[str, Path | str]] = []
        missing_ids: list[str] = []

        for sample_id in split_ids:
            row = rows_by_id.get(sample_id)
            if row is None:
                missing_ids.append(sample_id)
                continue

            image_path = self.dataset_root / row["image_path"]
            mask_path = self.dataset_root / row["mask_path"]
            self.samples.append({"id": sample_id, "image_path": image_path, "mask_path": mask_path})

        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                f"{len(missing_ids)} ids from split '{split}' were not found in index.csv. "
                f"Examples: {preview}"
            )

    @staticmethod
    def _read_split_ids(split_path: Path) -> list[str]:
        ids: list[str] = []
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                sample_id = line.strip()
                if sample_id:
                    ids.append(sample_id)
        return ids

    @staticmethod
    def _read_index_rows(index_path: Path) -> dict[str, dict[str, str]]:
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "image_path", "mask_path"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(f"index.csv missing required columns: {missing_str}")

            rows_by_id: dict[str, dict[str, str]] = {}
            for row in reader:
                sample_id = row["id"].strip()
                if sample_id:
                    rows_by_id[sample_id] = row
            return rows_by_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        sample_id = str(sample["id"])
        image_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"image not found for id={sample_id}: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"mask not found for id={sample_id}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            image_tensor = self._image_to_tensor(image)
            mask_tensor = self._mask_to_tensor(mask)

        return {"image": image_tensor, "mask": mask_tensor, "id": sample_id}

    @staticmethod
    def _image_to_tensor(image: Image.Image):
        image_np = np.array(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

    @staticmethod
    def _mask_to_tensor(mask: Image.Image):
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        return torch.from_numpy(mask_np.astype(np.int64)).contiguous()
