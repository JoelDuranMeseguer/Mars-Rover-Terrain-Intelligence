"""Simple sanity check for the AI4Mars dataset."""

from pathlib import Path

from mrti.data.dataset import AI4MarsSegmentationDataset


data_root = Path("/home/joel/Documentos/Mars-Rover-Terrain-Intelligence/data/processed/msl_ncam_v1")
split = "train"
num_samples = 3

dataset = AI4MarsSegmentationDataset(
    dataset_root=data_root,
    split=split,
)

print("Dataset root:", data_root)
print("Split:", split)
print("Samples:", len(dataset))

n = min(num_samples, len(dataset))

for i in range(n):
    sample = dataset[i]
    image = sample["image"]
    mask = sample["mask"]
    sample_id = sample["id"]

    print(
        f"[{i}] id={sample_id} "
        f"image_shape={tuple(image.shape)} image_dtype={image.dtype} "
        f"mask_shape={tuple(mask.shape)} mask_dtype={mask.dtype}"
    )