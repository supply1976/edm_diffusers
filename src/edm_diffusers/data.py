"""Dataset and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    from datasets import Image as HFImage
    from datasets import load_dataset
except Exception:  # datasets is optional until needed
    HFImage = None
    load_dataset = None

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Pillow is required for image loading") from exc


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class PreprocessConfig:
    resize: int | None = None
    crop: str = "center"
    crop_size: int | None = None


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, extensions: Iterable[str], transform, channels: int):
        self.root = root
        exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.suffix.lower() in exts and path.is_file()
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = transform
        self.channels = channels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            if self.channels == 1:
                image = image.convert("L")
            else:
                image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, 0


class HFDatasetWrapper(Dataset):
    def __init__(self, dataset, image_column: str, transform, channels: int):
        self.dataset = dataset
        self.image_column = image_column
        self.transform = transform
        self.channels = channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        record = self.dataset[index]
        image = record[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if self.channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def _resolve_image_column(dataset, fallback: str | None) -> str:
    if fallback:
        return fallback
    if HFImage is None:
        return "image"
    for name, feature in dataset.features.items():
        if isinstance(feature, HFImage):
            return name
    return "image"


def build_transforms(
    preprocess: PreprocessConfig,
    image_size: int,
    channels: int,
) -> transforms.Compose:
    resize_size = preprocess.resize
    crop_size = preprocess.crop_size or image_size

    pipeline = []
    if resize_size:
        pipeline.append(
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC)
        )

    crop_mode = preprocess.crop.lower()
    if crop_mode == "center":
        pipeline.append(transforms.CenterCrop(crop_size))
    elif crop_mode == "random":
        pipeline.append(transforms.RandomCrop(crop_size))
    else:
        raise ValueError(f"Unknown crop mode: {preprocess.crop}")

    pipeline.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels),
        ]
    )

    return transforms.Compose(pipeline)


def build_dataset(
    dataset_config: dict,
    preprocess: PreprocessConfig,
    image_size: int,
    channels: int,
):
    transform = build_transforms(preprocess, image_size=image_size, channels=channels)

    dataset_type = dataset_config.get("type", "local")
    if dataset_type == "local":
        if "path" not in dataset_config:
            raise ValueError("dataset.path is required for local datasets")
        root = Path(dataset_config["path"])
        extensions = dataset_config.get("extensions", SUPPORTED_EXTENSIONS)
        return ImagePathDataset(root, extensions, transform, channels)

    if dataset_type == "hf":
        if load_dataset is None:
            raise RuntimeError("datasets is not installed; add it to your environment")

        if "name" not in dataset_config:
            raise ValueError("dataset.name is required for Hugging Face datasets")
        name = dataset_config["name"]
        config_name = dataset_config.get("config")
        split = dataset_config.get("split", "train")
        dataset = load_dataset(name, config_name, split=split)
        image_column = _resolve_image_column(dataset, dataset_config.get("image_column"))
        return HFDatasetWrapper(dataset, image_column, transform, channels)

    raise ValueError(f"Unknown dataset type: {dataset_type}")
