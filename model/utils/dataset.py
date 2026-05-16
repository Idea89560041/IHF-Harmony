from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def resolve_path(root: str, item: str) -> Path:
    path = Path(item.strip())
    if path.is_absolute():
        return path
    return Path(root) / path


def read_list(list_path: str, root: str) -> list[Path]:
    with open(list_path, "r", encoding="utf-8") as handle:
        paths = [resolve_path(root, line) for line in handle if line.strip()]
    if not paths:
        raise ValueError(f"No samples were found in list file: {list_path}")
    return paths


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def build_transform(cfg: Any, intensity_range: str) -> transforms.Compose:
    options = set(getattr(cfg, "transform", []))
    height, width = int(cfg.height), int(cfg.width)
    scale = (float(cfg.scale_l), float(cfg.scale_h))
    pipeline: list = []

    if "random_resized_crop" in options:
        pipeline.append(
            transforms.RandomResizedCrop(
                (height, width),
                scale=scale,
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            )
        )
    else:
        pipeline.append(transforms.Resize((height, width), interpolation=InterpolationMode.BILINEAR))

    if "crop" in options:
        pipeline.append(transforms.RandomCrop((height, width)))
    if "h_flip" in options:
        pipeline.append(transforms.RandomHorizontalFlip(p=0.5))
    if "v_flip" in options:
        pipeline.append(transforms.RandomVerticalFlip(p=0.5))

    pipeline.append(transforms.ToTensor())
    if intensity_range == "minus_one_one" or "normalize" in options:
        pipeline.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(pipeline)


class MRIPairDataset(Dataset):
    """Unpaired source-target MRI slice dataset."""

    def __init__(self, cfg: Any, intensity_range: str = "zero_one") -> None:
        self.source_paths = read_list(cfg.source_list, cfg.source_root)
        self.target_paths = read_list(cfg.target_list, cfg.target_root)
        self.random_pair = bool(cfg.random_pair)
        self.return_name = bool(cfg.return_name)
        self.transform = build_transform(cfg, intensity_range)

    def __len__(self) -> int:
        return len(self.source_paths)

    def __getitem__(self, index: int):
        target_index = random.randrange(len(self.target_paths)) if self.random_pair else index % len(self.target_paths)
        source_path = self.source_paths[index]
        target_path = self.target_paths[target_index]

        source = self.transform(load_rgb(source_path))
        target = self.transform(load_rgb(target_path))

        if not self.return_name:
            return source, target
        name = f"{source_path.stem}_to_{target_path.stem}.png"
        return source, target, name


def get_dataset(cfg: Any, intensity_range: str = "zero_one") -> MRIPairDataset:
    return MRIPairDataset(cfg, intensity_range=intensity_range)
