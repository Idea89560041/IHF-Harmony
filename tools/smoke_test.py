from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import as_config
from model.losses import VGGLoss
from model.network import IHFHarmony
from model.utils.dataset import get_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal IHF-Harmony smoke test")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = as_config(yaml.safe_load(handle))

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = get_dataset(cfg.dataset.train, intensity_range=cfg.dataset.intensity_range)
    content, target, _ = dataset[0]
    content = content.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    model = IHFHarmony(
        pad_size=cfg.network.pad_size,
        in_channel=cfg.network.in_channel,
        out_channels=cfg.network.out_channels,
        weight_type=cfg.network.weight_type,
        style_dim=cfg.network.style_dim,
        style_base_channels=cfg.network.style_base_channels,
        use_squeeze=getattr(cfg.network, "use_squeeze", False),
    ).to(device)
    criterion = VGGLoss(cfg.loss.vgg_encoder, input_range=cfg.dataset.intensity_range).to(device)

    output = model(content, target)
    anatomy_loss, artifact_loss = criterion(content, target, output, cfg.loss.keep_ratio)
    total = cfg.loss.anatomy_weight * anatomy_loss + cfg.loss.artifact_weight * artifact_loss
    total.backward()

    print(f"device={device}")
    print(f"dataset_size={len(dataset)}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"loss={total.item():.6f}")
    print(f"config={Path(args.config).resolve()}")


if __name__ == "__main__":
    main()
