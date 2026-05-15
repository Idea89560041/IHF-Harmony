from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def build_vgg_encoder() -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(3, 3, 1),
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, 64, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 128, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 256, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 512, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


def channel_mean_std(features: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    flat = features.flatten(2)
    mean = flat.mean(dim=2)
    std = flat.var(dim=2, unbiased=False).add(eps).sqrt()
    return mean, std


def self_similarity(features: torch.Tensor) -> torch.Tensor:
    flat = features.flatten(2)
    flat = F.normalize(flat, dim=1)
    return torch.bmm(flat.transpose(1, 2), flat)


class VGGLoss(nn.Module):
    """VGG-based anatomy and artifact consistency objectives for IHF-Harmony."""

    def __init__(self, vgg_model: str, input_range: str = "zero_one") -> None:
        super().__init__()
        self.input_range = input_range
        encoder = build_vgg_encoder()

        model_path = Path(vgg_model)
        if not model_path.is_file():
            raise FileNotFoundError(f"VGG encoder weights not found: {model_path}")

        state = torch.load(str(model_path), map_location="cpu")
        encoder.load_state_dict(state)
        layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*layers[:4])
        self.enc_2 = nn.Sequential(*layers[4:11])
        self.enc_3 = nn.Sequential(*layers[11:18])
        self.enc_4 = nn.Sequential(*layers[18:31])

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def _to_vgg_range(self, image: torch.Tensor) -> torch.Tensor:
        if self.input_range == "minus_one_one":
            image = (image + 1.0) * 0.5
        return image.clamp(0.0, 1.0)

    def encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        image = self._to_vgg_range(image)
        outputs = []
        x = image
        for encoder in [self.enc_1, self.enc_2, self.enc_3, self.enc_4]:
            x = encoder(x)
            outputs.append(x)
        return outputs

    @staticmethod
    def anatomy_loss(output_features: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self_similarity(output_features), self_similarity(content_features))

    @staticmethod
    def artifact_loss(
        output_features: torch.Tensor,
        target_features: torch.Tensor,
        keep_ratio: float = 0.9,
    ) -> torch.Tensor:
        out_mean, out_std = channel_mean_std(output_features)
        target_mean, target_std = channel_mean_std(target_features)

        channel_distance = (out_mean - target_mean).pow(2)
        channels = channel_distance.size(1)
        keep = max(1, min(channels, int(round(channels * keep_ratio))))
        selected = channel_distance.topk(keep, dim=1, largest=False).indices

        def gather_channels(values: torch.Tensor) -> torch.Tensor:
            return values.gather(1, selected)

        mean_loss = F.mse_loss(gather_channels(out_mean), gather_channels(target_mean))
        std_loss = F.mse_loss(gather_channels(out_std), gather_channels(target_std))
        return mean_loss + std_loss

    def forward(
        self,
        content_images: torch.Tensor,
        target_images: torch.Tensor,
        harmonized_images: torch.Tensor,
        keep_ratio: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            content_features = self.encode(content_images)
            target_features = self.encode(target_images)

        output_features = self.encode(harmonized_images)
        loss_anatomy = self.anatomy_loss(output_features[-1], content_features[-1])
        loss_artifact = sum(
            self.artifact_loss(output_features[idx], target_features[idx], keep_ratio)
            for idx in range(3)
        )
        return loss_anatomy, loss_artifact
