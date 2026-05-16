from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def channel_stats(tensor: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    flat = tensor.flatten(2)
    mean = flat.mean(dim=2).view(tensor.size(0), tensor.size(1), 1, 1)
    std = flat.var(dim=2, unbiased=False).add(eps).sqrt().view_as(mean)
    return mean, std


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: bool = True,
        activation: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArtifactEncoder(nn.Module):
    """Compact target-domain encoder used to condition artifact-aware normalization."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, style_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=7, padding=3, norm=False),
            ConvBlock(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            ConvBlock(base_channels * 4, base_channels * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(base_channels * 4, style_dim)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        pooled = self.features(target).flatten(1)
        return self.proj(pooled)


class ArtifactAwareNorm(nn.Module):
    """Normalize source features and modulate them with target artifact statistics."""

    def __init__(self, channels: int, style_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or max(style_dim, min(channels, 256))
        self.anatomy_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
        )
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.to_scale = nn.Linear(hidden * 2, channels)
        self.to_bias = nn.Linear(hidden * 2, channels)

        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)

    def forward(self, features: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        mean, std = channel_stats(features)
        normalized = (features - mean) / std

        condition = torch.cat([self.anatomy_proj(features), self.style_proj(style_code)], dim=1)
        scale = 1.0 + self.to_scale(condition).view(features.size(0), features.size(1), 1, 1)
        bias = self.to_bias(condition).view_as(scale)
        return normalized * scale + bias


class InvertibleHierarchyBlock(nn.Module):
    """Hierarchical subtractive coupling followed by additive reconstruction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        hidden_channels: int | None = None,
        learn_fusion: bool = True,
    ) -> None:
        super().__init__()
        if out_channels % in_channels != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by in_channels ({in_channels})."
            )

        hidden = hidden_channels or max(in_channels * 2, min(out_channels, 256))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_splits = out_channels // in_channels
        self.delta_net = nn.Sequential(
            ConvBlock(in_channels, hidden),
            ConvBlock(hidden, hidden),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )
        self.aan = ArtifactAwareNorm(out_channels, style_dim)
        self.fusion_logit = nn.Parameter(torch.zeros(1), requires_grad=learn_fusion)
        self._cached_delta: torch.Tensor | None = None

    @property
    def fusion_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.fusion_logit)

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        delta = self.delta_net(source)
        self._cached_delta = delta

        states = []
        current = source
        for delta_i in delta.chunk(self.num_splits, dim=1):
            current = current - delta_i
            states.append(current)
        return torch.cat(states, dim=1)

    def decode(self, latent: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        if self._cached_delta is None:
            raise RuntimeError("decode() was called before encode(); call the full model forward().")

        latent = self.aan(latent, style_code)
        latent_parts = latent.chunk(self.num_splits, dim=1)
        delta_parts = self._cached_delta.chunk(self.num_splits, dim=1)

        recovered = latent_parts[-1] + delta_parts[-1]
        alpha = self.fusion_weight
        for idx in range(self.num_splits - 2, -1, -1):
            candidate = latent_parts[idx] + delta_parts[idx]
            recovered = alpha * candidate + (1.0 - alpha) * recovered
        return recovered


class IHFHarmony(nn.Module):
    """Invertible Hierarchy Flow model for unpaired MRI harmonization."""

    def __init__(
        self,
        pad_size: int = 10,
        in_channel: int = 3,
        out_channels: Iterable[int] = (30, 120, 480),
        weight_type: str = "learned",
        style_dim: int = 64,
        style_base_channels: int = 32,
        use_squeeze: bool = False,
    ) -> None:
        super().__init__()
        self.pad_size = pad_size
        self.in_channel = in_channel
        self.use_squeeze = use_squeeze
        self.padding = nn.ReflectionPad2d(pad_size) if pad_size > 0 else nn.Identity()
        self.artifact_encoder = ArtifactEncoder(in_channel, style_base_channels, style_dim)

        learn_fusion = weight_type == "learned"
        self.blocks = nn.ModuleList()
        current_channels = in_channel
        out_channels = list(out_channels)
        for idx, block_channels in enumerate(out_channels):
            self.blocks.append(
                InvertibleHierarchyBlock(
                    in_channels=current_channels,
                    out_channels=block_channels,
                    style_dim=style_dim,
                    learn_fusion=learn_fusion,
                )
            )
            current_channels = block_channels
            if self.use_squeeze and idx < len(out_channels) - 1:
                current_channels *= 4

    def forward(self, content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        style_code = self.artifact_encoder(target)
        x = self.padding(content)
        _, _, padded_h, padded_w = x.shape

        for idx, block in enumerate(self.blocks):
            x = block.encode(x)
            if self.use_squeeze and idx < len(self.blocks) - 1:
                x = F.pixel_unshuffle(x, downscale_factor=2)

        for idx in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[idx]
            x = block.decode(x, style_code)
            if self.use_squeeze and idx > 0:
                x = F.pixel_shuffle(x, upscale_factor=2)

        if self.pad_size > 0:
            x = x[:, :, self.pad_size : padded_h - self.pad_size, self.pad_size : padded_w - self.pad_size]
        return x


# Backward-compatible name used by older scripts and checkpoints.
HierarchyFlow = IHFHarmony
