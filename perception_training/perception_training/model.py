import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CircularConvBlock, SphericalConvBlock


def pitch_grid(height: int, vfov_deg: Tuple[float, float]) -> torch.Tensor:
    # Panorama row zero corresponds to alpha_max in the projection equation.
    return torch.linspace(math.radians(vfov_deg[1]), math.radians(vfov_deg[0]), height)


class StandardConvPair(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            CircularConvBlock(in_channels, out_channels),
            CircularConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SphericalConvPair(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pitch: torch.Tensor):
        super().__init__()
        self.net = nn.Sequential(
            SphericalConvBlock(in_channels, out_channels, pitch),
            SphericalConvBlock(out_channels, out_channels, pitch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        pitch: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        merged_channels = in_channels + skip_channels
        self.conv = (
            StandardConvPair(merged_channels, out_channels)
            if pitch is None
            else SphericalConvPair(merged_channels, out_channels, pitch)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SAUMNet(nn.Module):
    """Final SA-UMNet encoder-decoder from the perception design.

    Input contains current/aligned-previous normalized ranges and their
    explicit return-valid maps, with shape ``(B, 4, 16, 180)``.
    ``delta_t`` is the actual interval between those frames. The velocity head
    is normalized at ``reference_delta_t`` and scales inversely with the actual
    interval to preserve physical velocity units under sampling jitter.
    """

    def __init__(
        self,
        height: int = 16,
        width: int = 180,
        vfov_deg: Tuple[float, float] = (-7.0, 52.0),
        base_channels: int = 16,
        input_channels: int = 4,
        reference_delta_t: float = 0.096,
        min_delta_t: float = 0.05,
        max_delta_t: float = 0.15,
    ):
        super().__init__()
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError("SA-UMNet height and width must be divisible by 4")
        if base_channels != 16:
            raise ValueError("The final SA-UMNet design fixes base_channels=16 and bottleneck_channels=64")
        if input_channels != 4:
            raise ValueError("The frozen SA-UMNet input contract requires four channels")
        if reference_delta_t <= 0:
            raise ValueError("reference_delta_t must be positive")
        if not 0 < min_delta_t <= reference_delta_t <= max_delta_t:
            raise ValueError("delta_t bounds must satisfy 0 < min <= reference <= max")

        self.height = height
        self.width = width
        self.vfov_deg = tuple(vfov_deg)
        self.input_channels = int(input_channels)
        self.reference_delta_t = float(reference_delta_t)
        self.min_delta_t = float(min_delta_t)
        self.max_delta_t = float(max_delta_t)
        pitch = pitch_grid(height, self.vfov_deg)

        self.enc1 = SphericalConvPair(self.input_channels, 16, pitch)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = StandardConvPair(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = StandardConvPair(32, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.up2 = UpBlock(32, 16, 16, pitch=pitch)

        self.semantic_head = nn.Conv2d(16, 1, kernel_size=1)
        self.kinematic_head = nn.Conv2d(16, 3, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, delta_t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool1(skip1))
        feature = self.bottleneck(self.pool2(skip2))
        feature = self.up1(feature, skip2)
        feature = self.up2(feature, skip1)

        velocity = self.kinematic_head(feature)
        log_sigma = self.uncertainty_head(feature)
        if delta_t is not None:
            dt = delta_t.to(device=velocity.device, dtype=velocity.dtype).reshape(-1, 1, 1, 1)
            # Dataset/runtime reject invalid values. Clamping here also keeps a
            # directly invoked or exported model numerically safe.
            dt = dt.clamp(min=self.min_delta_t, max=self.max_delta_t)
            # Keep the tensor as the left operand. In PyTorch 2.0 tracing,
            # ``python_float / tensor`` becomes a CPU tensor constant and
            # breaks CUDA deployment of the serialized model.
            time_scale = torch.reciprocal(dt).mul(self.reference_delta_t)
            velocity = velocity * time_scale
            # sigma is a velocity standard deviation, so it obeys the same
            # inverse-time conversion as the velocity estimate.
            log_sigma = log_sigma + torch.log(time_scale)
        return {
            "mask_logits": self.semantic_head(feature),
            "velocity": velocity,
            "log_sigma": log_sigma.clamp(-5.0, 3.0),
        }
