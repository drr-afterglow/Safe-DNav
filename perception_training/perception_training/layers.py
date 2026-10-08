import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class HorizontalCircularPad2d(nn.Module):
    """Circular padding in azimuth and zero padding at the pitch limits."""

    def __init__(self, padding: Tuple[int, int]):
        super().__init__()
        self.pad_h, self.pad_w = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="constant", value=0.0)
        return x


class CircularConv2d(nn.Module):
    """Conv2d whose spatial padding respects the panoramic seam."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("CircularConv2d requires an odd kernel size")
        pad = kernel_size // 2
        self.pad = HorizontalCircularPad2d((pad, pad))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class CircularConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = CircularConv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class PitchModulatedDeformConv2d(nn.Module):
    """DCNv2 with offsets conditioned explicitly on each row's physical pitch.

    Learned content offsets and pitch offsets are bounded so the additional
    circular guard band always covers deformable samples across the panorama
    seam. The modulation mask is the DCNv2 amplitude modulation term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pitch: torch.Tensor,
        kernel_size: int = 3,
        max_offset_px: float = 0.75,
        pitch_hidden: int = 32,
        bias: bool = False,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("PitchModulatedDeformConv2d requires an odd kernel size")
        self.kernel_size = kernel_size
        self.kernel_points = kernel_size * kernel_size
        self.radius = kernel_size // 2
        self.guard = int(math.ceil(max_offset_px))
        self.max_offset_px = float(max_offset_px)
        self.register_buffer("pitch", pitch.float(), persistent=True)

        # This predictor is itself an HCP convolution, as required for every
        # spatial Conv2d in the panoramic backbone.
        self.offset_mask = CircularConv2d(
            in_channels,
            3 * self.kernel_points,
            kernel_size=kernel_size,
            bias=True,
        )
        self.pitch_offset = nn.Sequential(
            nn.Linear(2, pitch_hidden),
            nn.SiLU(),
            nn.Linear(pitch_hidden, 2 * self.kernel_points),
        )
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        # Keep the no-bias placeholder as a non-persistent buffer.  Unlike a
        # tensor created while tracing, buffers follow module device changes;
        # non-persistent keeps existing checkpoint state_dicts compatible.
        self.register_buffer("zero_bias", torch.zeros(out_channels), persistent=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.offset_mask.conv.weight)
        nn.init.zeros_(self.offset_mask.conv.bias)
        nn.init.zeros_(self.pitch_offset[-1].weight)
        nn.init.zeros_(self.pitch_offset[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.offset_mask(x)
        content_offset, modulation = torch.split(
            offset_mask,
            [2 * self.kernel_points, self.kernel_points],
            dim=1,
        )

        pitch_embedding = torch.stack([torch.sin(self.pitch), torch.cos(self.pitch)], dim=-1)
        pitch_offset = self.pitch_offset(pitch_embedding).transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        offset = self.max_offset_px * torch.tanh(content_offset + pitch_offset)
        modulation = torch.sigmoid(modulation)

        # One extra guard pixel lets bounded offsets cross the physical seam
        # while sampling circular values instead of implicit zeros.
        if self.guard > 0:
            offset = F.pad(offset, (self.guard, self.guard, 0, 0), mode="circular")
            offset = F.pad(offset, (0, 0, self.guard, self.guard), mode="constant", value=0.0)
            modulation = F.pad(modulation, (self.guard, self.guard, 0, 0), mode="circular")
            modulation = F.pad(
                modulation,
                (0, 0, self.guard, self.guard),
                mode="constant",
                value=0.0,
            )
        total_pad = self.radius + self.guard
        padded = F.pad(x, (total_pad, total_pad, 0, 0), mode="circular")
        padded = F.pad(
            padded,
            (0, 0, total_pad, total_pad),
            mode="constant",
            value=0.0,
        )
        # torchvision accepts ``None`` here, but tracing that branch embeds a
        # CPU zero tensor in the graph. Use a module buffer so exported models
        # remain device-agnostic.
        bias = self.bias if self.bias is not None else self.zero_bias
        out = deform_conv2d(
            padded,
            offset,
            self.weight,
            bias,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            mask=modulation,
        )
        if self.guard > 0:
            out = out[..., self.guard : -self.guard, self.guard : -self.guard]
        return out


class SphericalConvBlock(nn.Module):
    """Pitch-modulated DCNv2 + HCP block used at the outer U-Net level."""

    def __init__(self, in_channels: int, out_channels: int, pitch: torch.Tensor, kernel_size: int = 3):
        super().__init__()
        self.conv = PitchModulatedDeformConv2d(
            in_channels,
            out_channels,
            pitch,
            kernel_size=kernel_size,
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))
