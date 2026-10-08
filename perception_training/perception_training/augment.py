# Sensor-aware augmentation for four-channel lidar pairs.
from typing import Dict

import torch


class LidarAugment:
    def __init__(self, beam_dropout: float = 0.1, range_noise_std: float = 0.02, max_range_norm: float = 1.0):
        self.beam_dropout = float(beam_dropout)
        self.range_noise_std = float(range_noise_std)
        self.max_range_norm = float(max_range_norm)
        if not 0.0 <= self.beam_dropout < 1.0:
            raise ValueError("beam_dropout must be in [0, 1)")

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input"]
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError("four-channel input [curr_range, prev_range, curr_valid, prev_valid] is required")
        ranges = x[:, :2]
        valid = (x[:, 2:4] > 0.5).to(ranges.dtype)
        if self.range_noise_std > 0:
            noise = torch.randn_like(ranges) * self.range_noise_std
            ranges = torch.where(valid.bool(), ranges + noise, ranges)
        if self.beam_dropout > 0:
            # Draw one independent dropout mask per temporal frame.
            keep = (torch.rand_like(valid) > self.beam_dropout).to(valid.dtype)
            valid = valid * keep
        ranges = torch.where(valid.bool(), ranges, torch.full_like(ranges, self.max_range_norm))
        out = dict(batch)
        out["input"] = torch.cat([ranges.clamp(0.0, self.max_range_norm), valid], dim=1)
        return out
