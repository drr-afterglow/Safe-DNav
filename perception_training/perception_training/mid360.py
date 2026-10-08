from dataclasses import dataclass
from typing import Tuple

from .geometry import PanoramaSpec


@dataclass
class Mid360Spec:
    """Livox MID-360 style lidar approximation for Isaac data generation.

    MID-360 is a non-repetitive scanning lidar. Data collection replaces
    Orbit's fixed bootstrap rays on every capture with a temporally varying
    full-FOV low-discrepancy sequence. Raw returns are encoded into the
    high-resolution network panorama before postprocessing downsamples network
    predictions to the compact RL panorama.
    """

    max_range: float = 10.0
    horizontal_fov_deg: float = 360.0
    vertical_fov_deg: Tuple[float, float] = (-7.0, 52.0)
    nominal_point_rate_hz: int = 200000
    scan_duration: float = 0.096
    simulated_points_per_frame: int = 5760
    network_panorama_height: int = 16
    network_panorama_width: int = 180
    rl_panorama_height: int = 6
    rl_panorama_width: int = 36

    @property
    def network_panorama(self) -> PanoramaSpec:
        return PanoramaSpec(
            height=self.network_panorama_height,
            width=self.network_panorama_width,
            max_range=self.max_range,
            vfov_deg=self.vertical_fov_deg,
        )

    @property
    def rl_panorama(self) -> PanoramaSpec:
        return PanoramaSpec(
            height=self.rl_panorama_height,
            width=self.rl_panorama_width,
            max_range=self.max_range,
            vfov_deg=self.vertical_fov_deg,
        )

    @property
    def panorama(self) -> PanoramaSpec:
        return self.network_panorama

    @property
    def nominal_points_per_frame(self) -> int:
        return int(round(self.nominal_point_rate_hz * self.scan_duration))
