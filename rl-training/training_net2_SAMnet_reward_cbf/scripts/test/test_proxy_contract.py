import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
CFG_PATH = Path(__file__).resolve().parents[2] / "cfg" / "drone.yaml"
sys.path.insert(0, str(SCRIPTS_DIR))

from perception import PerceptionModule, compute_rayhitsdir


class TestProxyContract(unittest.TestCase):
    def make_cfg(self, interval=2):
        cfg = OmegaConf.load(CFG_PATH)
        cfg.perception.proxy_update_interval_steps = interval
        cfg.perception.proxy_near_recall = 1.0
        cfg.perception.proxy_far_recall = 1.0
        cfg.perception.proxy_recall_jitter = 0.0
        cfg.perception.proxy_false_positive_rate_min = 0.0
        cfg.perception.proxy_false_positive_rate_max = 0.0
        cfg.perception.proxy_velocity_noise_std_mps = [0.0, 0.0, 0.0]
        cfg.perception.proxy_velocity_noise_scale_min = 1.0
        cfg.perception.proxy_velocity_noise_scale_max = 1.0
        cfg.perception.proxy_velocity_bias_std_mps = 0.0
        cfg.perception.proxy_depth_noise_std_m = 0.0
        return cfg

    def make_inputs(self, cfg, batch=2, static_distance=8.0, dynamic_distance=2.0):
        directions = compute_rayhitsdir(
            "cpu",
            batch,
            cfg.sensor.lidar_hfov,
            cfg.sensor.lidar_vfov,
            cfg.sensor.lidar_hbeams,
            cfg.sensor.lidar_vbeams,
        )
        static = directions * static_distance
        dynamic = directions * cfg.sensor.lidar_range
        dynamic[:, 0] = directions[:, 0] * dynamic_distance
        is_dynamic = torch.zeros(
            batch, directions.shape[1], dtype=torch.bool
        )
        is_dynamic[:, 0] = True
        hit_index = torch.zeros_like(is_dynamic, dtype=torch.long)
        object_velocity = torch.tensor([[1.0, 0.0, 0.0]])
        position = torch.zeros(batch, 1, 3)
        quaternion = torch.zeros(batch, 1, 4)
        quaternion[..., 0] = 1.0
        ego_velocity = torch.zeros(batch, 1, 3)
        return (
            static,
            dynamic,
            is_dynamic,
            hit_index,
            object_velocity,
            position,
            quaternion,
            ego_velocity,
        )

    def test_shape_first_frame_and_scan_rate_cache(self):
        cfg = self.make_cfg(interval=2)
        proxy = PerceptionModule(cfg, 2, "cpu")
        inputs = self.make_inputs(cfg)

        static0, dynamic0 = proxy.process(*inputs)
        self.assertEqual(tuple(static0.shape), (2, 1, 6, 36))
        self.assertEqual(tuple(dynamic0.shape), (2, 5, 6, 36))
        self.assertTrue(torch.all(dynamic0[:, 0] == 1.0))
        self.assertTrue(torch.all(dynamic0[:, 1:] == 0.0))

        static1, dynamic1 = proxy.process(*inputs)
        self.assertTrue(torch.equal(static0, static1))
        self.assertTrue(torch.equal(dynamic0, dynamic1))

        _, dynamic2 = proxy.process(*inputs)
        self.assertLess(float(dynamic2[:, 0].min()), 1.0)
        self.assertTrue(torch.isfinite(dynamic2).all())

    def test_async_reset_neutralizes_only_reset_environment(self):
        cfg = self.make_cfg(interval=1)
        proxy = PerceptionModule(cfg, 2, "cpu")
        inputs = self.make_inputs(cfg)
        proxy.process(*inputs)
        proxy.process(*inputs)

        proxy.reset(torch.tensor([0]))
        _, dynamic = proxy.process(*inputs)
        self.assertTrue(torch.all(dynamic[0, 0] == 1.0))
        self.assertTrue(torch.all(dynamic[0, 1:] == 0.0))
        self.assertLess(float(dynamic[1, 0].min()), 1.0)

    def test_stationary_sphere_is_static_not_dynamic(self):
        cfg = self.make_cfg(interval=1)
        proxy = PerceptionModule(cfg, 2, "cpu")
        inputs = list(self.make_inputs(cfg))
        inputs[4] = torch.zeros(1, 3)
        proxy.process(*inputs)
        static, dynamic = proxy.process(*inputs)
        self.assertTrue(torch.all(dynamic[:, 0] == 1.0))
        self.assertLess(float(static.min()), 0.3)

    def test_static_surface_occludes_dynamic_sphere(self):
        cfg = self.make_cfg(interval=1)
        proxy = PerceptionModule(cfg, 2, "cpu")
        inputs = self.make_inputs(
            cfg, static_distance=1.0, dynamic_distance=2.0
        )
        proxy.process(*inputs)
        static, dynamic = proxy.process(*inputs)
        self.assertTrue(torch.all(dynamic[:, 0] == 1.0))
        self.assertLess(float(static.min()), 0.2)


if __name__ == "__main__":
    unittest.main()

