"""Perception-error-randomized five-channel proxy for PPO training."""

import torch
from omni_drones.utils.torch import quat_rotate_inverse


class PerceptionModule:
    """Mimic the frozen SA-UMNet RL contract without running the network."""

    def __init__(self, cfg, num_envs, device):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.d_max = float(cfg.sensor.lidar_range)
        self.H = int(cfg.sensor.lidar_vbeams)
        self.W = int(cfg.sensor.lidar_hbeams)

        v_min = max(-89.0, float(cfg.sensor.lidar_vfov[0]))
        v_max = min(89.0, float(cfg.sensor.lidar_vfov[1]))
        self.fov_v_rad = torch.tensor(
            [v_min, v_max], device=self.device
        ) * torch.pi / 180.0
        self.fov_h_rad = torch.tensor(
            [-180.0, 180.0], device=self.device
        ) * torch.pi / 180.0
        self.res_h = (self.fov_h_rad[1] - self.fov_h_rad[0]) / self.W
        self.res_v = (self.fov_v_rad[1] - self.fov_v_rad[0]) / max(self.H - 1, 1)

        pcfg = cfg.perception
        self.dyn_max_vel = float(pcfg.dyn_max_vel_mps)
        self.uncertainty_scale = float(pcfg.uncertainty_scale_mps)
        self.update_interval = int(
            getattr(pcfg, "proxy_update_interval_steps", 1)
        )
        self.near_recall = float(getattr(pcfg, "proxy_near_recall", 1.0))
        self.far_recall = float(getattr(pcfg, "proxy_far_recall", 1.0))
        self.recall_jitter = float(getattr(pcfg, "proxy_recall_jitter", 0.0))
        self.fpr_min = float(
            getattr(pcfg, "proxy_false_positive_rate_min", 0.0)
        )
        self.fpr_max = float(
            getattr(pcfg, "proxy_false_positive_rate_max", 0.0)
        )
        velocity_std = getattr(
            pcfg, "proxy_velocity_noise_std_mps", [0.0, 0.0, 0.0]
        )
        if isinstance(velocity_std, (int, float)):
            velocity_std = [float(velocity_std)] * 3
        if len(velocity_std) != 3:
            raise ValueError("proxy_velocity_noise_std_mps must contain 3 values")
        self.velocity_std = torch.tensor(
            velocity_std, device=self.device
        ).reshape(1, 1, 3)
        self.noise_scale_min = float(
            getattr(pcfg, "proxy_velocity_noise_scale_min", 1.0)
        )
        self.noise_scale_max = float(
            getattr(pcfg, "proxy_velocity_noise_scale_max", 1.0)
        )
        self.velocity_bias_std = float(
            getattr(pcfg, "proxy_velocity_bias_std_mps", 0.0)
        )
        self.fp_velocity_std = float(
            getattr(pcfg, "proxy_false_positive_velocity_std_mps", 0.0)
        )
        self.sigma_floor = float(
            getattr(pcfg, "proxy_sigma_floor_mps", 0.1)
        )
        self.sigma_error_gain = float(
            getattr(pcfg, "proxy_sigma_error_gain", 1.0)
        )
        self.sigma_max = float(
            getattr(pcfg, "proxy_sigma_max_mps", 3.0)
        )
        self.depth_noise_std = float(
            getattr(pcfg, "proxy_depth_noise_std_m", 0.0)
        )
        self.min_dynamic_speed = float(
            getattr(pcfg, "proxy_min_dynamic_speed_mps", 0.05)
        )
        self.occlusion_epsilon = float(
            getattr(pcfg, "proxy_occlusion_epsilon_m", 0.01)
        )
        self._validate_config()

        self.step_count = 0
        self.history_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.cached_static = torch.ones(
            self.num_envs, 1, self.H, self.W, device=self.device
        )
        self.cached_dynamic = torch.zeros(
            self.num_envs, 5, self.H, self.W, device=self.device
        )
        self.cached_dynamic[:, 0] = 1.0

    def _validate_config(self):
        if self.update_interval <= 0:
            raise ValueError("proxy_update_interval_steps must be positive")
        for name, value in (
            ("near recall", self.near_recall),
            ("far recall", self.far_recall),
            ("minimum FPR", self.fpr_min),
            ("maximum FPR", self.fpr_max),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"proxy {name} must be in [0, 1]")
        if self.fpr_min > self.fpr_max:
            raise ValueError("proxy FPR bounds are reversed")
        if self.noise_scale_min <= 0.0:
            raise ValueError("proxy velocity noise scale must be positive")
        if self.noise_scale_min > self.noise_scale_max:
            raise ValueError("proxy velocity noise scale bounds are reversed")

    def _to_body(self, points_w, pos_w, quat_w):
        batch, num_points, _ = points_w.shape
        if quat_w.ndim == 2:
            quat_w = quat_w.unsqueeze(1)
        return quat_rotate_inverse(
            quat_w.expand(batch, num_points, 4),
            points_w - pos_w,
        )

    def _pixel_indices(self, points):
        x, y, z = points.unbind(dim=-1)
        distance = torch.linalg.norm(points, dim=-1)
        theta_h = torch.atan2(y, x)
        theta_v = torch.asin(
            (z / distance.clamp_min(1e-6)).clamp(-1.0, 1.0)
        )
        u = ((theta_h - self.fov_h_rad[0]) / self.res_h).long()
        v = ((theta_v - self.fov_v_rad[0]) / self.res_v).long()
        return (
            v.clamp(0, self.H - 1) * self.W + u.clamp(0, self.W - 1),
            distance,
        )

    def _project_depth(self, points, valid):
        batch = points.shape[0]
        indices, distance = self._pixel_indices(points)
        valid = valid & (distance > 0.1) & (distance < self.d_max)
        values = torch.where(
            valid, distance, torch.full_like(distance, float("inf"))
        )
        image = torch.full(
            (batch, self.H * self.W), self.d_max, device=self.device
        )
        image.scatter_reduce_(
            1, indices, values, reduce="amin", include_self=True
        )
        return image.clamp(max=self.d_max).reshape(
            batch, 1, self.H, self.W
        )

    def _project_feature(self, points, valid, feature):
        batch = points.shape[0]
        indices, distance = self._pixel_indices(points)
        valid = valid & (distance > 0.1) & (distance < self.d_max)
        safe_indices = torch.where(
            valid, indices, torch.full_like(indices, self.H * self.W)
        )
        values = torch.where(valid, feature, torch.zeros_like(feature))
        image = torch.zeros(
            batch, self.H * self.W + 1,
            device=self.device,
            dtype=values.dtype,
        )
        image.scatter_(1, safe_indices, values)
        return image[:, :-1].reshape(batch, 1, self.H, self.W)

    def reset(self, env_ids, pos=None, rot=None):
        """Reset temporal state for only the asynchronously reset envs."""
        del pos, rot
        env_ids = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        if env_ids.numel() == 0:
            return
        self.history_valid[env_ids] = False
        self.cached_static[env_ids] = 1.0
        self.cached_dynamic[env_ids] = 0.0
        self.cached_dynamic[env_ids, 0] = 1.0

    def _sample_mask(self, true_dynamic, depth, valid):
        batch = true_dynamic.shape[0]
        recall = self.near_recall + (
            self.far_recall - self.near_recall
        ) * (depth / self.d_max).clamp(0.0, 1.0)
        jitter = (
            torch.rand(batch, 1, device=self.device) * 2.0 - 1.0
        ) * self.recall_jitter
        recall = (recall + jitter).clamp(0.0, 1.0)
        true_positive = true_dynamic & (
            torch.rand_like(recall) < recall
        )
        fpr = torch.empty(batch, 1, device=self.device).uniform_(
            self.fpr_min, self.fpr_max
        )
        false_positive = (
            (~true_dynamic)
            & valid
            & (torch.rand_like(depth) < fpr)
        )
        return true_positive | false_positive, false_positive

    def _compute_update(
        self,
        hits_static_w,
        hits_dyn_w,
        is_dyn_hit,
        dyn_hit_idx,
        all_dyn_vel_w,
        pos_w,
        quat_w,
        vel_w,
    ):
        batch, num_rays, _ = hits_static_w.shape
        static_body = self._to_body(hits_static_w, pos_w, quat_w)
        dynamic_body = self._to_body(hits_dyn_w, pos_w, quat_w)
        static_distance = torch.linalg.norm(static_body, dim=-1)
        dynamic_distance = torch.linalg.norm(dynamic_body, dim=-1)
        static_valid = (static_distance > 0.1) & (
            static_distance < self.d_max
        )
        dynamic_valid = is_dyn_hit & (dynamic_distance > 0.1) & (
            dynamic_distance < self.d_max
        )

        if all_dyn_vel_w.shape[0] == 0:
            object_velocity = torch.zeros(
                batch, num_rays, 3, device=self.device
            )
        else:
            safe_idx = dyn_hit_idx.clamp(
                0, all_dyn_vel_w.shape[0] - 1
            )
            object_velocity = all_dyn_vel_w[safe_idx]

        moving = torch.linalg.norm(
            object_velocity, dim=-1
        ) >= self.min_dynamic_speed
        visible_dynamic = (
            dynamic_valid
            & moving
            & (
                (~static_valid)
                | (
                    dynamic_distance + self.occlusion_epsilon
                    < static_distance
                )
            )
        )
        dynamic_nearer = dynamic_valid & (
            (~static_valid) | (dynamic_distance < static_distance)
        )
        current_points = torch.where(
            dynamic_nearer.unsqueeze(-1), dynamic_body, static_body
        )
        current_distance = torch.where(
            dynamic_nearer, dynamic_distance, static_distance
        )
        current_valid = static_valid | dynamic_valid

        predicted_dynamic, false_positive = self._sample_mask(
            visible_dynamic, current_distance, current_valid
        )

        relative_velocity_w = object_velocity - vel_w
        if quat_w.ndim == 2:
            quat_w = quat_w.unsqueeze(1)
        relative_velocity_b = quat_rotate_inverse(
            quat_w.expand(batch, num_rays, 4),
            relative_velocity_w,
        )
        radial_hat = current_points / current_distance.unsqueeze(
            -1
        ).clamp_min(1e-6)
        x, y, _ = current_points.unbind(dim=-1)
        theta_hat = torch.stack(
            [-y, x, torch.zeros_like(x)], dim=-1
        )
        theta_hat = theta_hat / torch.linalg.norm(
            theta_hat, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        phi_hat = torch.cross(radial_hat, theta_hat, dim=-1)
        spherical_velocity = torch.stack(
            [
                (relative_velocity_b * radial_hat).sum(-1),
                (relative_velocity_b * theta_hat).sum(-1),
                (relative_velocity_b * phi_hat).sum(-1),
            ],
            dim=-1,
        )

        noise_scale = torch.empty(
            batch, 1, 1, device=self.device
        ).uniform_(self.noise_scale_min, self.noise_scale_max)
        velocity_error = (
            torch.randn_like(spherical_velocity)
            * self.velocity_std.to(spherical_velocity.dtype)
            * noise_scale
        )
        if self.velocity_bias_std > 0.0:
            velocity_error += (
                torch.randn(batch, 1, 3, device=self.device)
                * self.velocity_bias_std
            )
        predicted_velocity = spherical_velocity + velocity_error
        fp_velocity = (
            torch.randn_like(predicted_velocity) * self.fp_velocity_std
        )
        predicted_velocity = torch.where(
            false_positive.unsqueeze(-1),
            fp_velocity,
            predicted_velocity,
        )
        predicted_velocity = torch.where(
            predicted_dynamic.unsqueeze(-1),
            predicted_velocity,
            torch.zeros_like(predicted_velocity),
        )
        sigma = (
            self.sigma_floor
            + self.sigma_error_gain * velocity_error.abs().mean(-1)
        ).clamp(self.sigma_floor, self.sigma_max)
        sigma = torch.where(
            predicted_dynamic, sigma, torch.zeros_like(sigma)
        )

        noisy_distance = current_distance
        if self.depth_noise_std > 0.0:
            noisy_distance = (
                current_distance
                + torch.randn_like(current_distance)
                * self.depth_noise_std
            ).clamp(0.1, self.d_max)
        noisy_points = radial_hat * noisy_distance.unsqueeze(-1)

        static_depth = self._project_depth(
            noisy_points, current_valid & (~predicted_dynamic)
        )
        dynamic_depth = self._project_depth(
            noisy_points, current_valid & predicted_dynamic
        )
        velocity_images = [
            self._project_feature(
                noisy_points,
                current_valid & predicted_dynamic,
                predicted_velocity[..., component],
            )
            for component in range(3)
        ]
        sigma_image = self._project_feature(
            noisy_points,
            current_valid & predicted_dynamic,
            sigma,
        )
        static_norm = (static_depth / self.d_max).clamp(0.0, 1.0)
        dynamic_norm = torch.cat(
            [
                (dynamic_depth / self.d_max).clamp(0.0, 1.0),
                (velocity_images[0] / self.dyn_max_vel).clamp(-1.0, 1.0),
                (velocity_images[1] / self.dyn_max_vel).clamp(-1.0, 1.0),
                (velocity_images[2] / self.dyn_max_vel).clamp(-1.0, 1.0),
                (sigma_image / self.uncertainty_scale).clamp(0.0, 1.0),
            ],
            dim=1,
        )

        first_frame = ~self.history_valid
        if first_frame.any():
            first_static = self._project_depth(
                noisy_points, current_valid
            ) / self.d_max
            static_norm[first_frame] = first_static[first_frame]
            dynamic_norm[first_frame] = 0.0
            dynamic_norm[first_frame, 0] = 1.0
        self.history_valid[:] = True
        return static_norm, dynamic_norm

    @torch.no_grad()
    def process(
        self,
        hits_static_w,
        hits_dyn_w,
        is_dyn_hit,
        dyn_hit_idx,
        all_dyn_vel_w,
        pos_w,
        quat_w,
        vel_w,
    ):
        if self.step_count % self.update_interval == 0:
            static, dynamic = self._compute_update(
                hits_static_w,
                hits_dyn_w,
                is_dyn_hit,
                dyn_hit_idx,
                all_dyn_vel_w,
                pos_w,
                quat_w,
                vel_w,
            )
            self.cached_static.copy_(static)
            self.cached_dynamic.copy_(dynamic)
        self.step_count += 1
        return self.cached_static.clone(), self.cached_dynamic.clone()


def compute_rayhitsdir(device, num_envs, h_fov, v_fov, h_num, v_num):
    if h_fov == 360:
        horizontal_angles = torch.linspace(
            0, h_fov, h_num + 1, device=device
        )[:h_num]
    else:
        horizontal_angles = torch.linspace(
            0, h_fov, h_num, device=device
        )
    vertical_angles = torch.linspace(
        v_fov[0], v_fov[1], v_num, device=device
    )
    horizontal, vertical = torch.meshgrid(
        horizontal_angles * torch.pi / 180,
        vertical_angles * torch.pi / 180,
        indexing="ij",
    )
    directions = torch.stack(
        (
            torch.cos(vertical) * torch.cos(horizontal),
            torch.cos(vertical) * torch.sin(horizontal),
            torch.sin(vertical),
        ),
        dim=-1,
    )
    return directions.reshape(h_num * v_num, -1).unsqueeze(
        0
    ).expand(num_envs, -1, -1)


def sphere_lidar_hits(
    lidar_range, spheres_pos, spheres_radius, pos_w, ray_hits_dir
):
    point = pos_w.unsqueeze(1).unsqueeze(2)
    direction = ray_hits_dir.unsqueeze(2)
    center = spheres_pos.unsqueeze(0).unsqueeze(0)
    radius = spheres_radius.reshape(1, 1, -1, 1)
    offset = center - point
    t_ca = torch.sum(offset * direction, dim=-1, keepdim=True)
    d_sq = torch.sum(
        offset * offset, dim=-1, keepdim=True
    ) - t_ca.square()
    discriminant = radius.square() - d_sq
    t0 = t_ca - torch.sqrt(discriminant.clamp_min(0.0))
    valid = (
        (discriminant >= 0)
        & (t0 > 0)
        & (t0 < float(lidar_range))
    )
    t0_values = t0.squeeze(-1)
    t0_values[~valid.squeeze(-1)] = float("inf")
    min_t, hit_idx = torch.min(t0_values, dim=-1)
    is_hit = torch.isfinite(min_t)
    final_t = torch.where(
        is_hit,
        min_t,
        torch.full_like(min_t, float(lidar_range)),
    )
    hits = pos_w.unsqueeze(1) + final_t.unsqueeze(-1) * ray_hits_dir
    return hits, is_hit, hit_idx

