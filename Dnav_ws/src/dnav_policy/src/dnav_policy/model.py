"""Pure-PyTorch inference graph matching the trained PPO actor."""

import torch
from torch import nn


class BetaActorHead(nn.Module):
    def __init__(self, feature_dim: int = 256, action_dim: int = 3):
        super().__init__()
        self.alpha_layer = nn.Linear(feature_dim, action_dim)
        self.beta_layer = nn.Linear(feature_dim, action_dim)

    def forward(self, features: torch.Tensor):
        alpha = 1.0 + torch.nn.functional.softplus(self.alpha_layer(features)) + 1e-6
        beta = 1.0 + torch.nn.functional.softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta


def _image_encoder(input_channels: int, first_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, first_channels, kernel_size=(5, 3), padding=(2, 1)),
        nn.ELU(),
        nn.Conv2d(first_channels, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
        nn.ELU(),
        nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
        nn.ELU(),
        nn.Flatten(),
        nn.Linear(576, 128),
        nn.LayerNorm(128),
    )


class PolicyActor(nn.Module):
    """Deterministic deployment view of the trained five-channel PPO policy."""

    STATIC_SHAPE = (1, 6, 36)
    DYNAMIC_SHAPE = (5, 6, 36)
    STATE_SIZE = 13

    def __init__(self, action_limit: float = 3.0):
        super().__init__()
        if action_limit <= 0.0:
            raise ValueError("action_limit must be positive")
        self.action_limit = float(action_limit)
        self.static_encoder = _image_encoder(1, 4)
        self.dynamic_encoder = _image_encoder(5, 8)
        self.fusion = nn.Sequential(
            nn.Linear(269, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
        )
        self.actor = BetaActorHead()

    @staticmethod
    def _validate(name: str, value: torch.Tensor, trailing_shape):
        if value.ndim != len(trailing_shape) + 1 or tuple(value.shape[1:]) != tuple(trailing_shape):
            raise ValueError(
                f"{name} must have shape (B, {', '.join(map(str, trailing_shape))}), "
                f"received {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")

    def distribution_parameters(
        self,
        lidar_static: torch.Tensor,
        lidar_dynamic: torch.Tensor,
        state: torch.Tensor,
    ):
        self._validate("lidar_static", lidar_static, self.STATIC_SHAPE)
        self._validate("lidar_dynamic", lidar_dynamic, self.DYNAMIC_SHAPE)
        self._validate("state", state, (self.STATE_SIZE,))
        if not (lidar_static.shape[0] == lidar_dynamic.shape[0] == state.shape[0]):
            raise ValueError("policy inputs must have the same batch size")
        static_feature = self.static_encoder(lidar_static)
        dynamic_feature = self.dynamic_encoder(lidar_dynamic)
        # TorchRL CatTensors sorts the three training keys by default. The
        # learned 269-vector order is therefore static, dynamic, state (not the
        # textual order used when CatTensors was constructed in ppo.py).
        feature = self.fusion(torch.cat((static_feature, dynamic_feature, state), dim=-1))
        return self.actor(feature)

    def forward(
        self,
        lidar_static: torch.Tensor,
        lidar_dynamic: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return the Beta-mean action mapped to body velocity in m/s."""
        alpha, beta = self.distribution_parameters(lidar_static, lidar_dynamic, state)
        normalized_action = alpha / (alpha + beta)
        return 2.0 * normalized_action * self.action_limit - self.action_limit
