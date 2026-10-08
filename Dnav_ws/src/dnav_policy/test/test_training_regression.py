"""Numerical regression against the original PPO implementation."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest

import torch

from dnav_policy.checkpoint import load_policy_checkpoint
from dnav_policy.model import PolicyActor


ROOT = Path(__file__).resolve().parents[4]
TRAINING_SCRIPTS = ROOT / "rl-training" / "training_net2_SAMnet_reward_cbf" / "scripts"
CHECKPOINT = ROOT / "Dnav_ws" / "weights" / "policy_best.pt"


def quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion_vector = quaternion[..., 1:]
    cross = 2.0 * torch.cross(quaternion_vector, vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.cross(quaternion_vector, cross, dim=-1)


def import_training_ppo():
    # ppo.py only needs this one OmniDrones function. Supplying it directly
    # keeps this regression runnable without starting Isaac Sim.
    omni_drones = ModuleType("omni_drones")
    omni_drones.__path__ = []
    omni_utils = ModuleType("omni_drones.utils")
    omni_utils.__path__ = []
    omni_torch = ModuleType("omni_drones.utils.torch")
    omni_torch.quat_rotate = quat_rotate
    sys.modules["omni_drones"] = omni_drones
    sys.modules["omni_drones.utils"] = omni_utils
    sys.modules["omni_drones.utils.torch"] = omni_torch
    sys.path.insert(0, str(TRAINING_SCRIPTS))
    from ppo import PPO

    return PPO


def make_training_policy():
    from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

    PPO = import_training_ppo()
    observation_spec = CompositeSpec(
        {
            "agents": CompositeSpec(
                {
                    "observation": CompositeSpec(
                        {
                            "state": UnboundedContinuousTensorSpec((13,)),
                            "lidar_static": UnboundedContinuousTensorSpec((1, 6, 36)),
                            "lidar_dynamic": UnboundedContinuousTensorSpec((5, 6, 36)),
                        }
                    )
                }
            ),
            "info": CompositeSpec(
                {"drone_state": UnboundedContinuousTensorSpec((1, 13))}
            ),
        }
    ).expand(1)
    action_spec = UnboundedContinuousTensorSpec((1, 3))
    cfg = SimpleNamespace(
        feature_extractor=SimpleNamespace(learning_rate=5e-6),
        actor=SimpleNamespace(learning_rate=5e-6, action_limit=3.0, clip_ratio=0.1),
        critic=SimpleNamespace(learning_rate=5e-6, clip_ratio=0.1),
        entropy_loss_coefficient=2e-5,
        cbf_lambda_init=0.33,
        cbf_lambda_lr=1e-3,
        cbf_cost_limit=0.02,
        cbf_lambda_max=1.0,
    )
    policy = PPO(cfg, observation_spec, action_spec, "cpu")
    policy.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=True)
    return policy


class TrainingImplementationRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # PPO names its optimization method ``train`` and therefore shadows
        # nn.Module.train/eval. The graph has no train/eval-sensitive layers.
        cls.reference = make_training_policy()
        cls.deployment = PolicyActor().eval()
        load_policy_checkpoint(cls.deployment, CHECKPOINT)

    def test_distribution_and_actions_match_original_ppo(self):
        from tensordict import TensorDict
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        torch.manual_seed(20260917)
        batch = 7
        static = torch.rand(batch, 1, 6, 36)
        dynamic = torch.rand(batch, 5, 6, 36) * 2.0 - 1.0
        dynamic[:, 0] = torch.rand(batch, 6, 36)
        state = torch.rand(batch, 13) * 2.0 - 1.0
        quaternion = torch.randn(batch, 1, 4)
        quaternion /= quaternion.norm(dim=-1, keepdim=True)
        drone_state = torch.zeros(batch, 1, 13)
        drone_state[..., 3:7] = quaternion
        reference_input = TensorDict(
            {
                "agents": {
                    "observation": {
                        "state": state.clone(),
                        "lidar_static": static.clone(),
                        "lidar_dynamic": dynamic.clone(),
                    }
                },
                "info": {"drone_state": drone_state},
            },
            batch_size=[batch],
        )

        with torch.inference_mode(), set_exploration_type(ExplorationType.MEAN):
            reference_output = self.reference(reference_input)
            actual_alpha, actual_beta = self.deployment.distribution_parameters(
                static, dynamic, state
            )
            actual_body = self.deployment(static, dynamic, state)
            actual_world = quat_rotate(
                quaternion.expand(batch, 1, 4), actual_body.reshape(batch, 1, 3)
            )

        torch.testing.assert_close(
            actual_alpha, reference_output["alpha"], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            actual_beta, reference_output["beta"], rtol=0.0, atol=0.0
        )
        expected_body = (
            2.0
            * reference_output["agents", "action_normalized"]
            * self.reference.cfg.actor.action_limit
            - self.reference.cfg.actor.action_limit
        )
        torch.testing.assert_close(actual_body, expected_body, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            actual_world,
            reference_output["agents", "action"],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
