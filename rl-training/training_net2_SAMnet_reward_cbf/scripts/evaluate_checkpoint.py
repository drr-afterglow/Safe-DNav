import gc
import hashlib
import json
import os
from datetime import datetime

import hydra
import torch
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from torchrl.envs.transforms import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType

from ppo import PPO
from utils import evaluate


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    sim_app = SimulationApp({"headless": True, "anti_aliasing": 1})
    try:
        asset_root = getattr(cfg, "isaac_asset_root", None)
        if asset_root:
            if not (
                os.path.isdir(os.path.join(asset_root, "Isaac"))
                and os.path.isdir(os.path.join(asset_root, "NVIDIA"))
            ):
                raise FileNotFoundError(f"invalid local Isaac asset root: {asset_root}")
            import carb
            carb.settings.get_settings().set(
                "/persistent/isaac/asset_root/default", asset_root
            )

        from env import NavigationEnv

        env = NavigationEnv(cfg)
        transforms = [
            VelController(
                LeePositionController(9.81, env.drone.params).to(cfg.device),
                yaw_control=False,
            )
        ]
        transformed_env = TransformedEnv(env, Compose(*transforms)).eval()

        policy = PPO(
            cfg.algo,
            transformed_env.observation_spec,
            transformed_env.action_spec,
            cfg.device,
        )
        checkpoint = os.path.abspath(str(cfg.checkpoint))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location=cfg.device)
        state_dict = (
            payload["policy_state_dict"]
            if isinstance(payload, dict) and "policy_state_dict" in payload
            else payload
        )
        policy.load_state_dict(state_dict)

        seeds = [int(seed) for seed in cfg.eval_seeds]
        runs = []
        for seed in seeds:
            print(f"[NavRL independent eval]: seed={seed}", flush=True)
            result = evaluate(
                env=transformed_env,
                policy=policy,
                seed=seed,
                cfg=cfg,
                exploration_type=ExplorationType.MEAN,
            )
            runs.append({key: float(value) for key, value in result.items()})
            print(json.dumps(runs[-1], sort_keys=True), flush=True)

        aggregate = {}
        for key in runs[0]:
            values = torch.tensor([run[key] for run in runs], dtype=torch.float64)
            aggregate[key + "_mean"] = float(values.mean())
            aggregate[key + "_std"] = float(values.std(unbiased=False))

        aggregate["eval/selection_score"] = (
            aggregate["eval/stats.reach_goal_mean"]
            - float(cfg.collision_score_weight)
            * aggregate["eval/stats.collision_mean"]
        )
        with open(checkpoint, "rb") as handle:
            checkpoint_sha256 = hashlib.sha256(handle.read()).hexdigest()

        report = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "num_envs_per_seed": int(cfg.env.num_envs),
            "episode_count": int(cfg.env.num_envs) * len(seeds),
            "seeds": seeds,
            "difficulty": {
                "static_obstacles": int(cfg.env.num_obstacles),
                "dynamic_obstacles": int(cfg.env_dyn.num_obstacles),
                "dynamic_velocity_range_mps": list(cfg.env_dyn.vel_range),
                "agent_velocity_range_mps": [float(env.vel_min), float(env.vel_max)],
            },
            "runs": runs,
            "aggregate": aggregate,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        output = str(getattr(cfg, "eval_output", "/root/autodl-tmp/NavRL_runs/evaluations/new_best_unseen_seeds.json"))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("[NavRL independent eval]: aggregate", flush=True)
        print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
        print(f"[NavRL independent eval]: report={output}", flush=True)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        sim_app.close()


if __name__ == "__main__":
    main()
