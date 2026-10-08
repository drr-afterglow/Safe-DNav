"此处为训练脚本,使用PPO算法训练无人机导航任务,集成了Isaac Sim仿真环境和Wandb实验跟踪"
import os
import hydra    
import datetime 
import wandb    
import torch    
from omegaconf import DictConfig, OmegaConf     
from omni.isaac.kit import SimulationApp        
from ppo import PPO                             
from omni_drones.controllers import LeePositionController                      
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite 
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats           
from torchrl.envs.transforms import TransformedEnv, Compose                     
from utils import evaluate                                                      
from torchrl.envs.utils import ExplorationType                                  

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


def _checkpoint_payload(policy, step, env_frames):
    return {
        "format_version": 2,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dicts": {
            "feature_extractor": policy.feature_extractor_optim.state_dict(),
            "actor": policy.actor_optim.state_dict(),
            "critic": policy.critic_optim.state_dict(),
        },
        "cbf_lambda": float(policy.cbf_lambda.item()),
        "step": int(step),
        "env_frames": int(env_frames),
    }


def _save_checkpoint(path, policy, step, env_frames):
    # Keep the conventional file deployment-compatible and store exact
    # continuation state beside it.
    torch.save(policy.state_dict(), path)
    stem, suffix = os.path.splitext(path)
    training_path = stem + "_train" + suffix
    torch.save(_checkpoint_payload(policy, step, env_frames), training_path)


def _load_checkpoint(path, policy, cfg):
    payload = torch.load(path, map_location=cfg.device)
    if isinstance(payload, dict) and "policy_state_dict" in payload:
        policy.load_state_dict(payload["policy_state_dict"])
        if bool(getattr(cfg, "resume_optimizer", False)):
            optimizer_states = payload.get("optimizer_state_dicts", {})
            for name, optimizer in (
                ("feature_extractor", policy.feature_extractor_optim),
                ("actor", policy.actor_optim),
                ("critic", policy.critic_optim),
            ):
                if name in optimizer_states:
                    optimizer.load_state_dict(optimizer_states[name])
        if bool(getattr(cfg, "resume_cbf_lambda", True)) and "cbf_lambda" in payload:
            policy.cbf_lambda.fill_(float(payload["cbf_lambda"]))
        return payload
    policy.load_state_dict(payload)
    return {"format_version": 1, "policy_state_dict": payload}


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)

def main(cfg):

    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    asset_root = getattr(cfg, "isaac_asset_root", None)
    if asset_root:
        if not (os.path.isdir(os.path.join(asset_root, "Isaac")) and os.path.isdir(os.path.join(asset_root, "NVIDIA"))):
            raise FileNotFoundError(f"invalid local Isaac asset root: {asset_root}")
        import carb
        carb.settings.get_settings().set("/persistent/isaac/asset_root/default", asset_root)

    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            dir=cfg.wandb.dir,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            dir=cfg.wandb.dir,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    from env import NavigationEnv
    env = NavigationEnv(cfg)
    env.update_curriculum(0)  # Apply phase zero before the first rollout.

    #对初始化的环境进行变换，RL输出的速度命令--转换器--无人机底层控制命令--控制环境中的无人机
    transforms = []
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    


    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    
    # 加载预训练权重，用于断点续训练（仅当配置中指定了 checkpoint 路径时才加载）
    checkpoint = getattr(cfg, "checkpoint", None)
    if checkpoint is not None and os.path.isfile(checkpoint):
        checkpoint_meta = _load_checkpoint(checkpoint, policy, cfg)
        print(
            f"[NavRL]: loaded checkpoint from {checkpoint}; "
            f"format={checkpoint_meta.get('format_version', 1)}, "
            f"cbf_lambda={policy.cbf_lambda.item():.6f}"
        )
    else:
        print("[NavRL]: training from scratch (no checkpoint loaded)")
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(cfg.device)

    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )

    # Evaluation selection and early stopping state
    best_eval_score = float("-inf")
    patience_reference_score = float("-inf")
    bad_eval_count = 0
    should_stop = False
    eval_seeds = [int(seed) for seed in getattr(cfg, "eval_seeds", [cfg.seed])]
    early_stop_patience = int(getattr(cfg, "early_stop_patience", 0))
    early_stop_min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))
    collision_score_weight = float(getattr(cfg, "collision_score_weight", 0.0))
    static_collision_score_weight = float(
        getattr(cfg, "static_collision_score_weight", 1.0)
    )
    dynamic_collision_score_weight = float(
        getattr(cfg, "dynamic_collision_score_weight", 1.0)
    )
    boundary_score_weight = float(getattr(cfg, "boundary_score_weight", 0.0))

    _save_checkpoint(
        os.path.join(run.dir, "checkpoint_initial.pt"),
        policy,
        step=-1,
        env_frames=0,
    )

    # Training Loop
    for i, data in enumerate(collector):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        env.update_curriculum(collector._frames)
        info["curriculum/active_dyn_obs"] = env.active_dyn_obs
        info["curriculum/vel_min"] = env.vel_min
        info["curriculum/vel_max"] = env.vel_max

        train_loss_stats = policy.train(data)
        info.update(train_loss_stats)
        if torch.cuda.is_available():
            info["system/gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(cfg.device) / (1024 ** 3)
            info["system/gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(cfg.device) / (1024 ** 3)
        del train_loss_stats

        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        del data

        if cfg.eval_interval > 0 and i > 0 and i % cfg.eval_interval == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            eval_runs = []
            for eval_seed in eval_seeds:
                print(f"[NavRL]: evaluating step {i} with seed {eval_seed}")
                eval_runs.append(evaluate(
                    env=transformed_env,
                    policy=policy,
                    seed=eval_seed,
                    cfg=cfg,
                    exploration_type=ExplorationType.MEAN,
                ))

            eval_info = {}
            for key in eval_runs[0].keys():
                values = [float(result[key]) for result in eval_runs]
                eval_info[key] = max(values) if "gpu_peak" in key else sum(values) / len(values)
                if len(values) > 1:
                    eval_info[key + "_std"] = torch.tensor(values).std(unbiased=False).item()
            for eval_seed, result in zip(eval_seeds, eval_runs):
                for key, value in result.items():
                    if key.startswith("eval/stats."):
                        eval_info[f"eval_seed/{eval_seed}/{key[5:]}"] = float(value)

            selection_score = (
                eval_info["eval/stats.reach_goal"]
                - collision_score_weight * eval_info["eval/stats.collision"]
                - static_collision_score_weight
                * eval_info["eval/stats.collision_static"]
                - dynamic_collision_score_weight
                * eval_info["eval/stats.collision_dynamic"]
                - boundary_score_weight
                * eval_info["eval/stats.boundary_violation"]
            )
            eval_info["eval/selection_score"] = selection_score

            # Always preserve the raw highest-scoring policy.  min_delta is used
            # only for early-stopping patience and must not hide a true maximum.
            if selection_score > best_eval_score:
                best_eval_score = selection_score
                _save_checkpoint(
                    os.path.join(run.dir, "checkpoint_best.pt"),
                    policy,
                    step=i,
                    env_frames=collector._frames,
                )
                eval_info["eval/is_best"] = 1.0
                print(f"[NavRL]: new best checkpoint at step {i}, score={selection_score:.4f}")
            else:
                eval_info["eval/is_best"] = 0.0

            if selection_score > patience_reference_score + early_stop_min_delta:
                patience_reference_score = selection_score
                bad_eval_count = 0
            else:
                bad_eval_count += 1
            eval_info["eval/best_selection_score"] = best_eval_score
            eval_info["eval/bad_eval_count"] = bad_eval_count
            should_stop = early_stop_patience > 0 and bad_eval_count >= early_stop_patience

            env.train()
            env.reset()
            info.update(eval_info)
            print(f"[NavRL]: evaluation done at step {i}; bad_eval_count={bad_eval_count}")

        run.log(info)

        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            _save_checkpoint(ckpt_path, policy, step=i, env_frames=collector._frames)
            print("[NavRL]: model saved at training step: ", i)

        if should_stop:
            print(f"[NavRL]: early stopping at step {i}")
            break

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    _save_checkpoint(
        ckpt_path,
        policy,
        step=i,
        env_frames=collector._frames,
    )
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    
