import argparse
import os
import sys
import traceback

import torch
import yaml


MOTION_METRIC_KEYS = (
    "ego_translation_m",
    "ego_rotation_deg",
    "ego_linear_speed_mps",
    "ego_angular_speed_deg_s",
    "alignment_position_error_m",
    "alignment_rotation_error_deg",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect supervised perception-training data in Isaac.")
    parser.add_argument("--config", default="perception_training/data_collection/config/default.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. env_dyn.num_obstacles=80")
    return parser.parse_args()


def set_by_dot(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    node[parts[-1]] = parsed


def load_config(path, overrides):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, value = override.split("=", 1)
        set_by_dot(cfg, key, value)
    return cfg


def summarize_batch(batch):
    mask = batch["mask"]
    dyn_pixels = mask.flatten(1).sum(dim=1)
    dyn_keep = dyn_pixels > 0
    velocity = batch["velocity"]
    speed = velocity.square().sum(dim=1).sqrt()
    if dyn_keep.any():
        dyn_speed = speed[mask[:, 0] > 0]
        speed_mean = float(dyn_speed.mean().detach().cpu()) if dyn_speed.numel() else 0.0
        speed_max = float(dyn_speed.max().detach().cpu()) if dyn_speed.numel() else 0.0
    else:
        speed_mean = 0.0
        speed_max = 0.0
    summary = {
        "batch": int(mask.shape[0]),
        "dyn_samples": int(dyn_keep.sum().detach().cpu()),
        "dyn_pixels_mean": float(dyn_pixels.float().mean().detach().cpu()),
        "dyn_pixels_max": float(dyn_pixels.max().detach().cpu()),
        "speed_mean": speed_mean,
        "speed_max": speed_max,
    }
    if "valid_return_fraction" in batch:
        summary["valid_return_fraction"] = float(batch["valid_return_fraction"].float().mean().detach().cpu())
    if "raw_point_count" in batch:
        summary["raw_point_count"] = int(batch["raw_point_count"][0].detach().cpu())
    if "nominal_point_count" in batch:
        summary["nominal_point_count"] = int(batch["nominal_point_count"][0].detach().cpu())
    if "panorama_occupancy" in batch:
        summary["panorama_occupancy"] = float(batch["panorama_occupancy"].float().mean().detach().cpu())
    if "row_occupancy" in batch:
        summary["row_occupancy"] = batch["row_occupancy"].float().mean(dim=0).detach().cpu().tolist()
    if "consecutive_frame_overlap" in batch:
        summary["consecutive_frame_overlap"] = float(
            batch["consecutive_frame_overlap"].float().mean().detach().cpu()
        )
    for key in MOTION_METRIC_KEYS:
        if key in batch:
            summary[key] = float(batch[key].float().mean().detach().cpu())
    if "delta_t" in batch:
        summary["delta_t"] = float(batch["delta_t"].float().mean().detach().cpu())
    return summary


def filter_samples(batch, min_dynamic_pixels, keep_empty_ratio):
    if min_dynamic_pixels <= 0 and keep_empty_ratio >= 1.0:
        return batch
    dyn_pixels = batch["mask"].flatten(1).sum(dim=1)
    positive = dyn_pixels >= float(min_dynamic_pixels)
    keep = positive
    if keep_empty_ratio > 0:
        empty = ~positive
        empty_keep = empty & (torch.rand_like(dyn_pixels.float()) < float(keep_empty_ratio))
        keep = keep | empty_keep
    if keep.any():
        return {key: value[keep] for key, value in batch.items()}
    return None


def limit_batch_size(batch, max_samples):
    if max_samples <= 0:
        return None
    bsz = batch["input"].shape[0]
    if bsz <= max_samples:
        return batch
    return {key: value[:max_samples] for key, value in batch.items()}


def main():
    args = parse_args()
    cfg_dict = load_config(args.config, args.overrides)
    collection_cfg = cfg_dict.get("collection", {})
    force_exit_after_success = bool(collection_cfg.get("force_exit_after_success", True))
    close_sim_app = bool(collection_cfg.get("close_sim_app", False))

    from omni.isaac.kit import SimulationApp

    print("[perception-data] starting SimulationApp...", flush=True)
    sim_app = SimulationApp({"headless": bool(cfg_dict["headless"]), "anti_aliasing": 1})
    try:
        print("[perception-data] importing torch/omegaconf/env...", flush=True)
        import torch
        from omegaconf import OmegaConf

        from perception_training.data_collection.env import PerceptionDataCollectionEnv
        from perception_training.data_collection.writer import ShardWriter

        cfg = OmegaConf.create(cfg_dict)
        torch.manual_seed(int(cfg.seed))
        print("[perception-data] creating collection environment...", flush=True)
        env = PerceptionDataCollectionEnv(cfg)
        overwrite = bool(cfg.collection.get("overwrite_output", True))
        writer = ShardWriter(cfg.collection.output_dir, cfg.collection.shard_size, overwrite=overwrite)
        print("[perception-data] warming up dynamic obstacles...", flush=True)
        for _ in range(int(cfg.collection.warmup_steps)):
            env.move_dynamic_obstacles()

        target = int(cfg.collection.num_samples)
        iteration = 0
        print(f"[perception-data] collecting target={target} samples...", flush=True)
        totals = {
            "raw_batches": 0,
            "raw_samples": 0,
            "kept_samples": 0,
            "dyn_samples": 0,
            "valid_return_fraction_sum": 0.0,
            "panorama_occupancy_sum": 0.0,
            "consecutive_frame_overlap_sum": 0.0,
            "row_occupancy_sum": None,
        }
        for key in (*MOTION_METRIC_KEYS, "delta_t"):
            totals[f"{key}_sum"] = 0.0
        while writer.num_samples < target:
            batch = env.collect_supervised_batch()
            stats = summarize_batch(batch)
            totals["raw_batches"] += 1
            totals["raw_samples"] += stats["batch"]
            totals["dyn_samples"] += stats["dyn_samples"]
            totals["valid_return_fraction_sum"] += stats.get("valid_return_fraction", 0.0) * stats["batch"]
            totals["panorama_occupancy_sum"] += stats.get("panorama_occupancy", 0.0) * stats["batch"]
            totals["consecutive_frame_overlap_sum"] += (
                stats.get("consecutive_frame_overlap", 0.0) * stats["batch"]
            )
            if "row_occupancy" in stats:
                if totals["row_occupancy_sum"] is None:
                    totals["row_occupancy_sum"] = [0.0] * len(stats["row_occupancy"])
                totals["row_occupancy_sum"] = [
                    total + value * stats["batch"]
                    for total, value in zip(totals["row_occupancy_sum"], stats["row_occupancy"])
                ]
            for key in (*MOTION_METRIC_KEYS, "delta_t"):
                totals[f"{key}_sum"] += stats.get(key, 0.0) * stats["batch"]
            batch = filter_samples(
                batch,
                int(cfg.collection.min_dynamic_pixels),
                float(cfg.collection.get("keep_empty_ratio", 0.0)),
            )
            if batch is not None:
                batch = limit_batch_size(batch, target - writer.num_samples)
            if batch is not None:
                totals["kept_samples"] += int(batch["input"].shape[0])
                writer.add_batch(batch)
            iteration += 1
            if iteration % int(cfg.collection.log_interval) == 0:
                keep_rate = totals["kept_samples"] / max(totals["raw_samples"], 1)
                dyn_rate = totals["dyn_samples"] / max(totals["raw_samples"], 1)
                print(
                    "[perception-data] "
                    f"collected={writer.num_samples}/{target} "
                    f"keep_rate={keep_rate:.3f} dyn_sample_rate={dyn_rate:.3f} "
                    f"dyn_px_mean={stats['dyn_pixels_mean']:.2f} dyn_px_max={stats['dyn_pixels_max']:.0f} "
                    f"speed_mean={stats['speed_mean']:.3f} speed_max={stats['speed_max']:.3f} "
                    f"valid_returns={stats.get('valid_return_fraction', 0.0):.3f} "
                    f"occupancy={stats.get('panorama_occupancy', 0.0):.3f} "
                    f"frame_overlap={stats.get('consecutive_frame_overlap', 0.0):.3f} "
                    f"ego_motion={stats.get('ego_translation_m', 0.0):.3f}m/"
                    f"{stats.get('ego_rotation_deg', 0.0):.2f}deg "
                    f"raw_points={stats.get('raw_point_count', 0)}/{stats.get('nominal_point_count', 0)}",
                    flush=True,
                )
        writer.flush()
        totals["valid_return_fraction_mean"] = totals["valid_return_fraction_sum"] / max(
            totals["raw_samples"], 1
        )
        totals["panorama_occupancy_mean"] = totals["panorama_occupancy_sum"] / max(
            totals["raw_samples"], 1
        )
        totals["consecutive_frame_overlap_mean"] = totals["consecutive_frame_overlap_sum"] / max(
            totals["raw_samples"], 1
        )
        if totals["row_occupancy_sum"] is not None:
            totals["row_occupancy_mean"] = [
                value / max(totals["raw_samples"], 1) for value in totals["row_occupancy_sum"]
            ]
        for key in (*MOTION_METRIC_KEYS, "delta_t"):
            totals[f"{key}_mean"] = totals[f"{key}_sum"] / max(totals["raw_samples"], 1)
        writer.write_manifest(cfg_dict, totals)
        print(f"[perception-data] done, total_saved={writer.num_samples}", flush=True)
        if force_exit_after_success:
            print("[perception-data] exiting without SimulationApp.close() to avoid Isaac shutdown segfault.", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    except Exception:
        print("[perception-data] ERROR during collection:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if close_sim_app:
            print("[perception-data] closing SimulationApp...", flush=True)
            sim_app.close()
        else:
            print("[perception-data] skipping SimulationApp.close().", flush=True)


if __name__ == "__main__":
    main()
