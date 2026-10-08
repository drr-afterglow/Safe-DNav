"""Evaluate a frozen SA-UMNet checkpoint without updating model parameters."""

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from perception_training.dataset import LidarPairDataset
from perception_training.losses import PerceptionLoss
from perception_training.metrics import PerceptionMetricAccumulator
from perception_training.model import SAUMNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-threshold", type=float)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    network_sensor = cfg["sensor"].get("network", cfg["sensor"])
    model = SAUMNet(
        height=network_sensor["height"],
        width=network_sensor["width"],
        vfov_deg=tuple(cfg["sensor"]["vfov_deg"]),
        base_channels=cfg["model"]["base_channels"],
        input_channels=cfg["model"].get("input_channels", 4),
        reference_delta_t=cfg["model"].get("reference_delta_t", 0.096),
        min_delta_t=cfg["model"].get("min_delta_t", 0.05),
        max_delta_t=cfg["model"].get("max_delta_t", 0.15),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = LidarPairDataset(
        args.roots,
        default_delta_t=cfg["model"].get("reference_delta_t", 0.096),
        min_delta_t=cfg["model"].get("min_delta_t", 0.05),
        max_delta_t=cfg["model"].get("max_delta_t", 0.15),
        cache_size=cfg["data"].get("cache_size", 4),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    criterion = PerceptionLoss(**cfg["loss"]).to(device)
    evaluation_cfg = cfg.get("evaluation", {})
    rl_sensor = cfg["sensor"].get("rl", {"height": 6, "width": 36})
    metrics = PerceptionMetricAccumulator(
        max_range_m=cfg["sensor"]["max_range"],
        mask_threshold=(args.mask_threshold if args.mask_threshold is not None else evaluation_cfg.get("mask_threshold", 0.5)),
        distance_bins_m=evaluation_cfg.get("distance_bins_m", [0.0, 2.0, 5.0, 10.0]),
        pool_output_size=tuple(evaluation_cfg.get("pool_output_size", [rl_sensor["height"], rl_sensor["width"]])),
        seam_width=evaluation_cfg.get("seam_width", 3),
        static_speed_hist_max_mps=evaluation_cfg.get("static_speed_hist_max_mps", 10.0),
        static_speed_hist_bins=evaluation_cfg.get("static_speed_hist_bins", 1024),
    )
    totals = {}
    count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            prediction = model(batch["input"], batch.get("delta_t"))
            losses = criterion(prediction, batch)
            metrics.update(prediction, batch)
            bsz = batch["input"].shape[0]
            count += bsz
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * bsz
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {key: value / max(count, 1) for key, value in totals.items()}
    report.update(metrics.compute())
    report.update({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch_1based": int(checkpoint.get("epoch", -1)) + 1,
        "checkpoint_selection_score": float(checkpoint.get("selection_score", checkpoint.get("best_score", float("nan")))),
        "dataset_roots": [str(Path(root).resolve()) for root in args.roots],
        "samples": count,
        "mask_threshold": (args.mask_threshold if args.mask_threshold is not None else evaluation_cfg.get("mask_threshold", 0.5)),
        "elapsed_seconds": elapsed,
        "samples_per_second": count / max(elapsed, 1e-9),
        "device": str(device),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False), end="")


if __name__ == "__main__":
    main()
