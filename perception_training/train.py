import argparse
import os
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import yaml

from perception_training.augment import LidarAugment
from perception_training.dataset import LidarPairDataset, ShardAwareSampler, split_dataset
from perception_training.losses import PerceptionLoss
from perception_training.metrics import PerceptionMetricAccumulator, composite_selection_score
from perception_training.model import SAUMNet


def log(message: str):
    print(f"[perception-train] {message}", flush=True)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, persistent_workers, prefetch_factor, sampler=None):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def resolve_amp_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    augment=None,
    phase="train",
    log_interval_steps=20,
    amp_enabled=False,
    amp_dtype=torch.float16,
    scaler=None,
    metric_config=None,
):
    model.train(optimizer is not None)
    metric_accumulator = PerceptionMetricAccumulator(**(metric_config or {}))
    totals = {}
    count = 0
    total_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        if augment is not None and optimizer is not None:
            batch = augment(batch)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                pred = model(batch["input"], batch.get("delta_t"))
                losses = criterion(pred, batch)
            metric_accumulator.update(pred, batch)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(losses["loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
        merged = losses
        bsz = batch["input"].shape[0]
        count += bsz
        for key, value in merged.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * bsz
        if log_interval_steps > 0 and (step == 1 or step % log_interval_steps == 0 or step == total_steps):
            avg_loss = totals.get("loss", 0.0) / max(count, 1)
            avg_iou = metric_accumulator.current_mask_iou()
            log(f"{phase} step {step}/{total_steps} samples={count} loss={avg_loss:.4f} iou={avg_iou:.4f}")
    stats = {key: value / max(count, 1) for key, value in totals.items()}
    stats.update(metric_accumulator.compute())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perception_training/config/default.yaml")
    parser.add_argument("--resume", help="Resume from a training checkpoint")
    args = parser.parse_args()
    log(f"loading config: {args.config}")
    cfg = load_config(args.config)

    evaluation_cfg = cfg.get("evaluation", {})
    rl_sensor = cfg["sensor"].get("rl", {"height": 6, "width": 36})
    metric_config = {
        "max_range_m": cfg["sensor"]["max_range"],
        "mask_threshold": evaluation_cfg.get("mask_threshold", 0.5),
        "distance_bins_m": evaluation_cfg.get("distance_bins_m", [0.0, 2.0, 5.0, 10.0]),
        "pool_output_size": tuple(
            evaluation_cfg.get("pool_output_size", [rl_sensor["height"], rl_sensor["width"]])
        ),
        "seam_width": evaluation_cfg.get("seam_width", 3),
        "static_speed_hist_max_mps": evaluation_cfg.get("static_speed_hist_max_mps", 10.0),
        "static_speed_hist_bins": evaluation_cfg.get("static_speed_hist_bins", 1024),
    }
    checkpoint_selection_cfg = evaluation_cfg.get("checkpoint_selection", {})

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["training"]["seed"])
    log(f"device={device} cuda_available={torch.cuda.is_available()}")
    amp_cfg = cfg["training"].get("amp", {})
    amp_requested = bool(amp_cfg.get("enabled", True))
    amp_enabled = amp_requested and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(amp_cfg.get("dtype", "float16"))
    cudnn_benchmark = bool(cfg["training"].get("cudnn_benchmark", False))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    matmul_precision = str(cfg["training"].get("matmul_precision", "highest"))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        allow_tf32 = bool(amp_cfg.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    else:
        allow_tf32 = False
    log(
        f"amp_enabled={amp_enabled} amp_dtype={str(amp_dtype).replace('torch.', '')} "
        f"amp_requested={amp_requested} allow_tf32={allow_tf32} "
        f"cudnn_benchmark={cudnn_benchmark} matmul_precision={matmul_precision}"
    )

    dataset_kwargs = {
        "default_delta_t": cfg["model"].get("reference_delta_t", 0.096),
        "min_delta_t": cfg["model"].get("min_delta_t", 0.05),
        "max_delta_t": cfg["model"].get("max_delta_t", 0.15),
        "cache_size": cfg["data"].get("cache_size", 4),
    }
    train_roots = cfg["data"].get("train_roots")
    val_roots = cfg["data"].get("val_roots")
    if train_roots is not None or val_roots is not None:
        if not train_roots or not val_roots:
            raise ValueError("data.train_roots and data.val_roots must both be non-empty")
        log(f"loading explicit train roots={train_roots}")
        log(f"loading explicit val roots={val_roots}")
        train_dataset = LidarPairDataset(train_roots, **dataset_kwargs)
        val_dataset = LidarPairDataset(val_roots, **dataset_kwargs)
        train_set = torch.utils.data.Subset(train_dataset, list(range(len(train_dataset))))
        val_set = torch.utils.data.Subset(val_dataset, list(range(len(val_dataset))))
        log(f"explicit datasets train={len(train_set)} val={len(val_set)}")
    else:
        log(f"loading dataset roots={cfg['data']['roots']}")
        dataset = LidarPairDataset(cfg["data"]["roots"], **dataset_kwargs)
        train_set, val_set = split_dataset(dataset, cfg["data"]["val_ratio"], cfg["training"]["seed"])
        log(f"dataset samples={len(dataset)} train={len(train_set)} val={len(val_set)}")
    pin_memory = bool(cfg["training"].get("pin_memory", True))
    persistent_workers = bool(cfg["training"].get("persistent_workers", False))
    prefetch_factor = int(cfg["training"].get("prefetch_factor", 2))
    log(
        f"dataloader batch_size={cfg['training']['batch_size']} num_workers={cfg['training']['num_workers']} "
        f"pin_memory={pin_memory} persistent_workers={persistent_workers} prefetch_factor={prefetch_factor}"
    )
    train_sampler = ShardAwareSampler(train_set, seed=cfg["training"]["seed"])
    train_loader = build_dataloader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        sampler=train_sampler,
    )
    val_loader = build_dataloader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    log("building model/loss/optimizer")
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
    loss_cfg = dict(cfg["loss"])
    loss_cfg.setdefault("max_range_m", cfg["sensor"]["max_range"])
    criterion = PerceptionLoss(**loss_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    augment = LidarAugment(**cfg["augmentation"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_score = -float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        log(f"resuming checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" not in checkpoint:
            raise ValueError("Resume checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", checkpoint.get("selection_score", -float("inf"))))
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler_source = "checkpoint"
        else:
            # Legacy checkpoints already contain the optimizer LR after the
            # completed epoch. Reconstruct CosineAnnealingLR's epoch counter
            # so the next scheduler step follows the original 80-epoch curve.
            scheduler.last_epoch = start_epoch
            scheduler.base_lrs = [float(cfg["training"]["lr"]) for _ in optimizer.param_groups]
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            scheduler_source = "legacy epoch/lr reconstruction"
        if amp_enabled and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        train_sampler.epoch = start_epoch
        if start_epoch >= int(cfg["training"]["epochs"]):
            raise ValueError(f"Checkpoint epoch {start_epoch} already reaches configured epochs")
        log(
            f"resume loaded completed_epochs={start_epoch} next_epoch={start_epoch + 1} "
            f"lr={optimizer.param_groups[0]['lr']:.10f} best_score={best_score:.6f} "
            f"scheduler={scheduler_source}"
        )

    run = None
    if cfg["wandb"]["enabled"]:
        log(f"initializing wandb mode={cfg['wandb']['mode']} project={cfg['wandb']['project']}")
        import wandb

        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            name=cfg["wandb"]["name"],
            mode=cfg["wandb"]["mode"],
            config=cfg,
        )

    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"starting training epochs={cfg['training']['epochs']} start_epoch={start_epoch + 1} "
        f"batch_size={cfg['training']['batch_size']}"
    )
    log_interval_steps = int(cfg["training"].get("log_interval_steps", 20))
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        log(f"epoch {epoch + 1}/{cfg['training']['epochs']} train")
        train_stats = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            augment,
            phase=f"train epoch {epoch + 1}",
            log_interval_steps=log_interval_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            metric_config=metric_config,
        )
        if len(val_set):
            log(f"epoch {epoch + 1}/{cfg['training']['epochs']} val")
            val_stats = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device,
                phase=f"val epoch {epoch + 1}",
                log_interval_steps=log_interval_steps,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                metric_config=metric_config,
            )
        else:
            val_stats = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_elapsed_seconds = time.perf_counter() - epoch_started
        scheduler.step()
        info = {f"train/{k}": v for k, v in train_stats.items()}
        info.update({f"val/{k}": v for k, v in val_stats.items()})
        info["lr"] = scheduler.get_last_lr()[0]
        info["epoch"] = epoch
        info["epoch_elapsed_seconds"] = epoch_elapsed_seconds
        info["train_samples_per_second"] = len(train_set) / max(epoch_elapsed_seconds, 1e-9)
        info["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
        )

        selection_metrics = val_stats if val_stats else train_stats
        selection_score = composite_selection_score(
            selection_metrics,
            dyn_max_vel_mps=checkpoint_selection_cfg.get("dyn_max_vel_mps", 5.0),
            iou_weight=checkpoint_selection_cfg.get("iou_weight", 1.0),
            recall_weight=checkpoint_selection_cfg.get("recall_weight", 1.0),
            velocity_weight=checkpoint_selection_cfg.get("velocity_weight", 1.0),
            static_speed_weight=checkpoint_selection_cfg.get("static_speed_weight", 1.0),
        )
        info["checkpoint_selection_score"] = selection_score
        log(str(info))
        if run is not None:
            run.log(info, step=epoch)

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "selection_score": selection_score,
            "best_score": max(best_score, selection_score),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        if selection_score > best_score:
            best_score = selection_score
            torch.save(ckpt, out_dir / "saumnet_best.pt")
            if run is not None:
                run.save(str(out_dir / "saumnet_best.pt"))
        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            torch.save(ckpt, out_dir / f"saumnet_epoch_{epoch + 1}.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": cfg["training"]["epochs"] - 1,
            "best_score": best_score,
        },
        out_dir / "saumnet_final.pt",
    )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
