"""Render diagnostic panels for collected SA-UMNet samples."""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize four-channel SA-UMNet samples")
    parser.add_argument("roots", nargs="+", help="Dataset directories or shard files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-range", type=float, default=10.0)
    return parser.parse_args()


def discover_shards(roots: List[str]) -> List[Path]:
    shards = []
    for root in roots:
        path = Path(root)
        if path.is_file() and path.suffix in (".pt", ".pth"):
            shards.append(path)
        elif path.is_dir():
            shards.extend(sorted(path.rglob("shard_*.pt")))
    shards = sorted(set(shards))
    if not shards:
        raise FileNotFoundError(f"No shard files found under {roots}")
    return shards


def build_index(shards: List[Path]) -> List[Tuple[Path, int]]:
    index = []
    for shard in shards:
        data = torch.load(shard, map_location="cpu")
        if not isinstance(data, (list, tuple)):
            data = [data]
        index.extend((shard, item) for item in range(len(data)))
    return index


def scalar(sample: Dict, key: str, default=float("nan")) -> float:
    if key not in sample:
        return default
    value = torch.as_tensor(sample[key]).reshape(-1)
    return float(value[0]) if value.numel() else default


def add_image(fig, ax, array, title, cmap, vmin=None, vmax=None, colorbar=True):
    image = ax.imshow(array, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("azimuth bin")
    ax.set_ylabel("elevation bin")
    if colorbar:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return image


def render_sample(sample: Dict, output_path: Path, max_range: float, label: str):
    x = torch.as_tensor(sample["input"]).float().cpu()
    if x.shape[0] != 4:
        raise ValueError(f"Expected four-channel input, got {tuple(x.shape)}")
    mask = torch.as_tensor(sample["mask"]).float().cpu()[0]
    velocity = torch.as_tensor(sample["velocity"]).float().cpu()
    current = x[0] * max_range
    previous = x[1] * max_range
    current_valid = x[2] > 0.5
    previous_valid = x[3] > 0.5
    common_valid = current_valid & previous_valid
    difference = torch.where(common_valid, current - previous, torch.full_like(current, float("nan")))
    dynamic = mask > 0.5
    speed = torch.linalg.vector_norm(velocity, dim=0)
    velocity_limit = max(float(velocity[:, dynamic].abs().max()) if dynamic.any() else 1.0, 0.25)

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    add_image(fig, axes[0, 0], current, "Current depth (m)", "viridis", 0.0, max_range)
    add_image(fig, axes[0, 1], previous, "Aligned previous depth (m)", "viridis", 0.0, max_range)
    add_image(fig, axes[0, 2], difference, "Current - previous (m)", "coolwarm", -2.0, 2.0)
    add_image(fig, axes[0, 3], mask, "Dynamic mask", "gray", 0.0, 1.0, colorbar=False)
    add_image(fig, axes[1, 0], current_valid, "Current return valid", "gray", 0.0, 1.0, colorbar=False)
    add_image(fig, axes[1, 1], previous_valid, "Previous return valid", "gray", 0.0, 1.0, colorbar=False)
    for channel, (row, col), name in zip(range(3), ((1, 2), (1, 3), (2, 0)), ("v_r", "v_theta", "v_phi")):
        values = torch.where(dynamic, velocity[channel], torch.full_like(mask, float("nan")))
        add_image(fig, axes[row, col], values, f"{name} (m/s)", "coolwarm", -velocity_limit, velocity_limit)
    speed_values = torch.where(dynamic, speed, torch.full_like(mask, float("nan")))
    add_image(fig, axes[2, 1], speed_values, "Speed magnitude (m/s)", "magma", 0.0, velocity_limit)

    axes[2, 2].imshow(current, aspect="auto", cmap="gray", vmin=0.0, vmax=max_range, origin="upper")
    axes[2, 2].imshow(np.ma.masked_where(~dynamic.numpy(), mask.numpy()), aspect="auto", cmap="autumn", alpha=0.75, vmin=0.0, vmax=1.0, origin="upper")
    axes[2, 2].set_title("Current depth + dynamic overlay", fontsize=9)
    axes[2, 2].set_xlabel("azimuth bin")
    axes[2, 2].set_ylabel("elevation bin")

    axes[2, 3].axis("off")
    info = [
        label,
        f"scene_id: {int(scalar(sample, 'scene_id', -1))}",
        f"delta_t: {scalar(sample, 'delta_t'):.3f} s",
        f"dynamic pixels: {int(mask.sum())}",
        f"valid occupancy: {float(current_valid.float().mean()):.3f}",
        f"ego translation: {scalar(sample, 'ego_translation_m'):.3f} m",
        f"ego rotation: {scalar(sample, 'ego_rotation_deg'):.3f} deg",
        f"alignment position error: {scalar(sample, 'alignment_position_error_m'):.4f} m",
        f"alignment rotation error: {scalar(sample, 'alignment_rotation_error_deg'):.3f} deg",
    ]
    axes[2, 3].text(0.02, 0.98, "\n".join(info), va="top", family="monospace", fontsize=10)
    fig.suptitle("SA-UMNet supervised sample diagnostic", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def render_contact_sheet(records, output_path: Path, max_range: float):
    columns = 5
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(20, max(rows * 2.3, 3)), squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (sample, label) in zip(axes.flat, records):
        x = torch.as_tensor(sample["input"]).float()
        mask = torch.as_tensor(sample["mask"]).float()[0]
        ax.imshow(x[0] * max_range, aspect="auto", cmap="gray", vmin=0.0, vmax=max_range, origin="upper")
        ax.imshow(np.ma.masked_where(mask.numpy() <= 0.5, mask.numpy()), aspect="auto", cmap="autumn", alpha=0.75, vmin=0.0, vmax=1.0, origin="upper")
        ax.set_title(f"{label} | dyn={int(mask.sum())}", fontsize=7)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Pilot samples: current depth with dynamic-mask overlay", fontsize=14)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    shards = discover_shards(args.roots)
    index = build_index(shards)
    count = min(max(args.count, 1), len(index))
    selected = random.Random(args.seed).sample(index, count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    records = []
    manifest_records = []
    for order, (shard, item_index) in enumerate(selected):
        if shard not in cache:
            cache[shard] = torch.load(shard, map_location="cpu")
        sample = cache[shard][item_index]
        scene_id = int(scalar(sample, "scene_id", -1))
        label = f"scene={scene_id} {shard.name}[{item_index}]"
        filename = f"sample_{order:03d}_scene_{scene_id}_{shard.stem}_{item_index:04d}.png"
        render_sample(sample, output_dir / filename, args.max_range, label)
        records.append((sample, label))
        manifest_records.append({"image": filename, "scene_id": scene_id, "shard": str(shard), "item_index": item_index})
        print(f"[visualize] wrote {filename}", flush=True)
    render_contact_sheet(records, output_dir / "contact_sheet.png", args.max_range)
    (output_dir / "selection.yaml").write_text(yaml.safe_dump({"seed": args.seed, "samples": manifest_records}, sort_keys=False), encoding="utf-8")
    print(f"[visualize] wrote {count} panels and contact_sheet.png to {output_dir}")


if __name__ == "__main__":
    main()
