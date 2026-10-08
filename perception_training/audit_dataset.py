"""Audit SA-UMNet shard datasets and report coverage statistics."""

import argparse
from pathlib import Path

import torch
import yaml


def discover(roots):
    shards = []
    for root in map(Path, roots):
        if root.is_file():
            shards.append(root)
        else:
            shards.extend(root.rglob("shard_*.pt"))
    shards = sorted(set(shards))
    if not shards:
        raise FileNotFoundError(f"No shards found under {roots}")
    return shards


def first_scalar(sample, key):
    value = torch.as_tensor(sample.get(key, 0.0)).reshape(-1)
    return float(value[0]) if value.numel() else 0.0


def audit(roots, max_range=10.0, seam_width=3):
    totals = {
        "samples": 0, "empty_samples": 0, "dynamic_pixels": 0,
        "far_dynamic_pixels": 0, "fast_dynamic_pixels": 0,
        "seam_dynamic_pixels": 0, "translation_ge_0_2m": 0,
        "rotation_ge_8deg": 0,
    }
    scenes = set()
    errors = []
    for shard in discover(roots):
        data = torch.load(shard, map_location="cpu")
        for index, sample in enumerate(data):
            tag = f"{shard}[{index}]"
            x = torch.as_tensor(sample["input"]).float()
            mask = torch.as_tensor(sample["mask"]).float()
            velocity = torch.as_tensor(sample["velocity"]).float()
            if tuple(x.shape[:1]) != (4,) or mask.shape[0] != 1 or velocity.shape[0] != 3:
                errors.append(f"{tag}: invalid channel shape")
                continue
            if not (torch.isfinite(x).all() and torch.isfinite(mask).all() and torch.isfinite(velocity).all()):
                errors.append(f"{tag}: non-finite tensor")
            if "ego_velocity_body_mps" in sample:
                ego_velocity = torch.as_tensor(sample["ego_velocity_body_mps"]).float().reshape(-1)
                if ego_velocity.numel() != 3 or not torch.isfinite(ego_velocity).all():
                    errors.append(f"{tag}: ego_velocity_body_mps must be a finite 3-vector")
                elif "ego_linear_speed_mps" in sample:
                    stored_speed = first_scalar(sample, "ego_linear_speed_mps")
                    if abs(float(torch.linalg.norm(ego_velocity)) - stored_speed) > 1e-4:
                        errors.append(f"{tag}: ego velocity vector/norm mismatch")
            binary_mask = mask[0] > 0.5
            current_valid = x[2] > 0.5
            if (binary_mask & ~current_valid).any():
                errors.append(f"{tag}: dynamic mask outside current valid return")
            if velocity[:, ~binary_mask].abs().max().item() > 1e-6:
                errors.append(f"{tag}: nonzero velocity outside dynamic mask")
            depth = x[0] * max_range
            speed = velocity.square().sum(0).sqrt()
            dyn_count = int(binary_mask.sum())
            totals["samples"] += 1
            totals["empty_samples"] += int(dyn_count == 0)
            totals["dynamic_pixels"] += dyn_count
            totals["far_dynamic_pixels"] += int((binary_mask & (depth >= 5.0) & (depth <= max_range)).sum())
            totals["fast_dynamic_pixels"] += int((binary_mask & (speed >= 1.5)).sum())
            totals["seam_dynamic_pixels"] += int(binary_mask[:, :seam_width].sum() + binary_mask[:, -seam_width:].sum())
            totals["translation_ge_0_2m"] += int(first_scalar(sample, "ego_translation_m") >= 0.2)
            totals["rotation_ge_8deg"] += int(first_scalar(sample, "ego_rotation_deg") >= 8.0)
            scenes.add(int(first_scalar(sample, "scene_id")))
    samples = max(totals["samples"], 1)
    dynamic = max(totals["dynamic_pixels"], 1)
    result = dict(totals)
    result.update({
        "scene_ids": sorted(scenes),
        "empty_sample_fraction": totals["empty_samples"] / samples,
        "far_dynamic_pixel_fraction": totals["far_dynamic_pixels"] / dynamic,
        "fast_dynamic_pixel_fraction": totals["fast_dynamic_pixels"] / dynamic,
        "seam_dynamic_pixel_fraction": totals["seam_dynamic_pixels"] / dynamic,
        "translation_ge_0_2m_fraction": totals["translation_ge_0_2m"] / samples,
        "rotation_ge_8deg_fraction": totals["rotation_ge_8deg"] / samples,
        "validation_errors": errors,
        "passed": not errors,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--max-range", type=float, default=10.0)
    parser.add_argument("--seam-width", type=int, default=3)
    args = parser.parse_args()
    report = audit(args.roots, args.max_range, args.seam_width)
    text = yaml.safe_dump(report, sort_keys=False)
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
