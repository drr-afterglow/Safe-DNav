#读取训练数据
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset


REQUIRED_KEYS = ("input", "mask", "velocity")


class LidarPairDataset(Dataset):
    """Dataset for SA-UMNet.

    Supported sample formats:
      - .pt/.pth: dict or list of dicts
      - .npz: arrays with keys input, mask, velocity, optional valid

    Expected tensor shapes per sample:
      input: (4, H, W), ordered as current/previous range then valid maps.
      Legacy two-channel samples are upgraded in memory without rewriting files.
      mask: (1, H, W)
      velocity: (3, H, W)
      valid: (1, H, W), optional
      delta_t: scalar seconds between frames, optional for legacy samples
      ego_velocity_body_mps: (3,) current FLU ego velocity, optional for legacy samples
    """

    def __init__(
        self,
        roots: Iterable[str],
        default_delta_t: float = 0.096,
        min_delta_t: float = 0.05,
        max_delta_t: float = 0.15,
        cache_size: int = 4,
    ):
        self.default_delta_t = float(default_delta_t)
        self.min_delta_t = float(min_delta_t)
        self.max_delta_t = float(max_delta_t)
        self.cache_size = max(int(cache_size), 1)
        roots = list(roots)
        self.files: List[Path] = []
        for root in roots:
            path = Path(root)
            if path.is_file():
                self.files.append(path)
            elif path.is_dir():
                self.files.extend(sorted(path.rglob("*.pt")))
                self.files.extend(sorted(path.rglob("*.pth")))
                self.files.extend(sorted(path.rglob("*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .pt/.pth/.npz samples found in {list(roots)}")

        self.index = []
        self._cache = OrderedDict()
        for file_idx, file_path in enumerate(self.files):
            if file_path.suffix == ".npz":
                self.index.append((file_idx, None))
                continue
            obj = torch.load(file_path, map_location="cpu")
            if isinstance(obj, (list, tuple)):
                self.index.extend((file_idx, i) for i in range(len(obj)))
            else:
                self.index.append((file_idx, None))

    def __len__(self) -> int:
        return len(self.index)

    def _load_file(self, file_idx: int):
        if file_idx in self._cache:
            data = self._cache.pop(file_idx)
            self._cache[file_idx] = data
            return data
        file_path = self.files[file_idx]
        if file_path.suffix == ".npz":
            data = dict(np.load(file_path))
        else:
            data = torch.load(file_path, map_location="cpu")
        self._cache[file_idx] = data
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, item_idx = self.index[idx]
        data = self._load_file(file_idx)
        if item_idx is not None:
            data = data[item_idx]

        sample = {key: torch.as_tensor(data[key]).float() for key in REQUIRED_KEYS}
        if sample["input"].ndim != 3 or sample["input"].shape[0] not in (2, 4):
            raise ValueError(f"input in {self.files[file_idx]} must have shape (2|4, H, W)")
        if sample["input"].shape[0] == 2:
            # Compatibility for the 3000-sample probe only. Formal collection
            # writes explicit valid maps from ray-hit truth.
            return_valid = (sample["input"] < 1.0 - 1e-6).to(sample["input"].dtype)
            sample["input"] = torch.cat([sample["input"], return_valid], dim=0)
        sample["valid"] = torch.as_tensor(data.get("valid", sample["input"][2:3])).float()
        sample["delta_t"] = torch.as_tensor(data.get("delta_t", self.default_delta_t)).float().reshape(())
        if "scene_id" in data:
            sample["scene_id"] = torch.as_tensor(data["scene_id"]).long().reshape(())
        if "ego_velocity_body_mps" in data:
            ego_velocity = torch.as_tensor(data["ego_velocity_body_mps"]).float().reshape(-1)
            if ego_velocity.numel() != 3 or not torch.isfinite(ego_velocity).all():
                raise ValueError(
                    f"ego_velocity_body_mps in {self.files[file_idx]} must be a finite 3-vector"
                )
            sample["ego_velocity_body_mps"] = ego_velocity
        for metadata_key in (
            "ego_linear_speed_mps",
            "ego_angular_speed_deg_s",
            "ego_translation_m",
            "ego_rotation_deg",
            "alignment_position_error_m",
            "alignment_rotation_error_deg",
        ):
            if metadata_key in data:
                sample[metadata_key] = torch.as_tensor(data[metadata_key]).float().reshape(())
        delta_t = float(sample["delta_t"])
        if not self.min_delta_t <= delta_t <= self.max_delta_t:
            raise ValueError(
                f"delta_t={delta_t:.6f}s in {self.files[file_idx]} is outside "
                f"[{self.min_delta_t:.6f}, {self.max_delta_t:.6f}]s"
            )
        return sample


class ShardAwareSampler(Sampler[int]):
    """Shuffle shards and samples while reading each shard contiguously."""

    def __init__(self, dataset: Dataset, seed: int = 0):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.groups = self._build_groups(dataset)

    @staticmethod
    def _build_groups(dataset: Dataset):
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, LidarPairDataset):
            parent = dataset.dataset
            groups = {}
            for subset_pos, parent_idx in enumerate(dataset.indices):
                file_idx, _ = parent.index[parent_idx]
                groups.setdefault(file_idx, []).append(subset_pos)
            return list(groups.values())
        return [list(range(len(dataset)))]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        group_order = torch.randperm(len(self.groups), generator=generator).tolist()
        for group_idx in group_order:
            group = self.groups[group_idx]
            order = torch.randperm(len(group), generator=generator).tolist()
            for item_idx in order:
                yield group[item_idx]

    def __len__(self):
        return len(self.dataset)


def split_dataset(dataset: Dataset, val_ratio: float, seed: int):
    if val_ratio <= 0:
        return torch.utils.data.Subset(dataset, list(range(len(dataset)))), torch.utils.data.Subset(dataset, [])

    # Prefer whole-scene validation so random poses from the same procedural
    # map cannot leak into both splits. Legacy/single-scene data falls back to
    # deterministic random splitting.
    if isinstance(dataset, LidarPairDataset):
        scene_to_indices = {}
        all_have_scene = True
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if "scene_id" not in sample:
                all_have_scene = False
                break
            scene_to_indices.setdefault(int(sample["scene_id"]), []).append(idx)
        if all_have_scene and len(scene_to_indices) >= 2:
            scenes = sorted(scene_to_indices)
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(scenes), generator=generator).tolist()
            target_val = max(1, int(round(len(dataset) * val_ratio)))
            val_scenes = set()
            val_count = 0
            for order_idx in order:
                if len(val_scenes) >= len(scenes) - 1:
                    break
                scene = scenes[order_idx]
                val_scenes.add(scene)
                val_count += len(scene_to_indices[scene])
                if val_count >= target_val:
                    break
            val_indices = [idx for scene in val_scenes for idx in scene_to_indices[scene]]
            train_indices = [
                idx for scene in scenes if scene not in val_scenes for idx in scene_to_indices[scene]
            ]
            return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)

    val_len = int(round(len(dataset) * val_ratio))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
