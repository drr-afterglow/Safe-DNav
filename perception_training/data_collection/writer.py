from pathlib import Path

import torch
import yaml


class ShardWriter:
    def __init__(self, output_dir, shard_size, overwrite=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for shard_path in self.output_dir.glob("shard_*.pt"):
                shard_path.unlink()
        self.shard_size = int(shard_size)
        self.buffer = []
        self.shard_idx = 0
        self.num_samples = 0

    def add_batch(self, batch):
        batch_cpu = {k: v.detach().cpu() for k, v in batch.items()}
        bsz = batch_cpu["input"].shape[0]
        for idx in range(bsz):
            self.buffer.append({key: value[idx] for key, value in batch_cpu.items()})
            self.num_samples += 1
            if len(self.buffer) >= self.shard_size:
                self.flush()

    def flush(self):
        if not self.buffer:
            return
        out_path = self.output_dir / f"shard_{self.shard_idx:05d}.pt"
        torch.save(self.buffer, out_path)
        print(f"[perception-data] saved {len(self.buffer)} samples to {out_path}")
        self.buffer = []
        self.shard_idx += 1

    def write_manifest(self, cfg, stats):
        manifest = {
            "num_samples": self.num_samples,
            "num_shards": self.shard_idx,
            "shard_size": self.shard_size,
            "stats": stats,
            "config": cfg,
        }
        out_path = self.output_dir / "manifest.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
        print(f"[perception-data] wrote manifest to {out_path}")
