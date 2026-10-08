import argparse
from pathlib import Path
from typing import Dict

import torch


def append_samples(path: str, batch: Dict[str, torch.Tensor]) -> None:
    """Append a supervised batch to a .pt shard.

    This helper is intentionally tiny so it can be imported from Isaac env code:

        from perception_training.save_samples import append_samples
        append_samples("perception_training/data/shard_000.pt", batch)
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    if out_path.exists():
        samples = torch.load(out_path, map_location="cpu")
        if not isinstance(samples, list):
            samples = [samples]
    batch_cpu = {k: v.detach().cpu() for k, v in batch.items()}
    bsz = batch_cpu["input"].shape[0]
    for idx in range(bsz):
        samples.append({key: value[idx] for key, value in batch_cpu.items()})
    torch.save(samples, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="A .pt file containing a batch dict.")
    parser.add_argument("--output", required=True, help="Output shard path.")
    args = parser.parse_args()
    batch = torch.load(args.input, map_location="cpu")
    append_samples(args.output, batch)


if __name__ == "__main__":
    main()
