#模型导出
import argparse
from pathlib import Path

import torch

from perception_training.model import SAUMNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    sensor = cfg.get("sensor", {"network": {"height": 16, "width": 180}, "vfov_deg": [-7.0, 52.0]})
    network_sensor = sensor.get("network", sensor)
    model_cfg = cfg.get(
        "model",
        {"base_channels": 16, "reference_delta_t": 0.096, "min_delta_t": 0.05, "max_delta_t": 0.15},
    )
    model = SAUMNet(
        height=network_sensor["height"],
        width=network_sensor["width"],
        vfov_deg=tuple(sensor["vfov_deg"]),
        base_channels=model_cfg["base_channels"],
        input_channels=model_cfg.get("input_channels", 4),
        reference_delta_t=model_cfg.get("reference_delta_t", 0.096),
        min_delta_t=model_cfg.get("min_delta_t", 0.05),
        max_delta_t=model_cfg.get("max_delta_t", 0.15),
    )
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    # Scripting preserves runtime device semantics. Tracing with PyTorch 2.0
    # records the CPU example device in ``Tensor.to`` and tensor factories,
    # producing an artifact that cannot subsequently run on CUDA.
    traced = torch.jit.script(model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)


if __name__ == "__main__":
    main()
