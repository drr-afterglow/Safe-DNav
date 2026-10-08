"""Strict loader for the final PPO training checkpoint."""

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Mapping, Optional

import torch

from .model import PolicyActor


FINAL_POLICY_SHA256 = "834affd0ecaebbe966254ef64ef322cea4ce5c241672a9dfe4de650c763fb8a0"


SOURCE_SHAPES = OrderedDict(
    [
        ("feature_extractor.module.0.module.0.weight", (4, 1, 5, 3)),
        ("feature_extractor.module.0.module.0.bias", (4,)),
        ("feature_extractor.module.0.module.2.weight", (16, 4, 5, 3)),
        ("feature_extractor.module.0.module.2.bias", (16,)),
        ("feature_extractor.module.0.module.4.weight", (16, 16, 5, 3)),
        ("feature_extractor.module.0.module.4.bias", (16,)),
        ("feature_extractor.module.0.module.7.weight", (128, 576)),
        ("feature_extractor.module.0.module.7.bias", (128,)),
        ("feature_extractor.module.0.module.8.weight", (128,)),
        ("feature_extractor.module.0.module.8.bias", (128,)),
        ("feature_extractor.module.1.module.0.weight", (8, 5, 5, 3)),
        ("feature_extractor.module.1.module.0.bias", (8,)),
        ("feature_extractor.module.1.module.2.weight", (16, 8, 5, 3)),
        ("feature_extractor.module.1.module.2.bias", (16,)),
        ("feature_extractor.module.1.module.4.weight", (16, 16, 5, 3)),
        ("feature_extractor.module.1.module.4.bias", (16,)),
        ("feature_extractor.module.1.module.7.weight", (128, 576)),
        ("feature_extractor.module.1.module.7.bias", (128,)),
        ("feature_extractor.module.1.module.8.weight", (128,)),
        ("feature_extractor.module.1.module.8.bias", (128,)),
        ("feature_extractor.module.3.module.0.weight", (256, 269)),
        ("feature_extractor.module.3.module.0.bias", (256,)),
        ("feature_extractor.module.3.module.2.weight", (256,)),
        ("feature_extractor.module.3.module.2.bias", (256,)),
        ("feature_extractor.module.3.module.3.weight", (256, 256)),
        ("feature_extractor.module.3.module.3.bias", (256,)),
        ("feature_extractor.module.3.module.5.weight", (256,)),
        ("feature_extractor.module.3.module.5.bias", (256,)),
        ("actor.module.0.module.alpha_layer.weight", (3, 256)),
        ("actor.module.0.module.alpha_layer.bias", (3,)),
        ("actor.module.0.module.beta_layer.weight", (3, 256)),
        ("actor.module.0.module.beta_layer.bias", (3,)),
        ("critic.module.weight", (1, 256)),
        ("critic.module.bias", (1,)),
        ("value_norm.running_mean", (1,)),
        ("value_norm.running_mean_sq", (1,)),
        ("value_norm.debiasing_term", ()),
        ("gae.gamma", ()),
        ("gae.lmbda", ()),
    ]
)


DEPLOYMENT_MAPPING = OrderedDict()
for local_prefix, source_prefix in (
    ("static_encoder", "feature_extractor.module.0.module"),
    ("dynamic_encoder", "feature_extractor.module.1.module"),
):
    for index in (0, 2, 4, 7, 8):
        for suffix in ("weight", "bias"):
            local = f"{local_prefix}.{index}.{suffix}"
            source = f"{source_prefix}.{index}.{suffix}"
            if source in SOURCE_SHAPES:
                DEPLOYMENT_MAPPING[local] = source
for index in (0, 2, 3, 5):
    for suffix in ("weight", "bias"):
        source = f"feature_extractor.module.3.module.{index}.{suffix}"
        if source in SOURCE_SHAPES:
            DEPLOYMENT_MAPPING[f"fusion.{index}.{suffix}"] = source
for name in ("alpha_layer", "beta_layer"):
    for suffix in ("weight", "bias"):
        DEPLOYMENT_MAPPING[f"actor.{name}.{suffix}"] = f"actor.module.0.module.{name}.{suffix}"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unwrap(payload) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "policy_state_dict" in payload:
        payload = payload["policy_state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a state dictionary")
    return payload


def validate_source_state_dict(state_dict: Mapping[str, torch.Tensor]) -> None:
    actual_keys = set(state_dict)
    expected_keys = set(SOURCE_SHAPES)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"checkpoint key mismatch; missing={missing}, unexpected={unexpected}")
    for key, shape in SOURCE_SHAPES.items():
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint value {key} is not a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"checkpoint shape mismatch for {key}: expected {shape}, got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"checkpoint value {key} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint value {key} contains NaN or Inf")


def load_policy_checkpoint(
    model: PolicyActor,
    path,
    *,
    expected_sha256: Optional[str] = FINAL_POLICY_SHA256,
) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"policy SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    state_dict = _unwrap(torch.load(str(path), map_location="cpu"))
    validate_source_state_dict(state_dict)
    deployment_state = OrderedDict(
        (local, state_dict[source].detach().clone())
        for local, source in DEPLOYMENT_MAPPING.items()
    )
    model.load_state_dict(deployment_state, strict=True)
    return digest
