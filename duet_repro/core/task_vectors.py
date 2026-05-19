from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn


StateDict = dict[str, Tensor]


def torch_load_checkpoint(path: str | Path, map_location: str = "cpu") -> object:
    """Load trusted local training checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class MergeReport:
    merged_keys: int
    skipped_keys: int
    incompatible_keys: list[str]


def unwrap_checkpoint(ckpt: object) -> StateDict:
    """Return a tensor state_dict from common PyTorch and Ultralytics checkpoints."""
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if isinstance(ckpt, Mapping):
        for key in ("model", "ema", "state_dict"):
            value = ckpt.get(key)
            if isinstance(value, nn.Module):
                return value.state_dict()
            if isinstance(value, Mapping):
                return {str(k): v for k, v in value.items() if torch.is_tensor(v)}
        return {str(k): v for k, v in ckpt.items() if torch.is_tensor(v)}

    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def load_state_dict(path: str | Path, map_location: str = "cpu") -> StateDict:
    ckpt = torch_load_checkpoint(path, map_location=map_location)
    return unwrap_checkpoint(ckpt)


def key_is_shared(key: str, exclude_patterns: Iterable[str]) -> bool:
    lowered = key.lower()
    return not any(pattern.lower() in lowered for pattern in exclude_patterns)


def task_vector(reference: StateDict, target: StateDict) -> StateDict:
    """Compute theta_target - theta_reference for compatible tensor parameters."""
    delta: StateDict = {}
    for key, ref_value in reference.items():
        tgt_value = target.get(key)
        if tgt_value is None:
            continue
        if ref_value.shape != tgt_value.shape:
            continue
        if not torch.is_floating_point(ref_value):
            continue
        delta[key] = tgt_value.detach().float() - ref_value.detach().float()
    return delta


def cosine_direction_loss(delta_old: StateDict, delta_new: StateDict, eps: float = 1e-8) -> Tensor:
    """Direction-consistency penalty used to discourage conflicting task vectors.

    The paper introduces a direction-consistency objective to reduce sign conflicts
    between task vectors. This implementation uses the negative cosine part as a
    stable PyTorch-friendly surrogate: aligned vectors contribute zero, conflicting
    vectors are penalized.
    """
    losses: list[Tensor] = []
    for key, old_value in delta_old.items():
        new_value = delta_new.get(key)
        if new_value is None or old_value.shape != new_value.shape:
            continue
        old_flat = old_value.flatten()
        new_flat = new_value.flatten()
        denom = old_flat.norm() * new_flat.norm() + eps
        cos = torch.dot(old_flat, new_flat) / denom
        losses.append(torch.relu(-cos))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def merge_state_dicts(
    reference: StateDict,
    old_model: StateDict,
    new_model: StateDict,
    *,
    alpha_old: float = 1.0,
    alpha_new: float = 1.0,
    shared_key_exclude: Iterable[str] = (),
) -> tuple[StateDict, MergeReport]:
    """Merge old and new task vectors into a single model.

    Shared parameters receive both old and new task vectors. Task-specific keys
    such as detection heads are kept from the new model, which is the practical
    choice when using a fixed-class YOLO head for course reproduction.
    """
    old_delta = task_vector(reference, old_model)
    new_delta = task_vector(reference, new_model)
    merged: StateDict = {}
    incompatible: list[str] = []
    merged_keys = 0
    skipped_keys = 0

    all_keys = set(reference) | set(old_model) | set(new_model)
    for key in sorted(all_keys):
        ref_value = reference.get(key)
        old_value = old_model.get(key)
        new_value = new_model.get(key)

        if new_value is None:
            skipped_keys += 1
            continue

        if (
            ref_value is not None
            and old_value is not None
            and ref_value.shape == old_value.shape == new_value.shape
            and torch.is_floating_point(ref_value)
            and key_is_shared(key, shared_key_exclude)
        ):
            merged[key] = (
                ref_value.detach().float()
                + alpha_old * old_delta[key]
                + alpha_new * new_delta[key]
            ).to(dtype=new_value.dtype)
            merged_keys += 1
        else:
            if ref_value is not None and ref_value.shape != new_value.shape:
                incompatible.append(key)
            merged[key] = new_value.detach().clone()
            skipped_keys += 1

    return merged, MergeReport(merged_keys, skipped_keys, incompatible)


def inject_state_dict_into_checkpoint(
    checkpoint_path: str | Path,
    merged_state: StateDict,
    output_path: str | Path,
    map_location: str = "cpu",
) -> None:
    """Save an Ultralytics/PyTorch checkpoint after replacing model weights."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    ckpt = torch_load_checkpoint(checkpoint_path, map_location=map_location)

    if isinstance(ckpt, nn.Module):
        ckpt.load_state_dict(merged_state, strict=False)
    elif isinstance(ckpt, dict):
        updated = False
        for key in ("model", "ema"):
            value = ckpt.get(key)
            if isinstance(value, nn.Module):
                value.load_state_dict(merged_state, strict=False)
                updated = True
        if not updated:
            ckpt["state_dict"] = merged_state
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
