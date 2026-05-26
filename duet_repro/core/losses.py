from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def direction_consistency_loss(old_vector: Tensor, new_vector: Tensor, eps: float = 1e-8) -> Tensor:
    """Penalize opposite directions between two task vectors."""
    old_flat = old_vector.flatten()
    new_flat = new_vector.flatten()
    cosine = torch.dot(old_flat, new_flat) / (old_flat.norm() * new_flat.norm() + eps)
    return torch.relu(-cosine)


def dynamic_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
    weight: float = 1.0,
) -> Tensor:
    """KL distillation loss for keeping old-task predictions stable."""
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    return weight * (temperature**2) * loss


def duet_modified_distillation_loss(
    student_cls: Tensor,
    teacher_cls: Tensor,
    student_boxes: Tensor,
    teacher_boxes: Tensor,
    *,
    cls_quantile: float = 0.75,
    bbox_quantile: float = 0.75,
    weight: float = 1.0,
) -> Tensor:
    """DuET-style dynamic distillation loss from the supplementary material.

    Classification predictions with low old-model confidence are filtered out.
    Bounding-box predictions with high old-model coordinate variance are filtered
    out. The remaining classification logits use MSE; bounding boxes use KL over
    coordinate distributions, matching the paper's loss formulation.
    """
    with torch.no_grad():
        teacher_conf = teacher_cls.softmax(dim=-1).amax(dim=-1)
        cls_threshold = torch.quantile(teacher_conf.flatten(), cls_quantile)
        cls_mask = teacher_conf >= cls_threshold

        teacher_var = teacher_boxes.var(dim=-1)
        bbox_threshold = torch.quantile(teacher_var.flatten(), bbox_quantile)
        bbox_mask = teacher_var <= bbox_threshold

    cls_loss = student_cls.new_tensor(0.0)
    if cls_mask.any():
        cls_loss = F.mse_loss(student_cls[cls_mask], teacher_cls.detach()[cls_mask])

    bbox_loss = student_boxes.new_tensor(0.0)
    if bbox_mask.any():
        student_box_log_prob = F.log_softmax(student_boxes[bbox_mask], dim=-1)
        teacher_box_prob = F.softmax(teacher_boxes.detach()[bbox_mask], dim=-1)
        bbox_loss = F.kl_div(student_box_log_prob, teacher_box_prob, reduction="batchmean")

    return weight * (cls_loss + bbox_loss)


class FeatureHook:
    """Small helper for feature-level distillation experiments."""

    def __init__(self, module: nn.Module):
        self.output: Tensor | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.output = output

    def close(self) -> None:
        self.handle.remove()


def feature_l2_distillation(student_feature: Tensor, teacher_feature: Tensor, weight: float = 1.0) -> Tensor:
    return weight * F.mse_loss(student_feature, teacher_feature.detach())
