from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from duet_repro.core.task_vectors import StateDict, torch_load_checkpoint


class DuETIncrementalDetect(nn.Module):
    """YOLO Detect wrapper that keeps old and current task heads separate.

    The DuET paper treats the detector head as task-specific parameters and
    concatenates task heads after merging the shared backbone/neck. This wrapper
    implements that idea for a fixed global YOLO class space: each task head
    predicts only its own class indices, and inference concatenates detections
    from the old and current heads before Ultralytics NMS.
    """

    def __init__(
        self,
        old_head: nn.Module,
        new_head: nn.Module,
        *,
        old_class_indices: Iterable[int],
        new_class_indices: Iterable[int],
        total_classes: int,
    ) -> None:
        super().__init__()
        self.old_head = copy.deepcopy(old_head)
        self.new_head = copy.deepcopy(new_head)
        self.old_class_indices = [int(i) for i in old_class_indices]
        self.new_class_indices = [int(i) for i in new_class_indices]
        self.nc = int(total_classes)
        self.nl = int(getattr(new_head, "nl", getattr(old_head, "nl", 0)))
        self.reg_max = int(getattr(new_head, "reg_max", getattr(old_head, "reg_max", 16)))
        self.no = self.nc + self.reg_max * 4
        self.stride = getattr(new_head, "stride", getattr(old_head, "stride", torch.zeros(self.nl))).clone()
        self.export = False
        self.dynamic = bool(getattr(new_head, "dynamic", False))
        self.end2end = False

    def _head_pred(self, head: nn.Module, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return head.forward_head(x, **head.one2many)

    def _global_scores(
        self,
        local_scores: torch.Tensor,
        class_indices: list[int],
    ) -> torch.Tensor:
        scores = local_scores.sigmoid()
        global_scores = scores.new_zeros((scores.shape[0], self.nc, scores.shape[-1]))
        if class_indices:
            if scores.shape[1] == len(class_indices):
                global_scores[:, class_indices, :] = scores
            else:
                global_scores[:, class_indices, :] = scores[:, class_indices, :]
        return global_scores

    def _inference_one(
        self,
        head: nn.Module,
        pred: dict[str, torch.Tensor],
        class_indices: list[int],
    ) -> torch.Tensor:
        boxes = head._get_decode_boxes(pred)
        scores = self._global_scores(pred["scores"], class_indices)
        return torch.cat((boxes, scores), dim=1)

    def forward(self, x: list[torch.Tensor]):
        old_pred = self._head_pred(self.old_head, x)
        new_pred = self._head_pred(self.new_head, x)

        if self.training:
            return {"old": old_pred, "new": new_pred}

        old_y = self._inference_one(self.old_head, old_pred, self.old_class_indices)
        new_y = self._inference_one(self.new_head, new_pred, self.new_class_indices)
        y = torch.cat((old_y, new_y), dim=2)
        preds = {"old": old_pred, "new": new_pred}
        return y if self.export else (y, preds)


def _checkpoint_models(ckpt: object) -> list[nn.Module]:
    if isinstance(ckpt, nn.Module):
        return [ckpt]
    if isinstance(ckpt, dict):
        models = []
        for key in ("model", "ema"):
            value = ckpt.get(key)
            if isinstance(value, nn.Module):
                models.append(value)
        return models
    return []


def _detect_head(model: nn.Module, detect_index: int = -1) -> nn.Module:
    modules = getattr(model, "model")
    return modules[detect_index]


def _set_detect_head(model: nn.Module, head: nn.Module, detect_index: int = -1) -> None:
    modules = getattr(model, "model")
    old = modules[detect_index]
    for attr in ("i", "f", "type", "np"):
        if hasattr(old, attr):
            setattr(head, attr, getattr(old, attr))
    modules[detect_index] = head


def inject_incremental_head_checkpoint(
    *,
    template_checkpoint_path: str | Path,
    merged_shared_state: StateDict,
    old_checkpoint_path: str | Path,
    new_checkpoint_path: str | Path,
    output_path: str | Path,
    old_class_indices: Iterable[int],
    new_class_indices: Iterable[int],
    total_classes: int,
    detect_index: int = -1,
    map_location: str = "cpu",
) -> None:
    """Save a DuET checkpoint with merged shared layers and concatenated heads."""
    template_ckpt = torch_load_checkpoint(template_checkpoint_path, map_location=map_location)
    old_ckpt = torch_load_checkpoint(old_checkpoint_path, map_location=map_location)
    new_ckpt = torch_load_checkpoint(new_checkpoint_path, map_location=map_location)

    old_models = _checkpoint_models(old_ckpt)
    new_models = _checkpoint_models(new_ckpt)
    if not old_models or not new_models:
        raise TypeError("old/new checkpoint must contain an nn.Module under 'model' or 'ema'.")

    old_head = copy.deepcopy(_detect_head(old_models[0], detect_index)).float()
    new_head = copy.deepcopy(_detect_head(new_models[0], detect_index)).float()

    updated = False
    for model in _checkpoint_models(template_ckpt):
        model.load_state_dict(merged_shared_state, strict=False)
        duet_head = DuETIncrementalDetect(
            old_head,
            new_head,
            old_class_indices=old_class_indices,
            new_class_indices=new_class_indices,
            total_classes=total_classes,
        )
        _set_detect_head(model, duet_head, detect_index)
        model.float()
        updated = True

    if not updated:
        raise TypeError("template checkpoint must contain an nn.Module under 'model' or 'ema'.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template_ckpt, output_path)
