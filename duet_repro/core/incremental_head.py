from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from duet_repro.core.task_vectors import StateDict, torch_load_checkpoint


class DuETIncrementalDetect(nn.Module):
    """把 YOLO Detect 包装成“旧任务 head + 新任务 head”的增量检测头。

    DuET 将 backbone/neck 视为共享参数，将检测头视为任务特定参数。
    本类保留旧任务 head 和当前任务 head：训练时分别返回两套预测，推理时
    把两套预测映射到同一个全局类别空间后拼接，再交给 Ultralytics 后处理。
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

        # 深拷贝 head，避免后续对包装模型的训练或保存意外修改原 checkpoint。
        self.old_head = copy.deepcopy(old_head)
        self.new_head = copy.deepcopy(new_head)

        # old/new class indices 是局部 head 输出到全局类别空间的映射表。
        self.old_class_indices = [int(i) for i in old_class_indices]
        self.new_class_indices = [int(i) for i in new_class_indices]

        # Ultralytics Detect head 依赖 nc/no/stride 等属性做解码和 NMS；
        # 包装后需要暴露同名属性，才能像普通 Detect head 一样工作。
        self.nc = int(total_classes)
        self.nl = int(getattr(new_head, "nl", getattr(old_head, "nl", 0)))
        self.reg_max = int(getattr(new_head, "reg_max", getattr(old_head, "reg_max", 16)))
        self.no = self.nc + self.reg_max * 4
        self.stride = getattr(new_head, "stride", getattr(old_head, "stride", torch.zeros(self.nl))).clone()
        self.export = False
        self.dynamic = bool(getattr(new_head, "dynamic", False))
        self.end2end = False

    def _head_pred(self, head: nn.Module, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """调用 Ultralytics v10/v11 Detect head 的 one2many 分支。"""
        return head.forward_head(x, **head.one2many)

    def _global_scores(
        self,
        local_scores: torch.Tensor,
        class_indices: list[int],
    ) -> torch.Tensor:
        """把局部 head 的类别分数填回全局类别维度。

        例如旧 head 只预测 1-10 类，新 head 只预测 11-20 类，推理时需要变成
        total_classes 维，否则两套 head 的输出不能直接拼接和统一 NMS。
        """
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
        """解码单个 head 的 box，并拼接映射后的全局类别分数。"""
        boxes = head._get_decode_boxes(pred)
        scores = self._global_scores(pred["scores"], class_indices)
        return torch.cat((boxes, scores), dim=1)

    def forward(self, x: list[torch.Tensor]):
        """训练返回分支字典，推理返回拼接后的检测结果。"""
        old_pred = self._head_pred(self.old_head, x)
        new_pred = self._head_pred(self.new_head, x)

        if self.training:
            # 训练阶段保留分支结构，方便损失函数分别处理旧/新任务预测。
            return {"old": old_pred, "new": new_pred}

        # 推理阶段把两个 head 的 anchor 维拼接，后续 NMS 会在全局类别空间统一筛选。
        old_y = self._inference_one(self.old_head, old_pred, self.old_class_indices)
        new_y = self._inference_one(self.new_head, new_pred, self.new_class_indices)
        y = torch.cat((old_y, new_y), dim=2)
        preds = {"old": old_pred, "new": new_pred}
        return y if self.export else (y, preds)


def _checkpoint_models(ckpt: object) -> list[nn.Module]:
    """从 Ultralytics checkpoint 中取出可修改的 model/ema 模型对象。"""
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
    """按索引获取 YOLO 模型的 Detect head，默认最后一层。"""
    modules = getattr(model, "model")
    return modules[detect_index]


def _set_detect_head(model: nn.Module, head: nn.Module, detect_index: int = -1) -> None:
    """替换 YOLO 模型中的 Detect head，并保留 Ultralytics 层元信息。"""
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
    """保存带 DuET 增量检测头的 checkpoint。

    template checkpoint 提供完整模型结构；merged_shared_state 写入融合后的共享层；
    old/new checkpoint 提供两套任务 head，最终组合成 DuETIncrementalDetect。
    """
    template_ckpt = torch_load_checkpoint(template_checkpoint_path, map_location=map_location)
    old_ckpt = torch_load_checkpoint(old_checkpoint_path, map_location=map_location)
    new_ckpt = torch_load_checkpoint(new_checkpoint_path, map_location=map_location)

    old_models = _checkpoint_models(old_ckpt)
    new_models = _checkpoint_models(new_ckpt)
    if not old_models or not new_models:
        raise TypeError("old/new checkpoint must contain an nn.Module under 'model' or 'ema'.")

    # head 使用 float 版本，避免半精度 checkpoint 在 CPU 保存/后续加载时出现类型问题。
    old_head = copy.deepcopy(_detect_head(old_models[0], detect_index)).float()
    new_head = copy.deepcopy(_detect_head(new_models[0], detect_index)).float()

    updated = False
    for model in _checkpoint_models(template_ckpt):
        # strict=False 是因为 template 的普通 Detect head 会被替换成自定义增量 head，
        # 两者参数名不可能完全一致。
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
