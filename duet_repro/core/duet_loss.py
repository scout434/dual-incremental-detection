"""
DuET 检测损失模块：集成蒸馏损失和方向一致性损失

实现论文 DuET (ICCV 2025) Section 3.5 中的完整训练损失函数。

训练损失的组成（论文 Eq 15）：
  LTotal =
      LDetector,                                    当 t = 1（基础任务）
      LDetector + λDistill * L*_Distill + λDC * LDC,  当 t ≥ 2（增量任务）

其中：
  1. LDetector: 标准检测损失（box_loss + cls_loss + dfl_loss），由 Ultralytics 提供
  2. L*_Distill: 改进的蒸馏损失，对分类和边界框分别使用动态过滤
  3. LDC: 方向一致性损失（论文 Sec 3.5），惩罚新旧任务向量的方向冲突

Direction Consistency Loss (LDC) 公式（论文 Eq 16）：
  LDC = Σ_{i∈θs} ReLU( - [(τ_st - τ_st-1) · (τ_st-1 - τ_st-2)]_i )

物理意义：
  - τ_st - τ_st-1: 当前任务对共享参数 θs 的更新方向
  - τ_st-1 - τ_st-2: 上一任务对共享参数 θs 的更新方向
  - 点积 < 0 表示方向冲突，ReLU 惩罚冲突的方向
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class DuETDetectionLoss(v8DetectionLoss):
    """
    DuET 检测损失函数：集成标准检测损失 + 蒸馏损失 + 方向一致性损失。

    该类继承自 Ultralytics 的 v8DetectionLoss，通过扩展 loss() 方法，
    在标准检测损失的基础上增加蒸馏损失和方向一致性损失。

    使用方式：
      model = YOLO("yolo11n.pt")
      model.criterion = DuETDetectionLoss(
          model,
          teacher_model=old_model,
          distill_weight=0.5,
          dc_weight=0.1,
      )
      # 后续训练会自动使用 DuETDetectionLoss

    Attributes:
        teacher_model: 教师模型（旧任务模型），用于蒸馏。
        distill_weight: 蒸馏损失的权重 λDistill。
        dc_weight: 方向一致性损失的权重 λDC。
        prev_task_vector: 上一任务的任务向量，用于计算 LDC。
        distill_temperature: 蒸馏温度。
        distill_alpha: 分类过滤的分位数阈值。
        bbox_alpha: 边界框过滤的分位数阈值。
    """

    def __init__(
            self,
            model,
            teacher_model: nn.Module | None = None,
            distill_weight: float = 0.01,
            dc_weight: float = 0.01,
            distill_temperature: float = 2.0,
            distill_cls_quantile: float = 0.75,
            distill_bbox_quantile: float = 0.75,
            curr_task_vector: dict | None = None,
            reference_state: dict | None = None,
            shared_key_exclude: Iterable[str] = (),
            old_class_indices: list[int] | tuple[int, ...] | None = None,
    ):
        """
        初始化 DuET 检测损失函数。

        Args:
            model: 当前训练的 YOLO 检测模型。
            teacher_model: 教师模型（旧任务模型），用于蒸馏知识。
            distill_weight: 蒸馏损失 L*_Distill 的权重 λDistill。
            dc_weight: 方向一致性损失 LDC 的权重 λDC。
            distill_temperature: 蒸馏温度 T。
            distill_cls_quantile: 分类蒸馏时置信度阈值的分位数。
            distill_bbox_quantile: 边界框蒸馏时方差阈值的分位数。
            curr_task_vector: 上一任务相对于基准预训练模型的任务向量，
                            用于计算方向一致性损失。
        """
        super().__init__(model)
        self.model = model
        self.teacher = teacher_model
        self.distill_weight = distill_weight
        self.dc_weight = dc_weight
        self.distill_temperature = distill_temperature
        self.distill_cls_quantile = distill_cls_quantile
        self.distill_bbox_quantile = distill_bbox_quantile
        self.curr_task_vector = curr_task_vector
        self.reference_state = reference_state
        self.shared_key_exclude = tuple(str(pattern).lower() for pattern in shared_key_exclude)
        self.old_class_indices = None if old_class_indices is None else tuple(int(i) for i in old_class_indices)
        self.task_vector_history: list[dict] = []
        self._teacher_dtype_synced = False
        if self.teacher is not None:
            self.teacher.eval()

    def set_teacher(self, teacher_model: nn.Module) -> None:
        """
        设置教师模型（蒸馏用）。

        Args:
            teacher_model: 教师模型，应该是在旧任务上训练/合并后的模型。
        """
        self.teacher = teacher_model

    def set_curr_task_vector(self, curr_task_vector: dict) -> None:
        """
        设置上一任务的任务向量（用于计算 LDC）。

        Args:
            curr_task_vector: 上一任务相对于基准预训练模型的任务向量。
        """
        self.curr_task_vector = curr_task_vector

    def set_reference_state(self, reference_state: dict, shared_key_exclude: Iterable[str] = ()) -> None:
        """Set the fixed reference model used to build differentiable task vectors."""
        self.reference_state = reference_state
        self.shared_key_exclude = tuple(str(pattern).lower() for pattern in shared_key_exclude)

    def _key_is_shared(self, key: str) -> bool:
        lowered = key.lower()
        return not any(pattern in lowered for pattern in self.shared_key_exclude)

    def _current_task_vector_with_grad(self) -> dict[str, torch.Tensor] | None:
        """Build theta_current - theta_reference without detaching current parameters."""
        if self.reference_state is None:
            return None

        delta: dict[str, torch.Tensor] = {}
        for key, param in self.model.named_parameters():
            ref_value = self.reference_state.get(key)
            if ref_value is None:
                continue
            if ref_value.shape != param.shape:
                continue
            if not torch.is_floating_point(param):
                continue
            if not self._key_is_shared(key):
                continue
            ref_value = ref_value.to(param.device, dtype=param.dtype)
            delta[key] = param.float() - ref_value.float()
        return delta

    @staticmethod
    def _scores_to_anchor_last(scores: torch.Tensor, nc: int) -> torch.Tensor:
        """Return class logits as [batch, anchors, classes]."""
        if scores.ndim == 3 and (scores.shape[1] == nc or scores.shape[1] < scores.shape[2]):
            return scores.permute(0, 2, 1).contiguous()
        return scores.contiguous()

    @staticmethod
    def _boxes_to_anchor_last(boxes: torch.Tensor) -> torch.Tensor:
        """Return DFL box logits as [batch, anchors, 4 * reg_max]."""
        if boxes.ndim == 3 and boxes.shape[1] % 4 == 0 and boxes.shape[1] < boxes.shape[2]:
            return boxes.permute(0, 2, 1).contiguous()
        return boxes.contiguous()

    def _ensure_teacher_dtype(self, preds: dict[str, torch.Tensor]) -> None:
        """
        将教师模型参数 dtype 同步到与学生模型一致（处理 FP16 训练）。

        训练时 Ultralytics 使用 FP16（half precision）前向传播，
        但教师模型从检查点加载后默认是 FP32。首次调用时检测 dtype
        不一致则递归将教师模型转为 FP16，避免 conv 层报错：
        "Input type (torch.cuda.HalfTensor) and weight type (torch.FloatTensor) should be the same"

        Args:
            preds: 学生模型的当前批次预测，从中获取 dtype 信息。
        """
        if self.teacher is None or self._teacher_dtype_synced:
            return
        student_dtype = preds["boxes"].dtype
        student_device = preds["boxes"].device
        first_param = next(self.teacher.parameters(), None)
        if first_param is None:
            self._teacher_dtype_synced = True
            return
        if first_param.dtype == student_dtype and first_param.device == student_device:
            self._teacher_dtype_synced = True
            return
        self.teacher.to(device=student_device, dtype=student_dtype)
        self.teacher.eval()
        self._teacher_dtype_synced = True

    def loss(
            self,
            preds: dict[str, torch.Tensor],
            batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算 DuET 检测损失：检测损失 + 蒸馏损失 + 方向一致性损失。

        Args:
            preds: 模型预测，包含 boxes、scores 和 feats。
            batch: 批次数据，包含图像和标注。

        Returns:
            (总损失, 损失的 detached 版本)，用于日志记录。
        """

        # 获取 preds 中任意张量的设备
        device = preds["boxes"].device

        if self.device != device:
            self.device = device

        # 将父类 v8DetectionLoss 中可能位于 CPU 的张量移动到 device
        if hasattr(self, 'proj') and self.proj.device != device:
            self.proj = self.proj.to(device)

        # Step 1.5: 同步教师模型 dtype 到学生模型（处理 FP16 训练，仅执行一次）
        self._ensure_teacher_dtype(preds)

        # Step 1: 计算标准检测损失（box + cls + dfl）
        batch_size = preds["boxes"].shape[0]
        detection_loss, loss_detach = super().loss(preds, batch)
        # v8DetectionLoss.loss() already returns a batch-scaled loss. Keep the
        # auxiliary terms on the same scale and do not multiply the detector
        # loss by batch_size a second time.
        total_loss = detection_loss.sum()
        loss_items = loss_detach.clone()

        # Step 2: 计算蒸馏损失（附录 B Eq 11-16）
        distill_loss = torch.tensor(0.0, device=detection_loss.device)
        if self.teacher is not None and self.distill_weight > 0:
            distill_loss = self._compute_distillation_loss(preds, batch)
            total_loss = total_loss + self.distill_weight * distill_loss * batch_size
            loss_items = torch.cat([loss_items, distill_loss.detach().unsqueeze(0)])

        # Step 3: 计算方向一致性损失（如果有历史任务向量）
        dc_loss = torch.tensor(0.0, device=detection_loss.device)
        if (self.curr_task_vector is not None or self.reference_state is not None) and len(self.task_vector_history) >= 1:
            dc_loss = self._compute_directional_consistency_loss().to(self.device)
            total_loss = total_loss + self.dc_weight * dc_loss * batch_size
            loss_items = torch.cat([loss_items, dc_loss.detach().unsqueeze(0)])

        return total_loss, loss_items

    def _compute_distillation_loss(
            self,
            preds: dict[str, torch.Tensor],
            batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算改进的蒸馏损失（L*_Distill，论文附录 B Eq 11-16）。

        论文公式：
          L*_Distill_cls  = (1/|M*_cls|)  Σ_{i∈M*_cls}  ||c_curr[i] − c_old[i]||²     ... (12)
          M*_cls          = { i | max(c_old[i]) ≥ τ_cls }                              ... (13)
            其中 τ_cls = max(c_old) 的 75th 分位数

          L*_Distill_bbox = (1/|M*_bbox|) Σ_{j∈M*_bbox} DKL(Softmax(b_curr[j]) ‖ Softmax(b_old[j]))  ... (14)
          M*_bbox         = { j | Var(b_old[j]) ≤ τ_bbox }                              ... (15)
            其中 τ_bbox = Var(b_old) 的 75th 分位数

          L*_Distill = L*_Distill_cls + L*_Distill_bbox                                 ... (16)

        与标准蒸馏的区别：仅对教师模型中高置信度分类样本（分类分支）
        和低方差边界框预测（回归分支）施加蒸馏损失。

        Args:
            preds: 学生模型在当前批次的原始检测头输出（tower 原始预测），
                   包含 boxes（边界框原始输出）和 scores（分类原始 logits）。
            batch: 批次数据。

        Returns:
            蒸馏损失标量。
        """
        device = preds["boxes"].device
        img = batch["img"].to(device)

        # 提取原始分类 logits 和 bbox 预测
        # YOLO 检测头输出: [..., 0:4=bbox, 4=obj, 5:5+nc=cls]
        # boxes: (batch, total_anchors, 4*reg_max)  bbox 原始输出
        # scores: (batch, total_anchors, nc)         cls 原始 logits
        student_bbox_raw = preds["boxes"]      # (batch, total_anchors, reg_max*4) 或类似
        student_cls_raw  = preds["scores"]     # (batch, total_anchors, nc)

        # Teacher 前向传播（无梯度，eval 模式）
        # teacher.model(img) 返回 (tower_features, predictions_dict)，与 Ultralytics 内部格式一致
        self.teacher.eval()
        student_bbox_raw = self._boxes_to_anchor_last(student_bbox_raw)
        student_cls_raw = self._scores_to_anchor_last(preds["scores"], int(getattr(self, "nc", preds["scores"].shape[1])))

        self.teacher.eval()
        with torch.no_grad():
            raw_output = self.teacher(img)
            # 统一处理：若返回元组则取第二个元素（预测字典），否则直接使用
            teacher_output = raw_output[1] if isinstance(raw_output, tuple) else raw_output

        teacher_bbox_raw = teacher_output["boxes"]
        teacher_cls_raw  = teacher_output["scores"]

        # 处理 teacher 输出可能的多层列表格式（取最后一层或拼接所有层）
        if isinstance(teacher_bbox_raw, (list, tuple)):
            # teacher 输出是逐层的列表，拼接所有层
            teacher_bbox_raw = torch.cat([t.detach() for t in teacher_bbox_raw], dim=1)
            teacher_cls_raw  = torch.cat([t.detach() for t in teacher_cls_raw],  dim=1)

        teacher_bbox_raw = self._boxes_to_anchor_last(teacher_bbox_raw)
        teacher_cls_raw = self._scores_to_anchor_last(teacher_cls_raw, int(getattr(self, "nc", teacher_cls_raw.shape[1])))

        if self.old_class_indices is not None:
            class_index = torch.tensor(self.old_class_indices, device=device, dtype=torch.long)
            max_nc = min(student_cls_raw.shape[2], teacher_cls_raw.shape[2])
            class_index = class_index[class_index < max_nc]
            if class_index.numel() == 0:
                return torch.tensor(0.0, device=device)
            student_cls_raw = student_cls_raw.index_select(2, class_index)
            teacher_cls_raw = teacher_cls_raw.index_select(2, class_index)

        # 学生模型分类shape只取前old_nc个
        if student_cls_raw.shape != teacher_cls_raw.shape:
            if student_cls_raw.shape[:2] != teacher_cls_raw.shape[:2]:
                return torch.tensor(0.0, device=device)
            old_nc = teacher_cls_raw.shape[2]
            # Student 的类别数
            new_nc = student_cls_raw.shape[2]
            if new_nc < old_nc:
                return torch.tensor(0.0, device=device)
            student_cls_raw = student_cls_raw[:, :, :old_nc]

        # 检查bbox形状是否一致
        if student_bbox_raw.shape != teacher_bbox_raw.shape:
            return torch.tensor(0.0, device=device)

        batch_size = student_cls_raw.shape[0]
        total_anchors = student_cls_raw.shape[1]

        # 将 teacher bbox 展平以计算方差（bbox 格式为 reg_max*4，沿最后一维）
        # bbox shape: (batch,4*reg_max,total_anchors) -> (batch, total_anchors, 4, reg_max)
        reg_max = student_bbox_raw.shape[2] // 4
        teacher_bbox_4d = teacher_bbox_raw.view(batch_size, total_anchors, 4, reg_max)
        student_bbox_4d = student_bbox_raw.view(batch_size, total_anchors, 4, reg_max)

        # Flatten spatial dims: (batch, anchors, 4, reg_max) -> (batch*anchors*4, reg_max)
        teacher_bbox_flat = teacher_bbox_4d.reshape(batch_size * total_anchors * 4, reg_max)
        student_bbox_flat = student_bbox_4d.reshape(batch_size * total_anchors * 4, reg_max)

        # ===== 分类蒸馏 (Eq 12-13) =====
        # 使用原始分类 logits（未经过 sigmoid/softmax）
        teacher_cls_flat = teacher_cls_raw.reshape(batch_size * total_anchors, -1).float()
        student_cls_flat = student_cls_raw.reshape(batch_size * total_anchors, -1).float()

        # 动态掩码：仅保留教师模型中高置信度的样本，论文使用原始 logits 的 max(c_old)。
        teacher_conf = teacher_cls_flat.amax(dim=1)   # (batch*anchors,)
        cls_threshold = torch.quantile(teacher_conf, self.distill_cls_quantile)
        cls_mask = teacher_conf >= cls_threshold                          # (batch*anchors,)

        cls_loss = torch.tensor(0.0, device=device)
        if cls_mask.any():
            cls_diff = student_cls_flat[cls_mask] - teacher_cls_flat[cls_mask].detach()
            cls_loss = cls_diff.pow(2).sum(dim=1).mean()

        # ===== 边界框蒸馏 (Eq 14-15) =====
        # 动态掩码：仅保留教师模型中低方差的 bbox 预测
        teacher_bbox_var = teacher_bbox_flat.var(dim=1)                 # (batch*anchors*4,)
        bbox_threshold  = torch.quantile(teacher_bbox_var.float(), self.distill_bbox_quantile)
        bbox_mask = teacher_bbox_var <= bbox_threshold                  # (batch*anchors*4,)

        bbox_loss = torch.tensor(0.0, device=device)
        if bbox_mask.any():
            # KL 散度: DKL(Softmax(b_curr) ‖ Softmax(b_old))
            b_curr_log = F.log_softmax(student_bbox_flat[bbox_mask], dim=1)
            b_curr_prob = b_curr_log.exp()
            b_old_log = F.log_softmax(teacher_bbox_flat[bbox_mask].detach(), dim=1)
            bbox_loss = (b_curr_prob * (b_curr_log - b_old_log)).sum(dim=1).mean()

        total_distill = cls_loss + bbox_loss

        # 按论文公式，除以 |M*_cls| 或 |M*_bbox| 做平均
        # batchmean 已在 KL 散度中隐式处理，MSE 用 reduction="mean" 也已处理
        return total_distill

    def _compute_directional_consistency_loss(self) -> torch.Tensor:
        """
        计算方向一致性损失（LDC，论文 Eq 16）。

        公式：LDC = Σ ReLU( - [(τ_st - τ_st-1) · (τ_st-1 - τ_st-2)] )

        物理意义：
          - τ_st - τ_st-1: 当前任务对共享参数的更新增量
          - τ_st-1 - τ_st-2: 上一任务对共享参数的更新增量
          - 点积 < 0 表示方向冲突，ReLU 将惩罚施加于冲突的方向

        Returns:
            方向一致性损失标量。
        """
        current_delta = self._current_task_vector_with_grad()
        if current_delta is None:
            current_delta = self.curr_task_vector

        if current_delta is None or len(self.task_vector_history) < 1:
            return torch.tensor(0.0)

        # 获取当前任务和上一任务的任务向量
        # prev_task_vector = τ_{t}（从基准到上一任务）
        # task_vector_history[-1] = τ_{t-1}
        curr_delta = current_delta  # τ_{t}
        prev_delta = self.task_vector_history[-1]   # τ_{t-1}

        losses: list[torch.Tensor] = []

        for key in curr_delta:
            if key not in prev_delta:
                continue

            delta_curr = curr_delta[key]  # τ_{t}
            delta_prev = prev_delta[key].to(delta_curr.device, dtype=delta_curr.dtype)  # τ_{t-1}

            if delta_curr.shape != delta_prev.shape:
                continue

            # 计算更新增量：τ_{t} - τ_{t-1}
            update_curr = delta_curr - delta_prev

            # 计算上一任务的更新方向：τ_{t-1} - τ_{t-2}
            if len(self.task_vector_history) >= 2:
                prev_prev_delta = self.task_vector_history[-2]
                if key in prev_prev_delta:
                    delta_prev_prev = prev_prev_delta[key].to(delta_curr.device, dtype=delta_curr.dtype)
                    update_prev = delta_prev - delta_prev_prev
                else:
                    update_prev = delta_prev
            else:
                # 如果没有更早的历史，使用 delta_prev(τ_{t-1}) 作为上一更新方向
                update_prev = delta_prev

            if update_curr.shape != update_prev.shape:
                continue

            # Eq. 16: penalize negative dot products between consecutive
            # shared-parameter updates. Cosine similarity is a different loss.
            update_curr_flat = update_curr.flatten()
            update_prev_flat = update_prev.flatten()

            dot_product = torch.dot(update_curr_flat, update_prev_flat)
            losses.append(F.relu(-dot_product))

        if not losses:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(losses).sum()

    def record_task_vector(self, task_vector: dict) -> None:
        """
        记录当前任务的任务向量到历史中（用于后续任务计算 LDC）。

        Args:
            task_vector: 当前任务相对于基准预训练模型的任务向量。
        """
        self.task_vector_history.append(task_vector)


def create_duet_criterion(
        model,
        teacher_model: nn.Module | None = None,
        distill_weight: float = 0.5,
        dc_weight: float = 0.1,
        distill_temperature: float = 2.0,
        distill_cls_quantile: float = 0.75,
        distill_bbox_quantile: float = 0.75,
        reference_state: dict | None = None,
        shared_key_exclude: Iterable[str] = (),
        old_class_indices: list[int] | tuple[int, ...] | None = None,
) -> DuETDetectionLoss:
    """
    创建 DuET 检测损失函数的工厂函数。

    这是将 DuET 损失集成到 YOLO 训练流程的推荐方式。

    使用方式：
      # 在任务 t 训练前（t ≥ 2）
      teacher = YOLO("task_{t-1}_merged.pt").model
      criterion = create_duet_criterion(
          model,
          teacher_model=teacher,
          distill_weight=0.5,
          dc_weight=0.1,
      )
      model.criterion = criterion

    Args:
        model: 当前 YOLO 检测模型。
        teacher_model: 教师模型（旧任务合并模型）。
        distill_weight: 蒸馏损失权重。
        dc_weight: 方向一致性损失权重。
        distill_temperature: 蒸馏温度。
        distill_cls_quantile: 分类蒸馏分位数。
        distill_bbox_quantile: 边界框蒸馏分位数。

    Returns:
        配置好的 DuETDetectionLoss 实例。
    """
    return DuETDetectionLoss(
        model,
        teacher_model=teacher_model,
        distill_weight=distill_weight,
        dc_weight=dc_weight,
        distill_temperature=distill_temperature,
        distill_cls_quantile=distill_cls_quantile,
        distill_bbox_quantile=distill_bbox_quantile,
        reference_state=reference_state,
        shared_key_exclude=shared_key_exclude,
        old_class_indices=old_class_indices,
    )
