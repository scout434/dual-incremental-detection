"""
DuET Module: 逐层动态任务向量融合模块

实现论文 DuET (ICCV 2025) Section 3.4 中的 DuET Module 核心算法。

DuET Module 的核心思想：
  对于每个参数层 l，根据该层新旧任务向量的相对幅度，动态计算 retention (αl) 和
  adaptation (βl) 系数，然后用加权任务向量更新共享参数。

数学公式（论文 Eq 9-12）：
  pl = (||τ_old|| - ||τ_curr||) / (||τ_old + τ_curr|| + ε)          ... (9)
  δl = γ * tanh(pl)                                                   ... (10)
  αl = αbase + clamp(δl, -γ, γ),  βl = 1 - αl                      ... (11)
  (θ_st)_incre = θ_s0 + αl * τ_old + βl * τ_curr                     ... (12)

其中：
  - τ_old = θ_st-1 - θ_s0 : 累积旧任务向量（从基准模型到上一合并模型）
  - τ_curr = θ_st - θ_s0  : 当前任务向量（从基准模型到当前训练模型）
  - αl ∈ [0, 1]: 保留系数，值越大越保留旧知识
  - βl ∈ [0, 1]: 适应系数，值越大越适应新知识
  - αl + βl = 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from duet_repro.core.task_vectors import StateDict, task_vector


# 论文中的默认超参数
DEFAULT_GAMMA = 0.5      # δl 的缩放因子，控制 αl 的动态范围
DEFAULT_ALPHA_BASE = 0.5  # αl 的基础值，配合 γ 可以让 αl 在 [0, 1] 范围内动态调整
DEFAULT_EPS = 1e-8        # 数值稳定性常数


def compute_p_factor(
    tau_old_flat: Tensor,
    tau_curr_flat: Tensor,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """
    计算论文 Eq 9 中的 p-factor（层重要性因子）。

    p-factor 是一个无量纲数，衡量"旧任务向量主导"还是"新任务向量主导"：
      - p > 0：旧任务向量的 ℓ1 范数更大，当前层更应该保留旧知识
      - p < 0：新任务向量的 ℓ1 范数更大，当前层更应该学习新知识
      - p ≈ 0：两者相当，需要平衡

    公式：p = (||τ_old||_1 - ||τ_curr||_1) / (||τ_old + τ_curr||_1 + ε)

    Args:
        tau_old_flat: 旧任务向量（已展平为一维张量）。
        tau_curr_flat: 新任务向量（已展平为一维张量）。
        eps: 防止除零的小常数。

    Returns:
        p-factor 标量张量。
    """
    norm_old = tau_old_flat.abs().sum()
    norm_curr = tau_curr_flat.abs().sum()
    norm_sum = (tau_old_flat + tau_curr_flat).abs().sum()
    return (norm_old - norm_curr) / (norm_sum + eps)


def compute_layer_coefficients(
    tau_old_flat: Tensor,
    tau_curr_flat: Tensor,
    gamma: float = DEFAULT_GAMMA,
    alpha_base: float = DEFAULT_ALPHA_BASE,
    eps: float = DEFAULT_EPS,
) -> tuple[float, float]:
    """
    计算论文 Eq 10-11 中的层系数 αl 和 βl。

    通过 p-factor 的双曲正切变换得到 δl，然后计算：
      αl = αbase + clamp(δl, -γ, γ)
      βl = 1 - αl

    Args:
        tau_old_flat: 旧任务向量（已展平）。
        tau_curr_flat: 新任务向量（已展平）。
        gamma: δl 的最大绝对值，控制 αl 的调整范围。
        alpha_base: αl 的基础值，建议设为 0.5。
        eps: 数值稳定性常数。

    Returns:
        (alpha, beta) 元组：保留系数和适应系数。
    """
    p = compute_p_factor(tau_old_flat, tau_curr_flat, eps)
    delta = gamma * torch.tanh(p)
    alpha = alpha_base + torch.clamp(delta, -gamma, gamma)
    alpha = float(torch.clamp(alpha, 0.0, 1.0).item())
    beta = 1.0 - alpha
    return alpha, beta


def _key_is_shared(key: str, exclude_patterns: Iterable[str]) -> bool:
    """
    判断参数键是否属于共享参数（可用于任务向量合并）。

    通过排除列表判断。YOLO 中的检测头层（detect、cv2、cv3、dfl）
    是任务特定的，需要排除。

    Args:
        key: 参数名称。
        exclude_patterns: 需要排除的模式列表。

    Returns:
        True 表示是共享参数；False 表示是任务特定参数。
    """
    lowered = key.lower()
    return not any(p.lower() in lowered for p in exclude_patterns)


def _is_head_cls_output_layer(key: str) -> bool:
    """
    判断参数键是否是检测头的分类输出层（cv3 的最终 Conv2d）。

    在 YOLO 中，cv3 是分类分支，其最后一层 Conv2d 的输出通道数等于 num_classes。
    这类层的特点是：键名中包含 "cv3"，且是最后一层（sub_idx = -1 或名称以 ".2" 结尾）。

    Args:
        key: 参数名称，格式如 "model.22.cv3.0.2.weight"。

    Returns:
        True 表示是分类输出层。
    """
    if "cv3" not in key.lower():
        return False
    # 匹配 cv3 的最后一层：cv3[N][2] 或 cv3.N.2
    lower = key.lower()
    # 检查是否是 cv3 的最终输出层（-1 位置，即索引 2）
    # 常见格式：model[-1].cv3[0][2].weight, model.22.cv3.0.2.weight
    import re
    return bool(re.search(r'cv3\[\d+\]\[2\]', lower) or re.search(r'cv3\.\d+\.2', lower))


def concat_head_params(
    new_head_sd: StateDict,
    old_head_sd: StateDict,
    old_nc: int,
    new_total_nc: int,
) -> StateDict:
    """
    拼接新旧检测头的分类输出层参数（论文 Eq 13）。

    这是 Incremental Head 的核心操作：将当前任务的检测头参数与历史任务的
    检测头参数拼接，使得最终模型能够同时识别旧类别和新类别。

    拼接逻辑：
      对于 cv3 的最终分类输出层（weight/bias）：
        新权重 = torch.cat([旧权重, 新权重中新增类别部分], dim=0)
        新偏置 = torch.cat([旧偏置, 新偏置中新增类别部分], dim=0)

      对于 cv2（box 回归层）和 dfl：
        直接使用新模型的参数（这些层与类别数无关）

    重要：此函数不修改共享参数（backbone、neck），只处理检测头参数。

    Args:
        new_head_sd: 当前任务训练后模型的检测头状态字典。
                     已被 expand_detect_head 扩展过，nc = old_nc + new_task_nc。
        old_head_sd: 上一轮合并后模型的检测头状态字典。
                     nc = old_nc。
        old_nc: 历史任务的总类别数。
        new_total_nc: 扩展后的总类别数（old_nc + 新任务类别数）。

    Returns:
        拼接后的检测头状态字典。

    Example:
        # Task 1 后 old_nc=5, Task 2 扩展到 new_total_nc=10
        # new_head_sd 包含扩展后的 cv3.weight shape=(10, c3, 1, 1)
        # old_head_sd 包含 Task1 的 cv3.weight shape=(5, c3, 1, 1)
        # 拼接后：cv3.weight shape=(10, c3, 1, 1)
        #   前 5 通道 = old_head_sd 的权重（Task1 学到的知识）
        #   后 5 通道 = new_head_sd 的权重中新增类别部分（Task2 学到的知识）
    """
    result = {}
    new_nc = new_total_nc - old_nc  # 当前任务新增的类别数

    for key, new_value in new_head_sd.items():
        if not _is_head_cls_output_layer(key):
            # 非分类输出层（如 cv2 的 box 回归层，dfl 层等）
            # 直接使用新模型的参数
            result[key] = new_value.clone()
            continue

        # 这是 cv3 的分类输出层，需要拼接
        old_value = old_head_sd.get(key)
        if old_value is None:
            # 旧模型中没有该参数（不应该发生），直接使用新模型参数
            result[key] = new_value.clone()
            continue

        # 验证旧模型的类别数
        old_cls_nc = old_value.shape[0]
        if old_cls_nc != old_nc:
            # 类别数不匹配，跳过拼接
            result[key] = new_value.clone()
            continue

        if new_nc <= 0:
            # 没有新增类别
            result[key] = new_value.clone()
            continue

        if new_value.dim() == 4:
            # weight: (out_channels, in_channels, H, W) → 在 dim=0 拼接
            old_w = old_value  # (old_nc, in_ch, 1, 1)
            new_w = new_value[new_nc:]  # (new_nc, in_ch, 1, 1)
            result[key] = torch.cat([old_w, new_w], dim=0)
        elif new_value.dim() == 1:
            # bias: (out_channels,) → 在 dim=0 拼接
            old_b = old_value
            new_b = new_value[:new_nc]
            result[key] = torch.cat([old_b, new_b], dim=0)
        else:
            # 其他形式，直接使用新模型参数
            result[key] = new_value.clone()

    return result


def merge_state_dicts_with_duet_module(
    reference: StateDict,
    old_model: StateDict,
    new_model: StateDict,
    *,
    gamma: float = DEFAULT_GAMMA,
    alpha_base: float = DEFAULT_ALPHA_BASE,
    shared_key_exclude: Iterable[str] = (),
    per_layer_report: bool = False,
    # ===== Incremental Head 拼接参数=====
    old_head_sd: StateDict | None = None,
    new_head_sd: StateDict | None = None,
    old_nc: int = 0,
    new_total_nc: int = 0,
) -> tuple[StateDict, dict]:
    """
    使用 DuET Module 的逐层动态融合算法合并状态字典，并可选地拼接检测头参数。

    本函数整合了论文 Sec 3.4 的两个核心操作：
      1. DuET Module：对共享参数（backbone + neck）进行逐层动态融合（Eq 12）
      2. Incremental Head：对检测头参数进行拼接（Eq 13）

    共享参数融合公式（Eq 12）：
      θ_merged[l] = θ_ref[l] + αl * τ_old[l] + βl * τ_curr[l]

    检测头拼接公式（Eq 13）：
      (θτt)_incre ← [θτt; (θτt-1)_incre]

    何时传入检测头参数：
      - 对于域增量学习（所有任务类别数相同）：不需要传入检测头参数
      - 对于类别增量学习（类别数递增）：需要传入 old_head_sd、new_head_sd、old_nc、new_total_nc

    Args:
        reference: 基准预训练模型的状态字典。
        old_model: 上一任务合并后模型的状态字典。
        new_model: 当前任务训练后模型的状态字典。
        gamma: 动态系数 γ，控制 αl 的调整范围（建议值 0.5）。
        alpha_base: 基础系数 αbase，αl 会在 [αbase-γ, αbase+γ] 范围内调整。
        shared_key_exclude: 不参与合并的参数名称模式列表。
        per_layer_report: 是否生成每个层的详细系数报告。
        old_head_sd: 上一轮合并后模型的检测头状态字典（用于拼接）。
        new_head_sd: 当前任务模型的检测头状态字典（已扩展，用于拼接）。
        old_nc: 历史任务的总类别数。
        new_total_nc: 扩展后的总类别数。

    Returns:
        (合并后的状态字典, 报告字典)。
    """
    merged: StateDict = {}
    incompatible: list[str] = []
    merged_keys = 0
    skipped_keys = 0
    per_layer_alphas: dict[str, float] = {}

    # 遍历所有参数（取三个状态字典的并集）
    all_keys = set(reference) | set(old_model) | set(new_model)

    for key in sorted(all_keys):
        ref_value = reference.get(key)
        old_value = old_model.get(key)
        new_value = new_model.get(key)

        if new_value is None:
            skipped_keys += 1
            continue

        # 判断是否是共享参数
        is_shared = (
            ref_value is not None
            and old_value is not None
            and ref_value.shape == old_value.shape == new_value.shape
            and torch.is_floating_point(ref_value)
            and _key_is_shared(key, shared_key_exclude)
        )

        if is_shared:
            # 共享参数：使用 DuET Module 逐层动态融合（Eq 12）
            tau_old_flat = (old_value.float() - ref_value.float()).flatten()
            tau_curr_flat = (new_value.float() - ref_value.float()).flatten()

            alpha_l, beta_l = compute_layer_coefficients(
                tau_old_flat, tau_curr_flat, gamma=gamma, alpha_base=alpha_base
            )

            merged[key] = (
                ref_value.detach().float()
                + alpha_l * (old_value.detach().float() - ref_value.detach().float())
                + beta_l * (new_value.detach().float() - ref_value.detach().float())
            ).to(dtype=new_value.dtype)

            merged_keys += 1
            if per_layer_report:
                per_layer_alphas[key] = alpha_l
        else:
            # 非共享参数：直接使用新模型参数
            merged[key] = new_value.detach().clone()
            # 统计跳过的层
            incompatible.append(key)
            skipped_keys += 1

    # ===== 检测头参数拼接（Eq 13）=====
    # 在共享参数融合完成后，将旧检测头参数拼接到新检测头参数中
    if old_head_sd is not None and new_head_sd is not None and old_nc > 0 and new_total_nc > old_nc:
        concat_sd = concat_head_params(
            new_head_sd,
            old_head_sd,
            old_nc=old_nc,
            new_total_nc=new_total_nc,
        )
        # 用拼接后的检测头参数覆盖合并结果中的检测头参数
        merged.update(concat_sd)

    report = {
        "merged_keys": merged_keys,
        "skipped_keys": skipped_keys,
        "incompatible_keys": incompatible,
        "gamma": gamma,
        "alpha_base": alpha_base,
    }
    if per_layer_report:
        report["per_layer_alphas"] = per_layer_alphas

    return merged, report
