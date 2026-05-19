"""
DuET RAI（Retention-Adaptability Index，保留-适应度指数）指标计算模块

本模块实现了 DuET 论文中用于评估双增量目标检测（DuIOD）模型性能的核心指标。

在增量学习中，模型面临两个相互矛盾的目标：
  1. 知识保留（Knowledge Retention）：不遗忘已学习的旧任务
  2. 知识适应（Knowledge Adaptation）：快速学习新的任务/域

RAI 指标通过量化这两个维度来解决增量学习的评估问题：

  保留指数 (Retention Index, RI)：
    衡量模型在旧任务上的知识保持能力。
    定义为增量训练后模型在旧任务上的 mAP 与单任务基准 mAP 的比值。
    RI 越接近 1.0，说明旧任务的遗忘越少。

  泛化指数 (Generalization Index, GI)：
    衡量模型在新任务上的适应能力。
    定义为增量训练后模型在新任务上的 mAP 与单任务基准 mAP 的比值。
    GI 越接近 1.0，说明新任务的学习效果越好。

  RAI 综合指标：
    RAI = (Avg RI + Avg GI) / 2
    即所有任务的平均 RI 和 GI 的算术平均。
    越高越好，综合反映模型的保留和适应能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class RaiResult:
    """
    RAI 评估结果的不可变数据结构。

    使用 @dataclass(frozen=True) 确保实例创建后不可修改，
    保证评估结果的一致性和线程安全性。
    使用 namedtuple 的替代方案，提供更清晰的字段访问方式。
    """
    avg_ri: float  # 所有任务的平均保留指数
    avg_gi: float  # 所有任务的平均泛化指数
    rai: float      # 综合 RAI 指标 = (avg_ri + avg_gi) / 2


def average(values: list[float]) -> float:
    """
    计算浮点数列表的算术平均值。

    这是一个简洁的辅助函数，避免重复调用 statistics.mean()。

    Args:
        values: 浮点数列表，可以为空列表。

    Returns:
        列表的算术平均值；若列表为空则返回 0.0。
    """
    return float(mean(values)) if values else 0.0


def retention_index(current_map: float, reference_map: float, eps: float = 1e-12) -> float:
    """
    计算单个任务的保留指数（Retention Index, RI）。

    保留指数衡量增量训练后模型在旧任务上的性能相对于单任务基准的保持程度。
    这是 DuET 评估框架中"知识保留"维度的量化指标。

    数学定义：
      RI = min(current_mAP / reference_mAP, 1.0)

    其中：
      - current_mAP：增量训练后模型在某个旧任务上的 mAP@0.5（或 COCO 标准的 mAP）
      - reference_mAP：该任务单独训练（单任务基线）时模型在同任务上的 mAP
      - 为什么要 min(..., 1.0)：防止参考 mAP 异常低时 RI 超过 1.0，确保 RI ∈ [0, 1]

    使用说明：
      - 通常需要为每个已学习的任务计算一次 RI
      - 旧任务的 reference_map 通常来自单任务基线实验（每个任务独立训练）
      - 所有任务的 RI 取平均得到 Avg RI

    Args:
        current_map: 增量训练后模型在该任务上的 mAP 值。
        reference_map: 该任务单任务训练时的基准 mAP 值。
        eps: 防止除零的小常数，当 reference_map <= eps 时返回 0.0。

    Returns:
        保留指数 RI，范围 [0.0, 1.0]。
        1.0 表示完全保留（无遗忘），0.0 表示完全遗忘。
    """
    if reference_map <= eps:
        # 参考 mAP 无效（极小或为零），无法计算有意义的比率，返回 0
        return 0.0
    return min(current_map / reference_map, 1.0)


def generalization_index(current_map: float, reference_map: float, eps: float = 1e-12) -> float:
    """
    计算单个任务的泛化指数（Generalization Index, GI）。

    泛化指数衡量增量训练后模型在新任务上的性能相对于单任务基准的达标程度。
    这是 DuET 评估框架中"知识适应"维度的量化指标。

    数学定义：
      GI = min(current_mAP / reference_mAP, 1.0)

    其中：
      - current_mAP：增量训练后模型在新任务上的 mAP 值
      - reference_mAP：该新任务单独训练时的基准 mAP 值
      - 同样的 min(..., 1.0) 防止异常值超出 [0, 1] 范围

    使用说明：
      - GI 的计算发生在每个新任务学习完成后
      - reference_map 通常来自该任务的单任务基线实验
      - 所有任务的 GI 取平均得到 Avg GI

    与 RI 的区别：
      - RI 衡量对"旧"知识的保持（回顾性）
      - GI 衡量对"新"知识的获取（前瞻性）
      - 两者共同反映增量学习方法的"保留-适应"权衡

    Args:
        current_map: 增量训练后模型在新任务上的 mAP 值。
        reference_map: 该任务单任务训练时的基准 mAP 值。
        eps: 防止除零的小常数。

    Returns:
        泛化指数 GI，范围 [0.0, 1.0]。
        1.0 表示新任务学习效果与单任务基线相当，0.0 表示完全无法学习新任务。
    """
    if reference_map <= eps:
        return 0.0
    return min(current_map / reference_map, 1.0)


def rai_from_indices(retention: list[float], generalization: list[float]) -> RaiResult:
    """
    根据保留指数和泛化指数列表，计算 RAI 综合评估结果。

    这是 RAI 评估流程的最终步骤，将多个任务的 RI 和 GI 汇总为单个综合指标。

    计算步骤：
      1. 计算所有任务保留指数的平均值：Avg RI = mean(retention)
      2. 计算所有任务泛化指数的平均值：Avg GI = mean(generalization)
      3. 计算综合指标：RAI = (Avg RI + Avg GI) / 2

    RAI 的解读：
      - RAI 越接近 1.0：模型在保留旧知识和学习新知识两方面都表现优秀
      - RAI 接近 0.5：模型在保留和适应之间做了妥协
      - RAI 接近 0.0：模型在保留和适应两方面都表现很差

    不同场景下 RI 和 GI 的权重考量：
      - 在类别增量场景中，RI（保留旧类别）和 GI（学习新类别）同等重要
      - 在域增量场景中，GI 可能更重要（需要快速适应新域）
      - 可以根据具体应用场景调整 RAI 的计算方式（如加权平均）

    Args:
        retention: 每个任务的保留指数（RI）列表。
        generalization: 每个任务的泛化指数（GI）列表。

    Returns:
        包含 avg_ri、avg_gi 和 rai 三个字段的 RaiResult 对象。
    """
    avg_ri = average(retention)      # 计算平均保留指数
    avg_gi = average(generalization)  # 计算平均泛化指数
    rai = (avg_ri + avg_gi) / 2.0    # RAI = 算术平均
    return RaiResult(avg_ri=avg_ri, avg_gi=avg_gi, rai=rai)
