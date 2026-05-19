"""
RAI (Retention-Adaptability Index, 保留-适应度指数) 评估脚本

该脚本用于计算 DuET 论文中提出的 RAI 指标，以量化评估双增量目标检测模型的性能。

RAI 指标包含三个维度：
  1. 保留指数 (Retention Index, RI)：衡量模型在已学习任务上的知识保持能力
     RI = min(current_mAP / reference_mAP, 1.0)
     - current_mAP：增量训练后模型在旧任务上的 mAP
     - reference_mAP：旧任务单独训练时模型在旧任务上的 mAP（基准）
     - RI 越接近 1.0，说明知识保持越好

  2. 泛化指数 (Generalization Index, GI)：衡量模型在新任务上的适应能力
     GI = min(current_mAP / reference_mAP, 1.0)
     - current_mAP：增量训练后模型在新任务上的 mAP
     - reference_mAP：新任务单独训练时模型在新任务上的 mAP（基准）
     - GI 越接近 1.0，说明新任务适应越好

  3. RAI 综合指标：
     RAI = (Avg RI + Avg GI) / 2
     - 所有任务的平均 RI 和 GI 的算术平均值
     - 越高越好，综合反映模型的保留和适应能力

使用方法：
    python eval_rai.py --metrics outputs/pascal_series_duet_yolo11n/training_history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 导入 DuET 核心指标计算函数
from duet_repro.core.metrics import rai_from_indices


def main() -> None:
    """
    读取评估指标 JSON 文件，计算并打印 RAI 指标。

    期望的 JSON 输入格式：
    {
        "retention": [0.85, 0.78, 0.72],      # 每个任务结束后的 RI 列表
        "generalization": [0.92, 0.88, 0.85]  # 每个任务结束后的 GI 列表
    }

    该 JSON 通常由评估脚本在每个任务后计算 mAP 并生成。
    """
    parser = argparse.ArgumentParser(
        description="计算 DuET 双增量目标检测的 RAI（保留-适应度指数）指标"
    )
    parser.add_argument(
        "--metrics",
        required=True,
        type=Path,
        help="包含 retention 和 generalization 列表的 JSON 文件路径"
    )
    args = parser.parse_args()

    # 读取 JSON 指标文件
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))

    # 从 JSON 中提取 RI 和 GI 列表（将值转换为浮点数）
    retention = [float(x) for x in payload.get("retention", [])]
    generalization = [float(x) for x in payload.get("generalization", [])]

    # 调用核心函数计算 RAI 综合指标
    result = rai_from_indices(retention=retention, generalization=generalization)

    # 打印结果：保留指数、泛化指数和综合 RAI
    # Avg RI: 平均保留指数，反映旧任务的知识保留水平
    # Avg GI: 平均泛化指数，反映新任务的学习适应水平
    # RAI:   综合指标，等于 (Avg RI + Avg GI) / 2
    print(f"Avg RI: {result.avg_ri:.4f}")
    print(f"Avg GI: {result.avg_gi:.4f}")
    print(f"RAI:    {result.rai:.4f}")


if __name__ == "__main__":
    main()
