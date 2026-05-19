"""
DuET 自动评估脚本

该脚本用于在每个任务训练完成后自动计算各任务的 mAP，
生成 RAI 指标计算所需的 JSON 文件，并计算最终 RAI。

使用流程：
  1. 运行训练脚本 train_ultralytics_duet.py
  2. 运行评估脚本 python eval_duet.py --history outputs/xxx/training_history.json --data-config configs/xxx.yaml
  3. 脚本会自动：
     - 加载每个任务的合并检查点
     - 在所有历史任务的验证集上计算 mAP
     - 生成 RAI 指标 JSON
     - 打印 RAI 报告
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def load_training_history(history_path: str | Path) -> list[dict]:
    """加载训练历史 JSON 文件。"""
    with open(history_path, encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: str | Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_single_checkpoint(
    checkpoint_path: str | Path,
    data_yaml: str | Path,
    device: int = 0,
) -> dict[str, float]:
    """
    评估单个检查点模型，返回各任务的 mAP。

    使用 Ultralytics 的验证器来评估模型。

    Args:
        checkpoint_path: 模型检查点路径。
        data_yaml: 数据集配置文件路径。
        device: GPU 设备编号。

    Returns:
        包含各任务 mAP 的字典。键为任务名称，值为 mAP@0.5。
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint_path))
    results = model.val(data=str(data_yaml), device=device, verbose=False)

    # 从结果中提取 mAP@0.5
    # Ultralytics 的验证结果通常包含以下属性：
    # results.box.map50    -> mAP@0.5
    # results.box.map      -> mAP@0.5:0.95
    # results.box.map75    -> mAP@0.75
    mAP50 = float(getattr(results, "box", None) or results).map50

    return {"mAP@0.5": mAP50}


def compute_retention_index(current: float, reference: float) -> float:
    """
    计算保留指数 (RI)。

    RI = min(current_mAP / reference_mAP, 1.0)

    Args:
        current: 当前模型在旧任务上的 mAP。
        reference: 参考模型（单任务基线）在同任务上的 mAP。

    Returns:
        保留指数，范围 [0, 1]。
    """
    if reference <= 1e-12:
        return 0.0
    return min(current / reference, 1.0)


def compute_generalization_index(current: float, reference: float) -> float:
    """
    计算泛化指数 (GI)。

    GI = min(current_mAP / reference_mAP, 1.0)

    Args:
        current: 当前模型在新任务上的 mAP。
        reference: 参考模型在同任务上的 mAP。

    Returns:
        泛化指数，范围 [0, 1]。
    """
    if reference <= 1e-12:
        return 0.0
    return min(current / reference, 1.0)


def compute_rai(retention_list: list[float], generalization_list: list[float]) -> dict[str, float]:
    """
    计算 RAI 综合指标。

    RAI = (Avg RI + Avg GI) / 2

    Args:
        retention_list: 各任务的保留指数列表。
        generalization_list: 各任务的泛化指数列表。

    Returns:
        包含 avg_ri、avg_gi 和 rai 的字典。
    """
    avg_ri = sum(retention_list) / len(retention_list) if retention_list else 0.0
    avg_gi = sum(generalization_list) / len(generalization_list) if generalization_list else 0.0
    rai = (avg_ri + avg_gi) / 2.0
    return {
        "avg_ri": avg_ri,
        "avg_gi": avg_gi,
        "rai": rai,
    }


def evaluate_and_generate_rai(
    history: list[dict],
    tasks: list[dict],
    baseline_maps: dict[str, float] | None = None,
    device: int = 0,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    评估所有任务的合并检查点，生成 RAI 指标。

    评估策略：
      对于每个合并检查点（代表任务 t 的最终模型）：
        1. 在任务 1 到任务 t 的验证集上评估，得到各任务 mAP
        2. 用当前 mAP / 参考 mAP 计算各任务的 RI
        3. 当前 mAP / 新任务参考 mAP 计算 GI

    Args:
        history: 训练历史列表，每个元素包含 task、trained_checkpoint、merged_checkpoint。
        tasks: 任务配置列表，每个元素包含 name 和 data。
        baseline_maps: 单任务基线的 mAP 字典，键为任务名，值为参考 mAP。
                     如果为 None，将跳过 RI 和 GI 计算，仅报告原始 mAP。
        device: GPU 设备编号。
        output_path: 输出 JSON 文件路径。

    Returns:
        完整的评估结果字典。
    """
    results = {
        "per_checkpoint_evaluation": {},
        "retention": [],
        "generalization": [],
        "baseline_maps": baseline_maps or {},
    }

    print("\n" + "=" * 60)
    print("DuET 评估流程")
    print("=" * 60)

    # 构建任务名到数据集路径的映射
    task_data_map = {t["name"]: t["data"] for t in tasks}

    for idx, history_entry in enumerate(history, start=1):
        task_name = history_entry["task"]
        merged_ckpt = history_entry["merged_checkpoint"]

        print(f"\n[评估 {idx}/{len(history)}] 任务: {task_name}")
        print(f"  检查点: {merged_ckpt}")

        if not Path(merged_ckpt).exists():
            print(f"  警告：检查点不存在，跳过！")
            continue

        # 评估该模型在所有已学习任务上的表现
        task_maps = {}
        for eval_task in history[:idx]:
            eval_task_name = eval_task["task"]
            data_yaml = task_data_map.get(eval_task_name)

            if data_yaml is None:
                continue

            print(f"  评估任务 '{eval_task_name}'...", end=" ", flush=True)
            try:
                mAP = evaluate_single_checkpoint(merged_ckpt, data_yaml, device)
                task_maps[eval_task_name] = mAP["mAP@0.5"]
                print(f"mAP@0.5 = {mAP['mAP@0.5']:.4f}")
            except Exception as e:
                print(f"失败: {e}")
                task_maps[eval_task_name] = 0.0

        results["per_checkpoint_evaluation"][task_name] = task_maps

        # 计算该任务结束后的 RI 和 GI
        if baseline_maps:
            # RI: 在旧任务上的知识保持
            # 每个已学习任务的 RI = 当前 mAP / 基线 mAP
            ri_list = []
            for prev_task_name in list(task_maps.keys())[:-1]:  # 排除当前任务本身
                if prev_task_name in baseline_maps:
                    ri = compute_retention_index(
                        task_maps[prev_task_name],
                        baseline_maps[prev_task_name],
                    )
                    ri_list.append(ri)

            # GI: 在新任务上的适应能力
            # GI = 当前 mAP（最终任务）/ 基线 mAP（新任务）
            gi = 0.0
            if task_name in baseline_maps:
                # 使用当前检查点在新任务上的 mAP 作为当前性能
                current_map = task_maps.get(task_name, 0.0)
                reference_map = baseline_maps[task_name]
                gi = compute_generalization_index(current_map, reference_map)

            results["retention"].append(sum(ri_list) / len(ri_list) if ri_list else 0.0)
            results["generalization"].append(gi)

    # 计算最终 RAI
    if results["retention"] and results["generalization"]:
        rai_results = compute_rai(results["retention"], results["generalization"])
        results["rai_summary"] = rai_results

        print("\n" + "=" * 60)
        print("RAI 评估结果")
        print("=" * 60)
        print(f"  Avg RI (保留指数): {rai_results['avg_ri']:.4f}")
        print(f"  Avg GI (泛化指数): {rai_results['avg_gi']:.4f}")
        print(f"  RAI:              {rai_results['rai']:.4f}")
        print("=" * 60)

    # 保存结果
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存至: {output_path}")

    return results


def main() -> None:
    """
    DuET 评估脚本主入口。

    使用方式：
      python eval_duet.py \
          --history outputs/pascal_series/training_history.json \
          --data-config configs/pascal_series_yolo.yaml \
          --baselines baselines.json \
          --output outputs/pascal_series/evaluation_results.json

    其中 baselines.json 格式：
      {
        "task1_base": 0.45,
        "task2_inc": 0.52,
        ...
      }
    """
    parser = argparse.ArgumentParser(
        description="DuET 双增量目标检测的自动评估脚本"
    )
    parser.add_argument(
        "--history",
        required=True,
        type=Path,
        help="训练历史 JSON 文件路径（由 train_ultralytics_duet.py 生成）"
    )
    parser.add_argument(
        "--data-config",
        required=True,
        type=Path,
        help="实验配置文件路径（YAML 格式），包含任务列表"
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        help="单任务基线 mAP 的 JSON 文件路径（可选）。"
             "如果提供，将计算 RI 和 GI 指标。"
             "格式：{'task_name': mAP_value, ...}"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="评估结果输出路径（JSON 格式）"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU 设备编号，默认 0"
    )
    args = parser.parse_args()

    # 加载训练历史
    history = load_training_history(args.history)
    print(f"已加载训练历史：{len(history)} 个任务")

    # 加载配置文件
    cfg = load_config(args.data_config)
    tasks = cfg["tasks"]
    print(f"已加载任务配置：{len(tasks)} 个任务")

    # 加载基线 mAP
    baseline_maps = None
    if args.baselines and args.baselines.exists():
        with open(args.baselines, encoding="utf-8") as f:
            baseline_maps = json.load(f)
        print(f"已加载基线 mAP：{len(baseline_maps)} 个任务")

    # 确定输出路径
    if args.output is None:
        output_path = args.history.parent / "evaluation_results.json"
    else:
        output_path = args.output

    # 执行评估
    results = evaluate_and_generate_rai(
        history=history,
        tasks=tasks,
        baseline_maps=baseline_maps,
        device=args.device,
        output_path=output_path,
    )

    # 生成 RAI 指标的简化 JSON（用于 eval_rai.py）
    if "rai_summary" in results:
        rai_payload = {
            "retention": results["retention"],
            "generalization": results["generalization"],
        }
        rai_json_path = args.history.parent / "rai_metrics.json"
        with open(rai_json_path, "w", encoding="utf-8") as f:
            json.dump(rai_payload, f, indent=2, ensure_ascii=False)
        print(f"\nRAI 指标 JSON 已保存至: {rai_json_path}")


if __name__ == "__main__":
    main()
