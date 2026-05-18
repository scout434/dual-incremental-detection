"""
DuET (Dual Incremental Object Detection) 主训练脚本

该脚本实现了 ICCV 2025 论文 DuET 的核心训练流程。
通过任务向量(Task Vector)合并技术，在 YOLO 检测器上实现双增量学习：
1. 类别增量学习 (Class-Incremental Detection)：逐步增加可检测的物体类别
2. 域增量学习 (Domain-Incremental Detection)：逐步适应不同的视觉域（天气，光照等）

核心流程（修正版）：
  1. 任务 t 训练开始前：仅扩展 current_weights 的检测头类别数（复制旧类别权重，初始化新类别为0）
  2. 训练当前任务 → 得到 trained_ckpt（此时检测头已扩展，但只有旧类别权重被复制）
  3. 任务 t 合并时：
     - 共享参数（backbone + neck）：DuET Module 逐层动态融合（Eq 12）
     - 检测头参数（新扩的部分）：拼接 trained_ckpt 的检测头 与 old_ckpt 的检测头（Eq 13）
  4. 保存 merged_ckpt，作为下一任务的 current_weights
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from duet_repro.core.task_vectors import (
    inject_state_dict_into_checkpoint,
    load_state_dict,
    task_vector,
)


def seed_everything(seed: int) -> None:
    """设置所有随机种子以保证实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DuETDetectionTrainer:
    """
    通过 monkey-patch BaseTrainer.optimizer_step 在每个 optimizer step 后动态更新任务向量的封装。

    原理：patch 在类层面生效，当 model.train() 内部创建 DetectionTrainer 实例时，
    该实例的 optimizer_step 自动使用被 patch 后的版本，无需手动创建 trainer。

    patch 完成后调用 model.train() 启动训练，训练结束后还原原始方法。
    """

    _patched: bool = False  # 类变量，记录是否已 patch（避免重复 patch）

    def __init__(
        self,
        base_trainer_cls,
        reference_state: dict,
        shared_key_exclude: tuple[str, ...],
    ):
        from ultralytics.utils.torch_utils import unwrap_model

        self._ref_state = reference_state
        self._shared_exclude = shared_key_exclude

        if DuETDetectionTrainer._patched:
            return

        # 保存原始的 optimizer_step
        _orig = base_trainer_cls.optimizer_step

        def _patched_step(self_) -> None:
            _orig(self_)  # 先完成参数更新
            # 参数已更新，计算当前任务向量
            criterion = getattr(unwrap_model(self_.model), "criterion", None)
            if criterion is None or not hasattr(criterion, "set_curr_task_vector"):
                return
            curr_state = {
                k: v.detach().clone()
                for k, v in unwrap_model(self_.model).state_dict().items()
            }
            curr_tv = task_vector(
                self._ref_state,
                curr_state,
                shared_key_exclude=self._shared_exclude,
            )
            criterion.set_curr_task_vector(curr_tv)

        # 类层面 patch（所有实例共享）
        base_trainer_cls.optimizer_step = _patched_step
        DuETDetectionTrainer._patched = True


def load_config(path: str | Path) -> dict:
    """加载 YAML 格式的实验配置文件。"""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_one_task(
    weights: str | Path,
    task: dict,
    cfg: dict,
    project_dir: Path,
    teacher_ckpt: str | Path | None = None,
    distill_weight: float = 0.0,
    dc_weight: float = 0.0,
    prev_task_vector: dict | None = None,
    task_vector_history: list[dict] | None = None,
    reference_state: dict | None = None,
    shared_key_exclude: tuple[str, ...] = (),
) -> Path:
    """
    训练单个任务。

    Args:
        weights: 初始权重文件路径。
        task: 任务配置字典，包含 'data' 和 'name' 字段。
        cfg: 全局实验配置字典。
        project_dir: Ultralytics 训练输出目录。
        teacher_ckpt: 旧任务模型路径（用于蒸馏）。
        distill_weight: 蒸馏损失权重。
        dc_weight: 方向一致性损失权重。
        prev_task_vector: 上一轮合并后的累积任务向量（用于 LDC）。
        task_vector_history: 历史任务向量列表（用于 LDC）。
        reference_state: 基准预训练模型状态字典（用于动态任务向量计算）。
        shared_key_exclude: 共享参数排除模式。

    Returns:
        训练产生的最佳模型检查点路径。0.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))

    # 设置 criterion（如果需要）
    criterion = None
    if teacher_ckpt is not None and (distill_weight > 0 or dc_weight > 0):
        from duet_repro.core.duet_loss import create_duet_criterion
        # 创建教师模型
        teacher = YOLO(str(teacher_ckpt)).model
        # 教师模型移动到device上
        teacher.to(cfg["training"].get("device", 0))
        criterion = create_duet_criterion(
            model=model.model,
            teacher_model=teacher,
            distill_weight=distill_weight,
            dc_weight=dc_weight,
        )
        if prev_task_vector is not None:
            criterion.set_curr_task_vector(prev_task_vector)
        if task_vector_history:
            for tv in task_vector_history:
                criterion.record_task_vector(tv)

        # 通过回调函数在训练开始前替换trainer的损失函数
        def on_train_start(trainer):
            from ultralytics.utils.torch_utils import unwrap_model
            original_model = unwrap_model(trainer.model)
            # 1.给损失函数的参数进行赋值
            if hasattr(original_model, "args"):
                criterion.hyp = original_model.args

            # 2.强制替换训练器中的criterion
            original_model.criterion=criterion

            # 3.强制替换EMA模型的criterion
            if trainer.ema and hasattr(trainer.ema, 'ema'):
                ema_model = unwrap_model(trainer.ema.ema)
                ema_model.criterion = criterion

            # 4.设置trainer的损失名称
            loss_names = list(trainer.loss_names)
            if distill_weight > 0:
                loss_names.append('distill_loss')
            if dc_weight > 0:
                loss_names.append('dc_loss')
            trainer.loss_names = loss_names

        model.add_callback("on_train_start", on_train_start)

    # 构建训练参数
    training = cfg["training"]
    model.train(
        data=task["data"],
        epochs=training.get("epochs", 50),
        imgsz=training.get("imgsz", 640),
        batch=training.get("batch", 16),
        workers=training.get("workers", 4),
        device=training.get("device", 0),
        optimizer=training.get("optimizer", "auto"),
        lr0=training.get("lr0", 0.01),
        project=str(project_dir / task["name"]),
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    if not best.exists():
        raise FileNotFoundError(f"Ultralytics 未生成最佳检查点文件: {best}")
    return best


def _get_task_nc(data_yaml: str | Path) -> int:
    """从 YOLO 数据集配置文件中读取类别数。"""
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return len(data.get("names", {}))


def _expand_detect_head(model, old_nc: int, new_nc: int) -> None:
    """
    扩展 YOLO 检测头的分类输出层以支持更多类别。

    操作逻辑：
      1. 创建新的 Conv2d 层，输出通道数扩展到 new_nc
      2. 将原始的 old_nc 个通道的权重/偏置复制到新层对应位置
      3. 新增的 (new_nc - old_nc) 个通道初始化为 0

    注意：此函数只扩展，不拼接。

    Args:
        model: YOLO 检测模型。
        old_nc: 扩展前的类别数。
        new_nc: 扩展后的类别数。
    """
    import torch.nn as nn
    from duet_repro.core.incremental_head import expand_detect_head as _expand

    if old_nc >= new_nc:
        return
    _expand(model, old_nc, new_nc, preserve_old=True)


def _extract_head_params(state_dict: dict) -> dict:
    """
    从完整状态字典中提取检测头相关的参数。

    检测头参数包括 cv2、cv3 和 dfl 的所有层。

    Ultralytics 的实际 key 格式是 "model.22.cv3.0.2.weight"（不含 [-1] 字面量），
    因此使用正则表达式匹配：检测头层序号（通常为最后一层，即最大数字）+ 子模块名。

    Args:
        state_dict: 完整的状态字典。

    Returns:
        仅包含检测头参数的子字典。
    """
    import re

    # 提取所有以 "model." 为前缀的参数键，找出其中层号最大的（即检测头层）
    model_keys = [k for k in state_dict.keys() if k.startswith("model.")]
    layer_numbers = set()
    for k in model_keys:
        m = re.match(r"model\.(\d+)\.", k)
        if m:
            layer_numbers.add(int(m.group(1)))

    if not layer_numbers:
        return {}

    head_layer = max(layer_numbers)
    # 匹配该层下的 cv2、cv3、dfl 子模块
    pattern = rf"^model\.{head_layer}\.(cv2|cv3|dfl)\."
    compiled = re.compile(pattern)
    return {k: v for k, v in state_dict.items() if compiled.match(k)}


def main(config_path: str = "configs/pascal_series_yolo.yaml") -> None:
    """
    DuET 双增量目标检测训练流程。

    正确流程（每轮任务 t）：
      1. 训练开始前：仅扩展 current_weights 的检测头类别数
         - 加载 current_weights
         - 扩展检测头：old_nc -> old_nc + task_nc
         - 旧类别权重复制到新层，新类别权重初始化为 0
         - 保存扩展后的检查点
      2. 训练：在扩展后的模型上训练 -> trained_ckpt
      3. 合并前准备：
         - 加载 trained_ckpt 的检测头参数（已扩展）
         - 加载 old_ckpt 的检测头参数（旧合并模型）
      4. 合并：
         - 共享参数：DuET Module 逐层动态融合（Eq 12）
         - 检测头参数：拼接 trained_ckpt（已扩展）的检测头 与 old_ckpt 的检测头（Eq 13）
      5. 保存 merged_ckpt，作为下一任务的 current_weights 和 old_ckpt

    Args:
        config_path: YAML 配置文件路径。默认为 configs/pascal_series_yolo.yaml，
                   可通过命令行 --config 参数覆盖。
    """
    parser = argparse.ArgumentParser(description="DuET 双增量目标检测训练脚本")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config if args.config else config_path)
    seed_everything(int(cfg["experiment"].get("seed", 42)))

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = output_dir / "runs"

    base_weights = cfg["detector"]["base_weights"]
    duet_cfg = cfg.get("duet", {})
    tasks = cfg["tasks"]

    history: list[dict] = []

    reference_state = load_state_dict(base_weights)

    # 当前权重路径（上一轮 merged_ckpt）
    current_weights = base_weights

    # 旧 merged 检查点路径（用于蒸馏和检测头拼接）
    old_ckpt: str | Path | None = None

    # 累积已学习的总类别数
    cumulative_nc = 0

    # 历史任务向量列表，记录每个已合并任务的累积任务向量 (θ_{task} - θ_ref)
    # 用于在后续任务中计算方向一致性损失 LDC
    task_vector_history: list[dict] = []

    # 在第一个任务训练前 patch BaseTrainer.optimizer_step
    # patch 成功后，每个 optimizer step 后都会动态更新 curr_task_vector
    if float(duet_cfg.get("dc_weight", 0.0)) > 0 or float(duet_cfg.get("distill_weight", 0.0)) > 0:
        from ultralytics.models.yolo.detect.train import DetectionTrainer

        DuETDetectionTrainer(
            base_trainer_cls=DetectionTrainer,
            reference_state=reference_state,
            shared_key_exclude=tuple(duet_cfg.get("shared_key_exclude", [])),
        )
        print("[DuET] optimizer_step 已 patch，训练中将动态更新任务向量")

    for task_index, task in enumerate(tasks, start=1):
        task_name = task["name"]
        data_yaml = task["data"]
        task_nc = _get_task_nc(data_yaml)
        new_total_nc = task_nc

        print(f"\n{'=' * 60}")
        print(f"[DuET] 任务 {task_index}/{len(tasks)}: {task_name}")
        print(f"{'=' * 60}")

        # ================================================================
        # 步骤 1：训练开始前——仅扩展检测头类别数
        # ================================================================
        if task_index >= 2:
            print(f"[Incremental Head] 扩展检测头：{cumulative_nc} 类 -> {new_total_nc} 类")
            try:
                from ultralytics import YOLO

                model_expanded = YOLO(str(current_weights))
                _expand_detect_head(model_expanded, cumulative_nc, new_total_nc)

                expanded_ckpt = output_dir / f"task_{task_index}_{task_name}_expanded.pt"
                model_expanded.save(str(expanded_ckpt))
                current_weights = expanded_ckpt
                print(f"[Incremental Head] 检测头已扩展，模型保存至: {expanded_ckpt}")
            except Exception as e:
                print(f"[Incremental Head] 扩展失败，使用原始权重继续: {e}")

        # ================================================================
        # 步骤 2：训练当前任务
        # ================================================================
        prev_tv = None
        if task_index >= 2 and old_ckpt is not None:
            # prev_task_vector = merged_{t-1} - reference，即 (θ_{t-1} - θ_0) 共享部分
            prev_tv = task_vector(
                reference_state,
                load_state_dict(old_ckpt),
                shared_key_exclude=tuple(duet_cfg.get("shared_key_exclude", [])),
            )

        print(f"[训练] 使用权重: {current_weights}")
        trained_ckpt = train_one_task(
            current_weights,
            task,
            cfg,
            project_dir,
            teacher_ckpt=old_ckpt if task_index >= 2 else None,
            distill_weight=float(duet_cfg.get("distill_weight", 0.0)),
            dc_weight=float(duet_cfg.get("dc_weight", 0.0)),
            prev_task_vector=prev_tv,
            task_vector_history=task_vector_history,
            reference_state=reference_state,
            shared_key_exclude=tuple(duet_cfg.get("shared_key_exclude", [])),
        )
        print(f"[训练] 训练完成，检查点: {trained_ckpt}")

        # ================================================================
        # 步骤 3：合并
        # ================================================================
        if task_index == 1 or not duet_cfg.get("enabled", True):
            # 第一个任务：直接保存
            merged_state = load_state_dict(trained_ckpt)
            merged_ckpt = output_dir / f"task_{task_index}_{task_name}_best.pt"
            inject_state_dict_into_checkpoint(
                trained_ckpt,
                merged_state,
                merged_ckpt,
            )
            print(f"[合并] 第一个任务，直接保存: {merged_ckpt}")
        else:
            # 增量任务：共享参数走 DuET Module，检测头参数走拼接
            if old_ckpt is None:
                raise RuntimeError("old_ckpt 应在第一个任务后被赋值")

            from ultralytics import YOLO

            # 加载三个关键状态字典
            reference = reference_state
            old_state = load_state_dict(old_ckpt)
            new_state = load_state_dict(trained_ckpt)

            # 加载 trained_ckpt 和 old_ckpt 的检测头参数
            # trained_ckpt 的检测头已经被 expand_detect_head 扩展过
            # old_ckpt 的检测头是上一轮合并后的结果
            trained_model = YOLO(str(trained_ckpt))
            trained_head_sd = _extract_head_params(trained_model.model.state_dict())
            old_head_sd = _extract_head_params(old_state)

            # 获取 DuET Module 的超参数
            gamma = float(duet_cfg.get("gamma", 0.5))
            alpha_base = float(duet_cfg.get("alpha_base", 0.5))
            use_duet_module = duet_cfg.get("use_duet_module", True)

            if use_duet_module:
                # ================================================
                # 共享参数：DuET Module 逐层动态融合（Eq 12）
                # 检测头参数：拼接（Eq 13）
                # ================================================
                from duet_repro.core.duet_module import merge_state_dicts_with_duet_module

                print(f"[DuET Module] 逐层动态融合 (gamma={gamma}, alpha_base={alpha_base})")
                print(f"[Incremental Head] 检测头拼接 (旧 {cumulative_nc} 类, 新 {new_total_nc} 类)")

                merged_state, report = merge_state_dicts_with_duet_module(
                    reference,
                    old_state,
                    new_state,
                    gamma=gamma,
                    alpha_base=alpha_base,
                    shared_key_exclude=duet_cfg.get("shared_key_exclude", []),
                    per_layer_report=duet_cfg.get("verbose_merge", False),
                    # 检测头拼接参数
                    old_head_sd=old_head_sd,
                    new_head_sd=trained_head_sd,
                    old_nc=cumulative_nc,
                    new_total_nc=new_total_nc,
                )

                print(f"[DuET Module] 已合并 {report['merged_keys']} 层，跳过 {report['skipped_keys']} 层")

                if duet_cfg.get("verbose_merge", False) and "per_layer_alphas" in report:
                    print("[DuET Module] 各层 αl 系数:")
                    for k, v in sorted(report["per_layer_alphas"].items()):
                        print(f"  {k}: α={v:.4f}")
            else:
                # 回退到简单固定系数合并（仅共享参数，检测头参数直接用新模型）
                from duet_repro.core.task_vectors import merge_state_dicts as simple_merge

                alpha_old = float(duet_cfg.get("alpha_old", 1.0))
                alpha_new = float(duet_cfg.get("alpha_new", 1.0))
                print(f"[DuET] 固定系数融合 (alpha_old={alpha_old}, alpha_new={alpha_new})")

                merged_state, report = simple_merge(
                    reference,
                    old_state,
                    new_state,
                    alpha_old=alpha_old,
                    alpha_new=alpha_new,
                    shared_key_exclude=duet_cfg.get("shared_key_exclude", []),
                )
                print(f"[DuET] 已合并 {report.merged_keys} 层，跳过 {report.skipped_keys} 层")

                # 固定系数模式下，检测头参数直接用新模型的（已扩展）
                # 不做拼接，这意味着旧类别的检测头知识会丢失
                # 在域增量（nc 不变）中这是可以接受的

            merged_ckpt = output_dir / f"task_{task_index}_{task_name}_duet.pt"
            inject_state_dict_into_checkpoint(trained_ckpt, merged_state, merged_ckpt)
            print(f"[合并] 检查点已保存至: {merged_ckpt}")

        # ================================================================
        # 步骤 4：记录任务向量 & 更新循环变量
        # ================================================================
        # 计算当前任务的累积任务向量（共享参数部分），用于后续任务计算 LDC
        current_task_tv = task_vector(
            reference_state,
            merged_state,
            shared_key_exclude=tuple(duet_cfg.get("shared_key_exclude", [])),
        )
        task_vector_history.append(current_task_tv)

        cumulative_nc = new_total_nc
        old_ckpt = merged_ckpt
        current_weights = merged_ckpt

        history.append({
            "task": task_name,
            "num_classes": task_nc,
            "cumulative_classes": cumulative_nc,
            "trained_checkpoint": str(trained_ckpt),
            "merged_checkpoint": str(merged_ckpt),
        })

    # 保存训练历史
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[DuET] 训练完成。历史记录已保存至 {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    # 默认使用 configs/pascal_series_yolo.yaml，方便在 PyCharm 中直接运行
    main(config_path="configs/human_activity_yolo.yaml")
