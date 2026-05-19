"""
DuET 任务向量 (Task Vector) 核心算法模块

本模块实现了 DuET 论文中任务向量方法的核心数学运算，是整个增量学习的理论基础。

核心概念：
  任务向量 (Task Vector)：描述模型从基准状态到某一任务特定状态的变化方向和幅度。
    delta_task = theta_task - theta_reference
  其中 theta_reference 是基准预训练模型的参数，theta_task 是经过任务微调后的模型参数。

关键算法：
  1. task_vector()：计算两个模型状态之间的任务向量
  2. cosine_direction_loss()：方向一致性损失，惩罚新旧任务向量方向冲突
  3. merge_state_dicts()：将多个任务向量合并到统一的模型中
  4. inject_state_dict_into_checkpoint()：将合并后的状态字典写回检查点文件

合并公式（DuET 核心）：
  theta_merged = theta_ref + alpha_old * delta_old + alpha_new * delta_new

这意味着合并模型的参数 = 基准模型参数 + 加权的旧任务增量 + 加权的新任务增量。
通过调节 alpha_old 和 alpha_new，可以控制旧知识与新知识在最终模型中的占比。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn


# 类型别名：PyTorch 状态字典
# 状态字典是参数字符串名称到张量的映射，如 {"model.0.conv.weight": tensor(...), ...}
StateDict = dict[str, Tensor]


def torch_load_checkpoint(path: str | Path, map_location: str = "cpu") -> object:
    """
    跨 PyTorch 版本安全加载本地训练检查点文件。

    由于 PyTorch 序列化格式在不同版本间可能有细微差异（如存储元数据的字典结构），
    此函数尝试使用 weights_only=False 加载，若遇到不支持的序列化类型
    （如旧版本中使用了 now-deprecated pickle 特性），则回退到不带 weights_only 的加载方式。

    Args:
        path: 检查点文件路径（通常为 .pt 或 .pth 文件）。
        map_location: 指定加载到 CPU 还是特定 GPU 设备。

    Returns:
        原始检查点对象（可能是 nn.Module、dict 或其他类型）。
    """
    try:
        # weights_only=False：允许加载包含任意 Python 对象的检查点
        # （包括非张量数据如优化器状态、随机数生成器状态等）
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 某些 PyTorch 版本中 weights_only=True 会拒绝加载包含自定义类实例的检查点
        return torch.load(path, map_location=map_location)


@dataclass
class MergeReport:
    """
    状态字典合并操作的统计报告。

    用于记录合并过程中各类参数键的处理情况，帮助调试和理解合并行为。
    """
    merged_keys: int       # 成功合并的参数量（浮点型共享参数）
    skipped_keys: int      # 跳过的参数量（形状不匹配或非浮点型参数）
    incompatible_keys: list[str]  # 形状不兼容的参数名称列表


def unwrap_checkpoint(ckpt: object) -> StateDict:
    """
    从各种常见格式的 PyTorch/Ultralytics 检查点中提取模型的状态字典。

    支持的检查点类型：
      1. nn.Module：直接调用 .state_dict()
      2. 字典类型：
         - 若包含 'model' 键且值为 nn.Module，从 model.state_dict() 获取
         - 若包含 'ema' 键且值为 nn.Module，从 ema.state_dict() 获取
         - 若包含 'state_dict' 键，从该键的值中提取所有张量
         - 否则将字典中所有 torch.Tensor 类型的值提取为状态字典

    Args:
        ckpt: PyTorch 检查点对象。

    Returns:
        标准格式的状态字典（参数字符串 -> 参数张量）。

    Raises:
        TypeError: 如果检查点类型不支持。
    """
    # 类型 1：nn.Module 对象，直接获取其 state_dict
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    # 类型 2：字典（Mapping），遍历常见键名寻找模型
    if isinstance(ckpt, Mapping):
        # 优先查找 'model' 和 'ema' 键
        for key in ("model", "ema", "state_dict"):
            value = ckpt.get(key)
            # 如果值是 nn.Module，递归调用获取其 state_dict
            if isinstance(value, nn.Module):
                return value.state_dict()
            # 如果值是普通字典，提取其中的张量
            if isinstance(value, Mapping):
                return {str(k): v for k, v in value.items() if torch.is_tensor(v)}
        # 如果没有标准的 model/ema/state_dict 键，则将所有张量收集为状态字典
        return {str(k): v for k, v in ckpt.items() if torch.is_tensor(v)}

    raise TypeError(f"不支持的检查点类型: {type(ckpt)!r}")


def load_state_dict(path: str | Path, map_location: str = "cpu") -> StateDict:
    """
    便捷函数：从检查点文件路径加载模型状态字典。

    整合了 torch_load_checkpoint 和 unwrap_checkpoint 的逻辑，
    提供从文件路径直接获取状态字典的接口。

    Args:
        path: 检查点文件路径。
        map_location: 加载设备（默认 CPU）。

    Returns:
        模型参数字典。
    """
    ckpt = torch_load_checkpoint(path, map_location=map_location)
    return unwrap_checkpoint(ckpt)


def key_is_shared(key: str, exclude_patterns: Iterable[str]) -> bool:
    """
    判断某个参数键是否属于"共享"参数（即可以被任务向量合并的层）。

    在 YOLO 等检测器中，检测头（detect head）的参数是任务特定的：
    它们的维度与检测类别数量直接相关，不同任务的类别数可能不同，
    因此不适合参与跨任务的向量合并。

    本函数通过排除列表（shared_key_exclude）判断参数是否属于任务特定层。
    排除列表中通常包含：detect（检测头）、dfl（分布式 Focal Loss）、cv2/cv3（辅助预测头）。

    Args:
        key: 参数名称，如 "model.22.detect.0.conv.weight"。
        exclude_patterns: 需要排除的模式列表，如 ["detect", "dfl", "cv2", "cv3"]。

    Returns:
        若参数名称中不包含任何排除模式，返回 True（是共享参数）；否则返回 False。
    """
    lowered = key.lower()
    # 排除列表中的任意一个模式出现在 key 中，则该参数不共享
    return not any(pattern.lower() in lowered for pattern in exclude_patterns)


def task_vector(
    reference: StateDict,
    target: StateDict,
    shared_key_exclude: Iterable[str] = (),
    device="cuda:0",
) -> StateDict:
    """
    计算任务向量：目标模型相对于基准模型的变化量。

    任务向量是 DuET 方法的核心概念，表示模型在特定任务上学习到的共享参数调整方向。
    数学定义：delta = theta_target - theta_reference

    与论文一致：只计算共享参数（shared parameters），即 backbone 和 neck 等
    跨任务可融合的参数。检测头参数（cv2、cv3、dfl 等）属于任务特定参数，
    因维度与类别数相关且形状可能不一致，不参与任务向量计算。

    仅处理满足以下条件的参数：
      1. 参数名称在 reference 和 target 中都存在
      2. 参数形状（shape）在两个状态字典中完全一致
      3. 参数是浮点型张量（排除整数类型的位置索引、布尔掩码等）
      4. 参数名称不在排除列表中（不属于检测头等任务特定层）

    Args:
        reference: 基准模型的状态字典（通常是预训练模型的参数）。
        target: 目标模型的状态字典（经过任务微调后的模型参数）。
        shared_key_exclude: 共享参数排除模式列表，如 ["detect", "dfl", "cv2", "cv3"]。
                           名称中包含任一模式的参数不参与任务向量计算。

    Returns:
        任务向量字典，只包含共享参数的变化量张量。
    """
    delta: StateDict = {}
    for key, ref_value in reference.items():
        tgt_value = target.get(key).to(device)
        ref_value = ref_value.to(device)
        if tgt_value is None:
            continue
        if ref_value.shape != tgt_value.shape:
            continue
        if not torch.is_floating_point(ref_value):
            continue
        if not key_is_shared(key, shared_key_exclude):
            continue
        delta[key] = tgt_value.detach().float() - ref_value.detach().float()
    return delta


def merge_state_dicts(
    reference: StateDict,
    old_model: StateDict,
    new_model: StateDict,
    *,
    alpha_old: float = 1.0,
    alpha_new: float = 1.0,
    shared_key_exclude: Iterable[str] = (),
) -> tuple[StateDict, MergeReport]:
    """
    将旧任务和新任务的任务向量合并到基准模型中。

    这是 DuET 方法的核心合并操作，实现了双增量目标检测的知识融合。

    合并逻辑：
      对于"共享参数"（如骨干网络 backbone 和颈部网络 neck）：
        theta_merged[key] = theta_ref[key] + alpha_old * delta_old[key] + alpha_new * delta_new[key]

      对于"任务特定参数"（如检测头 detect head）：
        - YOLO 检测头包含与类别数相关的全连接层
        - 不同任务的类别数不同导致维度不匹配，不能直接相加
        - 直接使用新模型的参数（假设新模型有更新颖的知识）

    共享参数判断条件（必须同时满足）：
      1. 该参数在 reference、old_model、new_model 中都存在
      2. 三个状态字典中该参数的形状完全一致
      3. 参数是浮点型张量
      4. 参数名称不在排除列表中（不属于检测头等任务特定层）

    Args:
        reference: 基准模型状态字典（预训练权重）。
        old_model: 旧任务训练后的模型状态字典。
        new_model: 新任务训练后的模型状态字典。
        alpha_old: 旧任务向量的加权系数，控制旧知识在合并中的重要性。
        alpha_new: 新任务向量的加权系数，控制新知识在合并中的重要性。
        shared_key_exclude: 不参与合并的参数名称模式列表（如检测头层）。

    Returns:
        合并后的状态字典，以及包含合并统计信息的 MergeReport 对象。
    """
    # Step 1: 分别计算旧任务和新任务相对于基准的任务向量（只含共享参数）
    old_delta = task_vector(reference, old_model, shared_key_exclude=shared_key_exclude)
    new_delta = task_vector(reference, new_model, shared_key_exclude=shared_key_exclude)

    merged: StateDict = {}           # 存储合并后的参数
    incompatible: list[str] = []    # 记录形状不兼容的参数名
    merged_keys = 0                  # 成功合并的参数数量
    skipped_keys = 0                 # 跳过的参数数量

    # 遍历所有参数（取三个状态字典的并集）
    all_keys = set(reference) | set(old_model) | set(new_model)
    for key in sorted(all_keys):
        ref_value = reference.get(key)
        old_value = old_model.get(key)
        new_value = new_model.get(key)

        # 如果新模型中没有该参数（被完全删除），跳过
        if new_value is None:
            skipped_keys += 1
            continue

        # 检查是否满足"共享参数"的条件
        is_shared = (
            ref_value is not None        # 在基准模型中存在
            and old_value is not None    # 在旧模型中存在
            and ref_value.shape == old_value.shape == new_value.shape   # 三者形状一致
            and torch.is_floating_point(ref_value)                     # 是浮点型参数
            and key_is_shared(key, shared_key_exclude)                # 不在排除列表中
        )

        if is_shared:
            # 共享参数：执行 DuET 合并公式
            # theta_merged = theta_ref + alpha_old * delta_old + alpha_new * delta_new
            merged[key] = (
                ref_value.detach().float()
                + alpha_old * old_delta[key]
                + alpha_new * new_delta[key]
            ).to(dtype=new_value.dtype)  # 转换回原始数据类型
            merged_keys += 1
        else:
            # 任务特定参数（形状不一致或被排除）：直接使用新模型的参数
            if ref_value is not None and ref_value.shape != new_value.shape:
                # 特别记录形状不兼容的情况（调试用）
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
    """
    将合并后的状态字典写入新的检查点文件。

    此函数读取原始检查点，用合并后的状态字典替换其中的模型权重，
    然后保存为新文件。它能正确处理 Ultralytics 特有的检查点结构
    （可能包含 model、ema 等嵌套 nn.Module 对象）。

    Args:
        checkpoint_path: 原始检查点文件路径（通常是新任务训练后的 best.pt）。
        merged_state: 已合并的模型状态字典。
        output_path: 输出检查点文件路径。
        map_location: 加载原始检查点时使用的设备（默认 CPU）。
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    # 加载原始检查点
    ckpt = torch_load_checkpoint(checkpoint_path, map_location=map_location)

    if isinstance(ckpt, nn.Module):
        # 检查点是 nn.Module 对象，直接替换其状态字典
        ckpt.load_state_dict(merged_state, strict=False)
    elif isinstance(ckpt, dict):
        # 检查点是字典类型（Ultralytics 的标准格式）
        # 尝试更新嵌套的 nn.Module 对象（model 或 ema）
        updated = False
        for key in ("model", "ema"):
            value = ckpt.get(key)
            if isinstance(value, nn.Module):
                value.load_state_dict(merged_state, strict=False)
                updated = True
        if not updated:
            # 如果没有找到 model/ema 键，直接设置 state_dict 键
            ckpt["state_dict"] = merged_state
    else:
        raise TypeError(f"不支持的检查点类型: {type(ckpt)!r}")

    # 确保输出目录存在，并保存合并后的检查点
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
