"""
DuET Incremental Head: 增量检测头模块

实现论文 DuET (ICCV 2025) Section 3.4 中的 Incremental Head 逻辑。

Incremental Head 的核心思想：
  将当前任务的任务特定参数（θτt）与历史任务的任务特定参数（(θτt-1)_incre）
  进行拼接（concatenate），从而扩展检测器的类别识别能力。

数学公式（论文 Eq 13）：
  (θτt)_incre ← [θτt; (θτt-1)_incre]

其中：
  - θτt 是当前任务训练后的检测头参数
  - (θτt-1)_incre 是上一轮合并后的检测头参数
  - [...] 表示参数拼接

在 YOLO 检测器中，检测头的分类分支（cv3）的最终输出层形状为：
  - cv3[i][-1].weight: (num_classes, c3, 1, 1)
  - cv3[i][-1].bias: (num_classes,)

拼接操作只针对分类头的最后一层，因为它的输出维度直接与类别数相关。
"""

from __future__ import annotations

import torch
import torch.nn as nn




def expand_detect_head(
    model: nn.Module,
    old_nc: int,
    new_nc: int,
    preserve_old: bool = True,
) -> None:
    """
    扩展 YOLO 检测头的分类输出层以支持更多类别。

    在类别增量学习中，每个任务会引入新的类别。YOLO 的检测头在初始化时
    固定了输出类别数（nc），需要在增量学习时扩展这个维度。

    实现方式：
      1. 创建新的 Conv2d 层，输出通道数扩展到 new_nc
      2. 将原始的 nc 个通道的权重/偏置复制到新层对应位置
      3. 新增的 (new_nc - old_nc) 个通道初始化为 0 或小随机值
      4. 替换模型中原有的检测头层

    注意：这只扩展了 cv3（分类分支）的最终输出层。
    cv2（box 分支）的输出维度只与 reg_max 相关，不随类别数变化。

    Args:
        model: YOLO 检测模型（nn.Module）。
        old_nc: 扩展前的类别数。
        new_nc: 扩展后的类别数。
        preserve_old: 是否保留旧类别的权重。如果为 True，旧类别权重保持不变；
                     如果为 False，尝试将旧权重复制到新位置。

    Raises:
        ValueError: 如果 old_nc >= new_nc（不需要扩展）。
        RuntimeError: 如果找不到检测头或 cv3 层。
    """
    if old_nc >= new_nc:
        raise ValueError(
            f"不需要扩展：old_nc={old_nc} >= new_nc={new_nc}。"
            "扩展只能在类别数增加时进行。"
        )

    _model = model.model.model
    detect=_model[-1]
    if not hasattr(detect, "cv3") or detect.cv3 is None:
        raise RuntimeError("模型检测头缺少 cv3 属性，无法扩展类别。")

    # 获取当前检测头的 reg_max（DFL 通道数）
    reg_max = detect.reg_max

    # 更新检测头的 nc 属性（这会影响推理时的后处理）
    detect.nc = new_nc
    detect.no = new_nc + reg_max * 4

    # 更新分类分支（cv3）的最终输出层
    for layer_idx in range(detect.nl):
        cv3_layer = detect.cv3[layer_idx]
        if not isinstance(cv3_layer, nn.Sequential):
            continue

        # 找到 cv3 最后一层（应该是 nn.Conv2d）
        old_conv = None
        old_bias = None
        old_in_channels = None

        for sub_module in cv3_layer:
            if isinstance(sub_module, nn.Conv2d) and sub_module.out_channels == old_nc:
                old_conv = sub_module
                old_bias = sub_module.bias
                old_in_channels = sub_module.in_channels
                break

        if old_conv is None:
            # 尝试更宽松的查找：找最后一个 Conv2d
            for sub_module in reversed(list(cv3_layer.modules())):
                if isinstance(sub_module, nn.Conv2d):
                    old_conv = sub_module
                    old_bias = sub_module.bias
                    old_in_channels = sub_module.in_channels
                    break

        if old_conv is None:
            continue

        # 创建新的 Conv2d 层
        new_conv = nn.Conv2d(
            old_in_channels,
            new_nc,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=(old_bias is not None),
        )

        with torch.no_grad():
            # 复制旧类别权重
            new_conv.weight[:old_nc, :, :, :] = old_conv.weight

            if old_bias is not None and new_conv.bias is not None:
                # bias 也是一维的，对应输出通道，直接覆盖前 old_nc 个
                new_conv.bias[:old_nc] = old_conv.bias

            # 新类别权重初始化为 0
            if preserve_old:
                nn.init.zeros_(new_conv.weight[old_nc:, :, :, :]) # 初始化剩余类别的输出权重
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias[old_nc:])

        # 替换最后一层
        for i, sub_module in enumerate(cv3_layer):
            if sub_module is old_conv:
                cv3_layer[i] = new_conv
                break

    # 更新 one2one 分支（如果有的话，用于端到端检测）
    if detect.end2end and hasattr(detect, "one2one_cv3"):
        _expand_one2one_cv3(detect, old_nc, new_nc, preserve_old)


def _expand_one2one_cv3(
    detect,
    old_nc: int,
    new_nc: int,
    preserve_old: bool = True,
) -> None:
    """
    扩展 one2one 分支的 cv3 层（用于端到端检测模式）。

    Args:
        detect: YOLO 检测头对象。
        old_nc: 扩展前的类别数。
        new_nc: 扩展后的类别数。
        preserve_old: 是否保留旧类别权重。
    """
    if not hasattr(detect, "one2one_cv3") or detect.one2one_cv3 is None:
        return

    for layer_idx in range(detect.nl):
        cv3_layer = detect.one2one_cv3[layer_idx]
        if not isinstance(cv3_layer, nn.Sequential):
            continue

        old_conv = None
        for sub_module in cv3_layer:
            if isinstance(sub_module, nn.Conv2d) and sub_module.out_channels == old_nc:
                old_conv = sub_module
                break

        if old_conv is None:
            continue

        new_conv = nn.Conv2d(
            old_conv.in_channels,
            new_nc,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            new_conv.weight[:old_nc, :, :, :] = old_conv.weight

            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias[:old_nc] = old_conv.bias

            if preserve_old:
                nn.init.zeros_(new_conv.weight[old_nc:, :, :, :])
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias[old_nc:])

        for i, sub_module in enumerate(cv3_layer):
            if sub_module is old_conv:
                cv3_layer[i] = new_conv
                break





