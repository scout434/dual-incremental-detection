"""
DuET 核心算法模块

本模块包含 DuET 双增量目标检测方法的核心算法实现：

  task_vectors.py:
    - task_vector(): 计算任务向量（模型参数变化量）
    - cosine_direction_loss(): 方向一致性损失（评估新旧任务向量的冲突程度）
    - incremental_direction_loss(): 论文 Eq (16) 的正确 LDC 实现
    - merge_state_dicts(): 核心合并函数，将多个任务向量融合到基准模型中
    - inject_state_dict_into_checkpoint(): 将合并后的状态字典写入检查点文件
    - load_state_dict(): 便捷的状态字典加载函数

  duet_module.py:
    - compute_p_factor(): 计算论文 Eq (9) 中的 p-factor
    - compute_layer_coefficients(): 计算论文 Eq (10-11) 中的层系数 αl 和 βl
    - merge_state_dicts_with_duet_module(): DuET Module 的逐层动态融合

  incremental_head.py:
    - expand_detect_head(): 扩展 YOLO 检测头以支持新类别数
    - concat_incremental_head_params(): 拼接增量检测头参数（论文 Eq 13）
    - expand_and_concat_detect_head(): 完整的 Incremental Head 流程

  duet_loss.py:
    - DuETDetectionLoss: 集成检测损失 + 蒸馏损失 + LDC 的完整损失类
    - create_duet_criterion(): DuET 损失工厂函数

  losses.py:
    - direction_consistency_loss(): 方向一致性损失（用于训练过程）
    - dynamic_distillation_loss(): 标准 KL 蒸馏损失
    - duet_modified_distillation_loss(): DuET 改进的蒸馏损失（带动态过滤）
    - feature_l2_distillation(): 特征级 L2 蒸馏损失
    - FeatureHook: 特征图钩子类（用于中间层特征提取）

  metrics.py:
    - retention_index(): 计算单个任务的保留指数 (RI)
    - generalization_index(): 计算单个任务的泛化指数 (GI)
    - rai_from_indices(): 计算综合 RAI 指标
    - RaiResult: RAI 结果数据结构

数学公式参考（DuET 论文）：
  任务向量定义：
    τ = θ_task - θ_reference

  DuET Module（逐层动态融合）：
    pl = (||τ_old|| - ||τ_curr||) / (||τ_old + τ_curr|| + ε)
    δl = γ * tanh(pl)
    αl = αbase + clamp(δl, -γ, γ),  βl = 1 - αl
    θ_merged[l] = θ_ref[l] + αl * τ_old[l] + βl * τ_curr[l]

  方向一致性损失（Eq 16）：
    LDC = Σ ReLU( - [(τ_st - τ_st-1) · (τ_st-1 - τ_st-2)] )

  Incremental Head（Eq 13）：
    (θτt)_incre ← [θτt; (θτt-1)_incre]

  RAI 指标：
    RI = min(current_mAP / reference_mAP, 1.0)
    GI = min(current_mAP / reference_mAP, 1.0)
    RAI = (mean(RI) + mean(GI)) / 2
"""
