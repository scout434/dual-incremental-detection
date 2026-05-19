"""
DuET 双增量目标检测课程复现工具包

该包实现了 ICCV 2025 论文 "Dual Incremental Object Detection via Exemplar-Free Task Arithmetic" (DuET)
的 PyTorch + Ultralytics YOLO 复现代码。

模块结构：
  - duet_repro.core: 核心算法实现
    - task_vectors: 任务向量计算与合并（DuET 核心）
    - losses: 各种知识保持损失函数
    - metrics: RAI 评估指标计算
  - duet_repro.data: 数据处理工具
    - make_yolo_subsets: YOLO 数据集任务划分脚本

使用方法：
  # 训练
  from duet_repro.core.task_vectors import merge_state_dicts
  merged, report = merge_state_dicts(reference, old_state, new_state)

  # 评估
  from duet_repro.core.metrics import rai_from_indices
  result = rai_from_indices(retention=[0.85, 0.78], generalization=[0.92, 0.88])
  print(f"RAI: {result.rai:.4f}")
"""

__version__ = "0.1.0"
