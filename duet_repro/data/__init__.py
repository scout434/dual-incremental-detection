"""
DuET 数据处理模块

本模块提供 YOLO 数据集处理相关的工具：

  make_yolo_subsets.py:
    将完整的 YOLO 格式数据集按类别拆分为多个任务子集，
    支持增量学习实验的数据准备。

    典型用法：
      python -m duet_repro.data.make_yolo_subsets \
          --src-root /path/to/full_dataset \
          --dst-root /path/to/tasks \
          --tasks "0,1,2,3|4,5,6,7|8,9,10,11|12,13,14,15" \
          --remap

    生成的目录结构：
      dst_root/
      ├── task_1/
      │   ├── data.yaml
      │   ├── images/train/, images/val/
      │   └── labels/train/, labels/val/
      ├── task_2/
      ├── task_3/
      └── task_4/
"""
