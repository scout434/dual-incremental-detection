# DuET 复现笔记

## 论文核心

DuET 面向 DuIOD 设置：模型需要同时处理类别增量和领域增量。论文不是简单微调检测器，而是训练不同任务模型后，用 task arithmetic 合并任务向量，并通过 direction consistency 降低新旧任务向量的冲突。

## 本复现实现的对应关系

| 论文模块 | 本项目实现 |
| --- | --- |
| Task arithmetic | `duet_repro/core/task_vectors.py` |
| Direction consistency | `duet_repro/core/task_vectors.py` 和 `duet_repro/core/losses.py` |
| Modified distillation | `duet_repro/core/losses.py` 中的 `duet_modified_distillation_loss` |
| Detector-agnostic | 使用 Ultralytics YOLO 权重的 state_dict 级合并 |
| RAI 指标 | `duet_repro/core/metrics.py` 和 `eval_rai.py` |
| Pascal/Diverse Weather 任务 | `configs/*.yaml` 和 `make_yolo_subsets.py` |

## 建议报告实验表

1. Sequential fine-tuning baseline。
2. Old task retention：每个阶段回测旧任务 mAP。
3. New task adaptation：每个阶段测试当前新任务 mAP。
4. DuET merge 后的 Avg RI、Avg GI、RAI。
5. 消融：不使用 direction consistency / 调整 alpha_old 与 alpha_new。

## 课程复现可接受简化

1. 先只用 YOLO11n，降低训练成本。
2. 先做 Pascal VOC 类增量，暂不做天气域。
3. 若显存不足，可将 `imgsz` 降到 416 或 512。
4. 若时间有限，可只复现 2-task 或 3-task 设置。
