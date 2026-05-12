# DuET Reproduction: Dual Incremental Object Detection

本项目用于复现 ICCV 2025 论文 **Dual Incremental Object Detection via Exemplar-Free Task Arithmetic** 的核心实验流程。

实现目标：

- 使用 PyTorch + Ultralytics YOLO 作为检测器底座。
- 支持类增量、域增量和双重增量目标检测的实验组织。
- 实现 DuET 的任务向量合并、方向一致性损失、动态蒸馏损失和 RAI 指标。
- 提供 Pascal Series / Diverse Weather Series 风格的数据划分脚本。

> 说明：论文官方实现未随论文公开。本项目是课程复现版本，重点复现论文核心思想和可验证实验流程，而不是逐行复刻作者私有代码。

## 1. 环境

推荐使用 conda：

```powershell
conda env create -f environment.yml
conda activate duet-repro
```

如果已有 PyTorch 环境：

```powershell
pip install -r requirements.txt
```

## 2. 项目结构

```text
configs/                         实验配置
duet_repro/                      DuET 核心代码
  core/task_vectors.py           任务向量计算与模型合并
  core/losses.py                 DC loss 与动态蒸馏损失
  core/metrics.py                RI/GI/RAI 指标
  data/make_yolo_subsets.py      YOLO 数据集过滤与任务划分
scripts/                         运行脚本
train_ultralytics_duet.py        增量训练与 DuET 合并入口
eval_rai.py                      计算 RAI 指标
```

## 3. 数据准备

论文中 Pascal Series 使用 Pascal VOC 与若干风格域数据，Diverse Weather Series 使用不同天气域数据。

为了课程复现，建议先从 Pascal VOC 风格实验开始：

1. 准备 YOLO 格式数据集。
2. 每个图像对应一个 `.txt` 标签文件。
3. 数据集目录示例：

```text
datasets/
  pascal_yolo/
    images/train/
    images/val/
    labels/train/
    labels/val/
```

使用过滤脚本生成每个增量任务的数据：

```powershell
python -m duet_repro.data.make_yolo_subsets `
  --src-root E:\datasets\pascal_yolo `
  --dst-root E:\datasets\duet_pascal_tasks `
  --splits train val `
  --tasks "0,1,2,3,4|5,6,7,8,9|10,11,12,13,14|15,16,17,18,19" `
  --copy-images
```

## 4. 训练

先修改 `configs/pascal_series_yolo.yaml` 中的数据路径，然后运行：

```powershell
python train_ultralytics_duet.py --config configs/pascal_series_yolo.yaml
```

脚本会按任务顺序：

1. 训练当前任务模型。
2. 构造旧任务向量和新任务向量。
3. 对共享参数执行 DuET merge。
4. 保存每个阶段的 merged checkpoint。

## 5. 评估 RAI

收集每个任务阶段在旧类/旧域和新类/新域上的 mAP 后，写入 JSON：

```json
{
  "retention": [0.91, 0.88, 0.86],
  "generalization": [0.72, 0.76, 0.79]
}
```

运行：

```powershell
python eval_rai.py --metrics outputs/pascal_metrics.json
```

## 6. 推荐复现路线

课程时间有限时，建议按下面路线推进：

1. 先完成 Pascal VOC 的 4-task 类增量实验。
2. 用 YOLO11n 或 YOLO11s 降低算力需求。
3. 复现 baseline sequential fine-tuning。
4. 加入 DuET task-vector merge。
5. 对比 Avg RI、Avg GI、RAI。
6. 如有余力，再加入 Diverse Weather 或跨域任务。

