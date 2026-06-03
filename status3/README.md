# Status3 DuET 复现说明

`status3` 对应论文里的 Diverse Weather Two Phase 场景，当前训练实现和
`status1` 成功路线保持一致。

## 技术路线

任务顺序如下：

```text
T1: daytime_sunny_1_4 -> 类别 [0, 1, 2, 3]
T2: night_sunny_5_7   -> 类别 [4, 5, 6]
```

模型结构采用“累计单 Detect head”：

```text
T1: 4 类 head
T2: 7 类 head = 旧 4 类行 + 新 3 类行
```

每个增量阶段的核心逻辑：

1. 先构建当前累计类别数的 YOLO Detect head。
2. T2 训练时使用 T1 merged checkpoint 作为 teacher。
3. `distill_weight: 0.01` 用于旧知识蒸馏。
4. `dc_weight: 0.01` 用于 DuET 方向一致性约束。
5. `shared_key_exclude: model.23`，所以整个 Detect head 不参与共享层 task vector 融合。
6. backbone/neck 等共享层通过 DuET task vector 融合。
7. Detect head 合并时保留旧模型完整 Detect head 的非分类输出结构，旧类分类行来自旧模型，新类分类行来自当前任务模型。
8. 最终模型仍然是一个标准 YOLO Detect head，不是外挂并联多头。

## 主训练

```powershell
python status3\train_duet.py --config status3\configs\train.yaml
```

输出目录：

```text
status3/output/main
```

主要模型命名：

```text
status3/output/main/task_1_daytime_sunny_1_4_best.pt
status3/output/main/task_2_night_sunny_5_7_duet.pt
```

## 只跑 T2

如果已经完成 T1，可以直接从 T2 开始：

```powershell
python status3\train_duet.py --config status3\configs\train_t2_only.yaml
```

这个配置默认复用：

```text
status3/output/main/task_1_daytime_sunny_1_4_best.pt
```

输出目录：

```text
status3/output/t2_only
```

## GI 分母参考模型

评估 Avg GI 前，需要单独训练两个参考模型：

```powershell
python status3\train_duet.py --config status3\configs\ref_daytime.yaml
python status3\train_duet.py --config status3\configs\ref_night.yaml
```

参考模型输出：

```text
status3/output/ref_daytime/task_1_daytime_sunny_5_7_best.pt
status3/output/ref_night/task_1_night_sunny_1_4_best.pt
```

这些模型只用于计算 GI 分母，不参与主训练。

## 评估指标

```powershell
python status3\eval_paper_metrics.py --plan status3\configs\eval.yaml
```

输出文件：

```text
status3/output/main/metrics.json
status3/output/main/rai_metrics.json
```

指标含义：

```text
Avg RI = 旧任务保持能力平均值
Avg GI = 未见切片泛化能力平均值
RAI    = (Avg RI + Avg GI) / 2
```

RI 计算项：

```text
RI_DaytimeSunny_1_4_after_T2 = final / T1
```

GI 计算项：

```text
GI_DaytimeSunny_5_7_at_T2
GI_NightSunny_1_4_at_T2
```

评估时，累计模型会按 `global_class_indices` 映射到全局类别通道；参考模型分母保持局部标签空间。

## 推荐运行顺序

```powershell
python status3\train_duet.py --config status3\configs\train.yaml

python status3\train_duet.py --config status3\configs\ref_daytime.yaml
python status3\train_duet.py --config status3\configs\ref_night.yaml

python status3\eval_paper_metrics.py --plan status3\configs\eval.yaml
```
