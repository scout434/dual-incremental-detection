# Status2 DuET 复现说明

`status2` 对应论文里的 Multi Phase 多阶段场景，当前实现已经同步为
`status1` 成功复现的技术路线。

## 技术路线

任务顺序如下：

```text
T1: watercolor_1_3 -> 类别 [0, 1, 2]
T2: comic_4_6      -> 类别 [3, 4, 5]
T3: clipart_7_13   -> 类别 [6, 7, 8, 9, 10, 11, 12]
T4: voc_14_20      -> 类别 [13, 14, 15, 16, 17, 18, 19]
```

模型结构采用“累计单 Detect head”：

```text
T1: 3 类 head
T2: 6 类 head  = 旧 3 类行 + 新 3 类行
T3: 13 类 head = 旧 6 类行 + 新 7 类行
T4: 20 类 head = 旧 13 类行 + 新 7 类行
```

每个增量阶段的核心逻辑：

1. 先构建当前累计类别数的 YOLO Detect head。
2. 当前任务训练时使用上一阶段 merged checkpoint 作为 teacher。
3. `distill_weight: 0.01` 用于旧知识蒸馏。
4. `dc_weight: 0.01` 用于 DuET 方向一致性约束。
5. `shared_key_exclude: model.23`，所以 `model.23` 整个 Detect head 不参与共享层 task vector 融合。
6. backbone/neck 等共享层通过 DuET task vector 融合。
7. Detect head 合并时，保留旧模型完整 Detect head 的非分类输出结构，旧类分类行来自旧模型，新类分类行来自当前任务模型。
8. 最终模型仍然是一个标准 YOLO Detect head，不是外挂并联多头。

## 主训练

从 T1 到 T4 完整训练：

```powershell
python status2\train_duet.py --config status2\configs\train.yaml
```

输出目录：

```text
status2/output/main
```

主要模型命名：

```text
status2/output/main/task_1_watercolor_1_3_best.pt
status2/output/main/task_2_comic_4_6_duet.pt
status2/output/main/task_3_clipart_7_13_duet.pt
status2/output/main/task_4_voc_14_20_duet.pt
```

## 只跑 T2

如果已经完成 T1，可以直接从 T2 开始：

```powershell
python status2\train_duet.py --config status2\configs\train_t2_only.yaml
```

这个配置默认复用：

```text
status2/output/main/task_1_watercolor_1_3_best.pt
```

输出目录：

```text
status2/output/t2_only
```

## GI 分母参考模型

评估 Avg GI 之前，需要单独训练四个参考模型，作为 GI 的分母：

```powershell
python status2\train_duet.py --config status2\configs\ref_watercolor.yaml
python status2\train_duet.py --config status2\configs\ref_comic.yaml
python status2\train_duet.py --config status2\configs\ref_clipart.yaml
python status2\train_duet.py --config status2\configs\ref_voc.yaml
```

参考模型输出：

```text
status2/output/ref_watercolor/task_1_watercolor_4_6_best.pt
status2/output/ref_comic/task_1_comic_1_3_best.pt
status2/output/ref_clipart/task_1_clipart_1_6_best.pt
status2/output/ref_voc/task_1_voc_1_13_best.pt
```

这些模型只用于计算 GI 分母，不参与主训练。

## 评估指标

运行：

```powershell
python status2\eval_paper_metrics.py --plan status2\configs\eval.yaml
```

输出文件：

```text
status2/output/main/metrics.json
status2/output/main/rai_metrics.json
```

指标含义：

```text
Avg RI = 旧任务保持能力平均值
Avg GI = 未见切片泛化能力平均值
RAI    = (Avg RI + Avg GI) / 2
```

RI 计算项：

```text
RI_T4_Watercolor_1_3 = final / T1
RI_T4_Comic_4_6      = final / T2
RI_T4_Clipart_7_13   = final / T3
```

GI 计算项：

```text
T2:
  GI_T2_Watercolor_4_6
  GI_T2_Comic_1_3

T3:
  GI_T3_Watercolor_4_6
  GI_T3_Comic_1_3
  GI_T3_Clipart_1_6

T4:
  GI_T4_Watercolor_4_6
  GI_T4_Comic_1_3
  GI_T4_Clipart_1_6
  GI_T4_VOC_1_13
```

评估脚本会自动处理累计模型和参考模型的标签空间差异：

```text
累计模型：局部标签映射到全局类别通道后评估。
参考模型：保留局部标签评估，作为 GI denominator。
```

## 推荐运行顺序

```powershell
python status2\train_duet.py --config status2\configs\train.yaml

python status2\train_duet.py --config status2\configs\ref_watercolor.yaml
python status2\train_duet.py --config status2\configs\ref_comic.yaml
python status2\train_duet.py --config status2\configs\ref_clipart.yaml
python status2\train_duet.py --config status2\configs\ref_voc.yaml

python status2\eval_paper_metrics.py --plan status2\configs\eval.yaml
```
