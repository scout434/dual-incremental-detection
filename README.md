# DuET-CIOD 复现

> 论文：*Dual Incremental Object Detection via Exemplar-Free Task Arithmetic* (ICCV 2025)
>
> 本指南面向课程复现、测试和验收人员，用于从环境搭建、数据准备、模型训练到论文指标评估完整复现本项目。

---

## 一、技术路线说明

本项目复现的是 DuET 的核心思想：

```text
共享层：Backbone / Neck 使用 task vector 进行 DuET 融合
检测头：Detect head 作为任务特定模块，不参与共享层 task vector 融合
旧类保护：旧类分类行从上一阶段模型保留
新类学习：新类分类行来自当前阶段训练后的模型
损失函数：标准检测损失 + 蒸馏损失 + 方向一致性损失
```

关键配置：

```yaml
duet:
  enabled: true
  use_duet_module: true
  gamma: 0.1
  alpha_base: 0.5
  distill_weight: 0.01
  dc_weight: 0.01
  shared_key_exclude:
    - model.23
```

其中 `model.23` 是 YOLO11n 的 Detect head。排除它表示：

```text
Detect head 不参与 shared task vector 融合。
backbone/neck 参与 DuET Module 融合。
```

以两阶段任务为例：

```text
T1: 训练 10 类 head
T2: 扩展为 20 类累计 head
最终: 旧类行来自 T1，新类行来自 T2，backbone/neck 用 DuET 融合
```

---

## 二、交付文件清单（这个要在最后对准一下）

项目目录应包含以下主要内容：

```text
dual-incremental-detection-master/
├── README.md                         # 项目复现说明
├── yolo11n.pt                         # YOLO11n 预训练权重
├── yolo11n.yaml                       # YOLO11n 模型结构
├── ultralytics/                       # 本项目使用的 Ultralytics 源码
├── data_process/
│   └── prepare_data.py                # 三个场景统一数据预处理入口
├── duet_repro/
│   └── core/
│       ├── duet_module.py             # DuET task vector 融合模块
│       ├── duet_loss.py               # 蒸馏损失与方向一致性损失
│       ├── incremental_head.py        # 增量检测头相关实现
│       └── task_vectors.py            # task vector 计算与 checkpoint 工具
├── status1/
│   ├── train_duet.py                  # 场景1训练入口
│   ├── eval_paper_metrics.py          # 场景1论文指标评估
│   ├── run_ablation.py                # 场景1消融实验入口
│   ├── configs/                       # 场景1训练、评估、消融配置
│   └── output/                        # 场景1模型、日志、指标输出
├── status2/
│   ├── train_duet.py                  # 场景2训练入口
│   ├── eval_paper_metrics.py          # 场景2论文指标评估
│   ├── configs/                       # 场景2训练与评估配置
│   └── output/                        # 场景2模型、日志、指标输出
└── status3/
    ├── train_duet.py                  # 场景3训练入口
    ├── eval_paper_metrics.py          # 场景3论文指标评估
    ├── configs/                       # 场景3训练与评估配置
    └── output/                        # 场景3模型、日志、指标输出
```

**不随代码自动提供、需要自行准备的内容：**

- 各场景 YOLO 格式数据集。
- YOLO11n 预训练权重 `yolo11n.pt`。
- 如需直接评估，需准备已经训练好的 `status*/output/` 模型文件。

---

## 三、核心代码说明

| 文件 | 作用 |
| --- | --- |
| `duet_repro/core/task_vectors.py` | 计算 `theta_task - theta_reference`，并提供 state_dict 合并工具 |
| `duet_repro/core/duet_module.py` | 实现 DuET Module 的逐层动态融合 |
| `duet_repro/core/duet_loss.py` | 实现 `L_Distill` 和 `L_DC`，并接入 YOLO 检测损失 |
| `duet_repro/core/incremental_head.py` | 增量检测头与并联 head 辅助实现 |
| `status1/train_duet.py` | 场景1主训练脚本 |
| `status1/run_ablation.py` | 场景1消融实验脚本 |
| `status1/eval_paper_metrics.py` | 场景1论文指标计算 |
| `status2/train_duet.py` | 场景2多阶段训练脚本 |
| `status2/eval_paper_metrics.py` | 场景2论文指标计算 |
| `status3/train_duet.py` | 场景3天气数据训练脚本 |
| `status3/eval_paper_metrics.py` | 场景3论文指标计算 |

---

## 四、环境搭建

环境安装更为麻烦，我们提供了两种方式配置环境：conda一键配置（时间较久）、自行下载单包

### 4.1 创建并激活 conda 环境

推荐使用项目复现环境：

```powershell
conda env create -f environment.yml
conda activate duet-repro
```

### 4.2 单独下载

手动逐个安装 environment.yml 依赖

```
1. 创建并进入 conda 环境
conda create -n duet-repro python=3.10 pip -y
conda activate duet-repro

2. 安装 PyTorch + CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

3. 安装 conda 基础依赖
conda install numpy pyyaml tqdm pillow -c conda-forge -y

4. 安装 pip 依赖
pip install "ultralytics>=8.3.0"
pip install opencv-python pandas matplotlib rich

5. 验证 PyTorch / CUDA
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

6. 验证 Ultralytics
python -c "import ultralytics; print(ultralytics.__version__)"

可选：如果想固定到之前日志里的 Ultralytics 版本
pip install ultralytics==8.4.41
```

---

## 五、代码部署

### 5.1 放入预训练权重

### 5.2 放入训练好的权重

关于预训练和训练好的权重我们都放在了相应文件文件目录中，直接git clone即可

## 六、数据准备（没实验过，不完整）

所有数据集最终都整理为 YOLO 检测格式：

```text
dataset_root/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

标签格式为：

```text
class_id x_center y_center width height
```

本项目提供统一的数据预处理入口。正常情况下，不需要修改 Python 代码，直接使用已经写好的 YAML 配置即可运行：

```powershell
python data_process\prepare_data.py --plan data_process\configs\status1.yaml
python data_process\prepare_data.py --plan data_process\configs\status3.yaml
```

如果后期要换 zip 名、换数据目录、换任务划分，只需要修改对应的 YAML 文件：

```text
data_process/configs/status1.yaml
data_process/configs/status3.yaml
```

### 6.1 默认 zip 放置位置

默认配置下，场景1和场景3的 zip 放置位置如下：

```text
status1: data/downloads/
status3: data/downloads/
```

如果你的 zip 文件名或目录不同，直接修改对应 `status*.yaml` 中的 `zip_root`、`voc_zips`、`clipart_zip` 或 `*_zips` 里的 `zip` 即可。

预处理完成后，脚本会生成对应任务的 `data.yaml`，并根据 `update_configs` 自动更新训练和评估配置中的数据路径。

### 6.2 数据处理 YAML 主要字段说明

`prepare_data.py` 会完全按照 `--plan` 指定的 YAML 执行。YAML 中主要字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `kind` | 数据处理类型。`voc_clipart` 表示场景1的 VOC/Clipart 原始数据处理；`voc_zip_slices` 表示场景3的 VOC XML 格式 zip 切片处理。 |
| `zip_root` | 原始 zip 文件所在目录。脚本会从这里读取压缩包。 |
| `data_root` | 处理后数据集的输出根目录。场景1写入 `data/status1`，场景3写入 `data/status3`；场景3的解压缓存会放在 `data_root/raw` 下。 |
| `update_configs` | 预处理完成后，需要自动更新 `data:` 路径的训练/评估配置文件。 |
| `voc_zips` | 场景1使用，VOC 官方 zip 文件名列表。 |
| `clipart_zip` | 场景1使用，Clipart zip 文件名。 |
| `voc_tasks` | 场景1使用，定义要生成哪些 VOC 子任务数据集。 |
| `clipart_tasks` | 场景1使用，定义要生成哪些 Clipart 子任务数据集。 |
| `class_names` | 场景3使用，定义 VOC XML 中类别名到全局类别编号的顺序；当前为 `bike, bus, car, motor, person, rider, truck`，对齐论文表 S6。 |
| `daytime_sunny_zips` / `night_sunny_zips` | 场景3使用，定义 Daytime Sunny 和 Night Sunny 原始 zip 文件名。当前分别指向 `Daytime_Sunny.zip` 和 `Night-Sunny.zip`。 |
| `daytime_sunny_tasks` / `night_sunny_tasks` | 场景3使用，定义要从对应天气域里切出的任务数据集。 |
| `name` | 任务名称，同时也是默认输出目录名。训练配置中的任务名需要和它对应。 |
| `zip` | 当前任务读取的 zip 文件名，写在 `*_zips` 下面。 |
| `class_indices` | 当前任务对应的全局类别编号。增量训练时用它对齐最终检测头中的类别通道。 |
| `train_split` / `val_split` | 指定当前任务使用哪个 split 作为训练集和验证集。 |
| `train_splits` | 场景1 VOC 使用，表示多个 VOC split 合并为训练集。 |

例如场景3中的一个任务可以写成：

```yaml
night_sunny_zips:
  night_sunny_5_7:
    zip: Night-Sunny.zip

night_sunny_tasks:
  - name: night_sunny_5_7
    class_indices: [4, 5, 6]
    train_split: train
    val_split: test
```

这表示脚本会从 `zip_root` 下读取 `Night-Sunny.zip`，生成 `night_sunny_5_7` 数据集，并只保留全局类别 `[4, 5, 6]`，也就是论文中的 `person, rider, truck`。

## 七、路径修改

本项目的路径主要在各场景 `configs/*.yaml` 中修改。

### 7.1 数据路径

例如：

```yaml
tasks:
  - name: daytime_sunny_1_4
    data: /your/path/to/daytime_sunny_1_4/data.yaml
    class_indices: [0, 1, 2, 3]
    labels_are_global: false
```

如果数据标签是局部编号，例如当前任务只有 3 类，标签为 `0,1,2`，则写：

```yaml
labels_are_global: false
```

代码会根据 `class_indices` 自动把局部标签映射到累计 head 的通道位置。

### 7.2 输出路径

主实验输出目录统一在：

```text
status1/output/main
status2/output/main
status3/output/main
```

参考模型输出目录：

```text
status1/output/ref_*
status2/output/ref_*
status3/output/ref_*
```

### 7.3 GPU 设备

配置文件中：

```yaml
training:
  device: 0
```

如果服务器只暴露一张卡，或者使用了：

```bash
CUDA_VISIBLE_DEVICES=1
```

程序内部通常仍应写：

```yaml
device: 0
```

否则容易出现：

```text
CUDA error: invalid device ordinal
```

---

## 八、训练全流程

### 8.1 场景1主实验

论文对应地址和训练流程：

```text
主文 Table 2：VOC[1:10] -> Clipart[11:20] 两阶段 DuIOD 结果
补充材料 Table S7：VOC[1:10] -> Clipart[11:20] 不同 base detector 的详细结果

主实验：VOC[1:10] 训练 T1 -> Clipart[11:20] 训练 T2 -> DuET 融合共享层 -> 输出最终 T2 模型
T2-only：复用已经存在的 T1 checkpoint，跳过 T1，只训练 Clipart[11:20]
参考模型：单独训练 VOC[11:20] 和 Clipart[1:10]，只用于计算 Avg GI 分母
```

完整训练：

```powershell
python status1\train_duet.py --config status1\configs\train.yaml
```

如果已经训练好 T1，只从 T2 开始：

```powershell
python status1\train_duet.py --config status1\configs\train_t2_only.yaml
```

训练 GI 分母参考模型：

```powershell
python status1\train_duet.py --config status1\configs\ref_voc.yaml
python status1\train_duet.py --config status1\configs\ref_clipart.yaml
```

主要输出：

```text
status1/output/main/task_1_voc_1_10_best.pt
status1/output/main/task_2_clipart_11_20_duet.pt

status1/output/ref_clipart/task_1_clipart_1_10_best.pt
status1/output/ref_voc/task_1_voc_11_20_best.pt
```

### 8.2 场景2主实验

对应论文表格：

```text
主文 Table 2：Watercolor[1:3] -> Comic[4:6] -> Clipart[7:13] -> VOC[14:20] 多阶段 DuIOD 结果
补充材料 Table S12：该多阶段实验在不同 base detector 上的详细结果
补充材料 Table S11：该多阶段实验的计算复杂度对比
```

训练流程：

```text
主实验：Watercolor[1:3] -> Comic[4:6] -> Clipart[7:13] -> VOC[14:20]
每个增量阶段：先从上一阶段模型扩展 head，再训练当前任务，最后用 DuET 融合共享层并保留累计 head
T2-only：复用已经存在的 Watercolor[1:3] checkpoint，从 Comic[4:6] 开始继续跑后续任务
参考模型：分别训练 Watercolor[4:6]、Comic[1:3]、Clipart[1:6]、VOC[1:13]，用于计算 Avg GI 分母
```

完整训练：

```powershell
python status2\train_duet.py --config status2\configs\train.yaml
```

如果已经训练好 T1，只从 T2 开始：

```powershell
python status2\train_duet.py --config status2\configs\train_t2_only.yaml
```

训练 GI 分母参考模型：

```powershell
python status2\train_duet.py --config status2\configs\ref_watercolor.yaml
python status2\train_duet.py --config status2\configs\ref_comic.yaml
python status2\train_duet.py --config status2\configs\ref_clipart.yaml
python status2\train_duet.py --config status2\configs\ref_voc.yaml
```

主要输出：

```text
status2/output/main/task_1_watercolor_1_3_best.pt
status2/output/main/task_2_comic_4_6_duet.pt
status2/output/main/task_3_clipart_7_13_duet.pt
status2/output/main/task_4_voc_14_20_duet.pt
```

### 8.3 场景3主实验

对应论文表格 和 训练流程：

```text
主文 Table 1：Daytime Sunny[1:4] -> Night Sunny[5:7] 详细结果
主文 Table 3：DuET 在该场景下不同 base detector 的汇总结果
补充材料 Table S9：Daytime Sunny[1:4] -> Night Sunny[5:7] 不同 base detector 的详细结果

主实验：Daytime Sunny[1:4] 训练 T1 -> Night Sunny[5:7] 训练 T2 -> DuET 融合共享层 -> 输出最终 T2 模型
T2-only：复用已经存在的 Daytime Sunny[1:4] checkpoint，跳过 T1，只训练 Night Sunny[5:7]
参考模型：单独训练 Daytime Sunny[5:7] 和 Night Sunny[1:4]，只用于计算 Avg GI 分母
```

完整训练：

```powershell
python status3\train_duet.py --config status3\configs\train.yaml
```

如果已经训练好 T1，只从 T2 开始：

```powershell
python status3\train_duet.py --config status3\configs\train_t2_only.yaml --output-dir status3/output/main
```

训练 GI 分母参考模型：

```powershell
python status3\train_duet.py --config status3\configs\ref_daytime.yaml
python status3\train_duet.py --config status3\configs\ref_night.yaml
```

主要输出：

```text
status3/output/main/task_1_daytime_sunny_1_4_best.pt
status3/output/main/task_2_night_sunny_5_7_duet.pt

status3/output/ref_daytime/task_1_daytime_sunny_5_7_best.pt
status3/output/ref_night/task_1_night_sunny_1_4_best.pt
```

---

## 九、消融实验

对应论文表格   和  实验流程：

```text
主文 Table 4：YOLO11n 上 DuET 不同组件和损失项的消融实验

消融实验位于 `status1`，只针对场景1：VOC[1:10] -> Clipart[11:20]
```

消融组件包括：

```text
Seq FT
Incremental Head
DuET Module
L_Distill
L_DC
```

生成所有消融配置：

```powershell
python status1\run_ablation.py --all --materialize-only
```

运行完整方法：

```powershell
python status1\run_ablation.py --name 05_full
python status1\eval_paper_metrics.py --plan status1\configs\ablations\05_full_eval.yaml
```

单独运行每组消融：

```powershell
python status1\run_ablation.py --name 00_no_seqft
python status1\run_ablation.py --name 01_seqft
python status1\run_ablation.py --name 02_seqft_incremental_head
python status1\run_ablation.py --name 03_seqft_incremental_head_duet
python status1\run_ablation.py --name 04_seqft_incremental_head_duet_distill
python status1\run_ablation.py --name 05_full
```

评估对应消融：

```powershell
python status1\eval_paper_metrics.py --plan status1\configs\ablations\00_no_seqft_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\01_seqft_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\02_seqft_incremental_head_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\03_seqft_incremental_head_duet_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\04_seqft_incremental_head_duet_distill_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\05_full_eval.yaml
```

---

## 十、评估全流程

### 10.1 场景1评估

```powershell
python status1\eval_paper_metrics.py --plan status1\configs\eval.yaml
```

### 10.2 场景2评估

```powershell
python status2\eval_paper_metrics.py --plan status2\configs\eval.yaml
```

### 10.3 场景3评估

```powershell
python status3\eval_paper_metrics.py --plan status3\configs\eval.yaml
```

评估输出：

```text
metrics.json
rai_metrics.json
```

指标含义：

```text
Avg RI = 旧任务保持能力
Avg GI = 未见切片泛化能力
RAI    = (Avg RI + Avg GI) / 2
```

---

## 十一、提供给测试小组的执行命令

本节给测试小组直接照抄执行。默认测试人员已经进入项目根目录：

```powershell
cd E:\project\test\dual-incremental-detection-master\dual-incremental-detection-master
conda activate incremental_learning
```

如果测试环境的 conda 环境名不同，请先切换到已经安装 `torch`、`ultralytics`、`yaml` 等依赖的环境。

### 11.1 快速验收：只复算已有模型指标

如果交付包中已经包含 `status*/output/` 下的训练权重，可以跳过训练，直接执行三组评估命令：

```powershell
python status1\eval_paper_metrics.py --plan status1\configs\eval.yaml
python status2\eval_paper_metrics.py --plan status2\configs\eval.yaml
python status3\eval_paper_metrics.py --plan status3\configs\eval.yaml
```

评估完成后检查以下文件是否生成或更新：

```text
status1/output/main/metrics.json
status1/output/main/rai_metrics.json
status2/output/main/metrics.json
status2/output/main/rai_metrics.json
status3/output/main/metrics.json
status3/output/main/rai_metrics.json
```

验收时主要看 `metrics.json` 中的三个字段：

```text
avg_ri_percent
avg_gi_percent
rai_percent
```

当前工作区已经记录的主实验结果见本文第十四节。

### 11.2 场景1：VOC[1:10] -> Clipart[11:20]

如果需要从训练开始完整复现，依次执行：

```powershell
python status1\train_duet.py --config status1\configs\train.yaml
python status1\train_duet.py --config status1\configs\ref_voc.yaml
python status1\train_duet.py --config status1\configs\ref_clipart.yaml
python status1\eval_paper_metrics.py --plan status1\configs\eval.yaml
```

如果已经有主模型和参考模型权重，只需要执行最后一条评估命令：

```powershell
python status1\eval_paper_metrics.py --plan status1\configs\eval.yaml
```

关键输出：

```text
status1/output/main/task_1_voc_1_10_best.pt
status1/output/main/task_2_clipart_11_20_duet.pt
status1/output/ref_voc/task_1_voc_11_20_best.pt
status1/output/ref_clipart/task_1_clipart_1_10_best.pt
status1/output/main/metrics.json
```

### 11.3 场景2：Watercolor -> Comic -> Clipart -> VOC

如果需要从训练开始完整复现，依次执行：

```powershell
python status2\train_duet.py --config status2\configs\train.yaml
python status2\train_duet.py --config status2\configs\ref_watercolor.yaml
python status2\train_duet.py --config status2\configs\ref_comic.yaml
python status2\train_duet.py --config status2\configs\ref_clipart.yaml
python status2\train_duet.py --config status2\configs\ref_voc.yaml
python status2\eval_paper_metrics.py --plan status2\configs\eval.yaml
```

如果已经有主模型和参考模型权重，只需要执行最后一条评估命令：

```powershell
python status2\eval_paper_metrics.py --plan status2\configs\eval.yaml
```

关键输出：

```text
status2/output/main/task_1_watercolor_1_3_best.pt
status2/output/main/task_2_comic_4_6_duet.pt
status2/output/main/task_3_clipart_7_13_duet.pt
status2/output/main/task_4_voc_14_20_duet.pt
status2/output/main/metrics.json
```

### 11.4 场景3：Daytime Sunny[1:4] -> Night Sunny[5:7]

如果需要从训练开始完整复现，依次执行：

```powershell
python status3\train_duet.py --config status3\configs\train.yaml
python status3\train_duet.py --config status3\configs\ref_daytime.yaml
python status3\train_duet.py --config status3\configs\ref_night.yaml
python status3\eval_paper_metrics.py --plan status3\configs\eval.yaml
```

如果已经有主模型和参考模型权重，只需要执行最后一条评估命令：

```powershell
python status3\eval_paper_metrics.py --plan status3\configs\eval.yaml
```

关键输出：

```text
status3/output/main/task_1_daytime_sunny_1_4_best.pt
status3/output/main/task_2_night_sunny_5_7_duet.pt
status3/output/ref_daytime/task_1_daytime_sunny_5_7_best.pt
status3/output/ref_night/task_1_night_sunny_1_4_best.pt
status3/output/main/metrics.json
```

### 11.5 消融实验：场景1 Table 4 对应流程

消融实验只在 `status1` 场景下执行。若需要先生成所有消融配置：

```powershell
python status1\run_ablation.py --all --materialize-only
```

单独运行各组消融训练：

```powershell
python status1\run_ablation.py --name 00_no_seqft
python status1\run_ablation.py --name 01_seqft
python status1\run_ablation.py --name 02_seqft_incremental_head
python status1\run_ablation.py --name 03_seqft_incremental_head_duet
python status1\run_ablation.py --name 04_seqft_incremental_head_duet_distill
python status1\run_ablation.py --name 05_full
```

如果消融权重已经存在，可以直接执行评估：

```powershell
python status1\eval_paper_metrics.py --plan status1\configs\ablations\00_no_seqft_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\01_seqft_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\02_seqft_incremental_head_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\03_seqft_incremental_head_duet_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\04_seqft_incremental_head_duet_distill_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\05_full_eval.yaml
```

每组消融的评估结果位于：

```text
status1/output/ablations/<ablation_name>/metrics.json
status1/output/ablations/<ablation_name>/rai_metrics.json
```

### 11.6 测试记录要求

测试小组验收时建议记录以下内容：

| 项目 | 需要记录 |
| --- | --- |
| 环境 | Python、PyTorch、Ultralytics、CUDA 是否可用 |
| 数据 | `data/status1`、`data/status2`、`data/status3` 是否存在 |
| 权重 | `status*/output/main` 和 `status*/output/ref_*` 是否存在关键 `.pt` 文件 |
| 主实验 | 三个场景的 `avg_ri_percent`、`avg_gi_percent`、`rai_percent` |
| 消融实验 | `01_seqft` 到 `05_full` 的 Avg RI、Avg GI、RAI |
| 日志 | 训练日志位于 `status*/output/**/logs/`，评估结果位于 `metrics.json` |

如果评估报错找不到 checkpoint，优先检查对应 `eval.yaml` 中的 `checkpoint_aliases` 和 `status*/output/*/eval_manifest.json`。

---

## 十二、输出文件说明

一次完整训练结束后，主输出目录通常包含：

```text
reference_full_head.pt
resolved_config.yaml
training_history.json
eval_manifest.json
task_*_best.pt
task_*_duet.pt
logs/train_*.txt
runs/<task_name>/weights/best.pt
```

其中：

| 文件 | 作用 |
| --- | --- |
| `reference_full_head.pt` | task vector 的 reference 模型 |
| `task_1_*_best.pt` | T1 训练完成模型 |
| `task_*_duet.pt` | 增量阶段 DuET 融合后的最终模型 |
| `training_history.json` | 每个任务的训练、初始化、合并记录 |
| `eval_manifest.json` | 评估脚本解析 checkpoint 的依据 |
| `metrics.json` | 详细 RI/GI/RAI 指标 |
| `rai_metrics.json` | 简化版论文指标 |

如果 `eval_manifest.json` 出现：

```json
"latest_checkpoint": null,
"history": []
```

说明该输出目录不是完整训练结果，通常是训练中断或新旧实验文件混在一起。

---

## 十三、关键注意事项

1. **不要混用输出目录。**  
   每次完整实验建议使用干净的 `status*/output/main`，否则 `eval_manifest.json` 可能指向旧模型。

2. **T2-only 训练要确认输出目录。**  
   如果希望续跑结果仍写入主目录，应使用：

   ```powershell
   python status3\train_duet.py --config status3\configs\train_t2_only.yaml --output-dir status3/output/main
   ```

3. **确认 `class_indices` 与标签空间一致。**  
   局部标签使用 `labels_are_global: false`；全局标签使用 `labels_are_global: true`。

4. **确认 GPU 编号。**  
   单卡环境一般写 `device: 0`。

5. **评估必须看 `eval_manifest.json`。**  
   `checkpoint: final` 实际会解析到 manifest 中的 `latest_checkpoint`。

6. **GI 分母模型必须单独训练。**  
   `ref_*` 模型只用于计算 Avg GI denominator，不参与主训练。

---

## 十四、实验结果记录

### 14.1 主实验汇总

| 实验 | 训练顺序 | Avg RI (%) | Avg GI (%) | RAI (%) |
| --- | --- | ---: | ---: | ---: |
| status1 | VOC[1:10] -> Clipart[11:20] | 86.88 | 44.33 | 65.61 |
| status2 | Watercolor[1:3] -> Comic[4:6] -> Clipart[7:13] -> VOC[14:20] | 46.08 | 110.91 | 78.50 |
| status3 | Daytime Sunny[1:4] -> Night Sunny[5:7] | 54.61 | 26.60 | 40.61 |

### 14.2 status1 过程性指标

| 指标 | 分子 mAP50 | 分母 mAP50 | 比例 (%) |
| --- | ---: | ---: | ---: |
| RI_VOC_1_10 | 0.7200 | 0.8288 | 86.88 |
| GI_VOC_11_20 | 0.1415 | 0.7968 | 17.76 |
| GI_Clipart_1_10 | 0.2795 | 0.3942 | 70.91 |

### 14.3 status2 过程性指标

| 指标 | 分子 mAP50 | 分母 mAP50 | 比例 (%) |
| --- | ---: | ---: | ---: |
| RI_T4_VOC_1_13 | 0.5751 | 0.7544 | 76.23 |
| RI_T4_VOC_14_20 | 0.0272 | 0.3423 | 7.94 |
| RI_T4_Clipart_1_6 | 0.3825 | 0.3876 | 98.68 |
| RI_T4_Clipart_7_13 | 0.0077 | 0.5246 | 1.47 |
| GI_T2_Watercolor_4_6 | 0.3326 | 0.3377 | 98.47 |
| GI_T2_Comic_1_3 | 0.2157 | 0.2138 | 100.90 |
| GI_T2_Clipart_1_6 | 0.2354 | 0.2637 | 89.28 |
| GI_T4_Watercolor_4_6 | 0.2855 | 0.3377 | 84.52 |
| GI_T4_Comic_1_3 | 0.3890 | 0.2138 | 181.94 |
| GI_T4_Clipart_1_6 | 0.3825 | 0.2637 | 145.05 |
| GI_T4_VOC_1_13 | 0.5751 | 0.7544 | 76.23 |

### 14.4 status3 过程性指标

| 指标 | 分子 mAP50 | 分母 mAP50 | 比例 (%) |
| --- | ---: | ---: | ---: |
| RI_DaytimeSunny_1_4_after_T2 | 0.2555 | 0.4678 | 54.61 |
| GI_DaytimeSunny_5_7_at_T2 | 0.0478 | 0.5140 | 9.30 |
| GI_NightSunny_1_4_at_T2 | 0.2036 | 0.4639 | 43.89 |

### 14.5 消融实验记录

当前工作区中 `01_seqft` 到 `05_full` 均已有 `metrics.json`；`00_no_seqft` 目录存在，但未发现对应评估指标文件，因此暂记为未评估。

| 消融组 | 组件设置 | Avg RI (%) | Avg GI (%) | RAI (%) |
| --- | --- | ---: | ---: | ---: |
| 00_no_seqft | 无 Seq FT / 无 Incremental Head / 无 DuET / 无损失项 | 未评估 | 未评估 | 未评估 |
| 01_seqft | Seq FT only | 0.12 | 13.22 | 6.67 |
| 02_seqft_incremental_head | Seq FT + Incremental Head | 21.02 | 22.68 | 21.85 |
| 03_seqft_incremental_head_duet | Seq FT + Incremental Head + DuET Module | 69.34 | 29.83 | 49.58 |
| 04_seqft_incremental_head_duet_distill | Seq FT + Incremental Head + DuET Module + L_Distill | 88.62 | 41.69 | 65.15 |
| 05_full | Seq FT + Incremental Head + DuET Module + L_Distill + L_DC | 89.23 | 43.38 | 66.31 |

---

## Reference

DuET: *Dual Incremental Object Detection via Exemplar-Free Task Arithmetic*, ICCV 2025.

Ultralytics YOLO: https://github.com/ultralytics/ultralytics
