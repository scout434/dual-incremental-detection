# DuET Dual Incremental Object Detection Reproduction

本项目用于复现 ICCV 2025 论文 **Dual Incremental Object Detection via Exemplar-Free Task Arithmetic** 的核心实验流程。

当前代码已整理为更接近深度学习算法库的结构：旧版 `status1/`、`status3/` 只保留配置和说明，实际训练、评估入口统一放在 `tools/`，旧脚本保存在 `legacy/` 以保证已有逻辑可追溯。

## 1. 环境

推荐使用 conda：

```powershell
conda env create -f environment.yml
conda activate duet-repro
```

如果已经有合适的 PyTorch 环境：

```powershell
pip install -r requirements.txt
```

## 2. 当前目录结构

```text
data/                 数据目录，不参与代码整理
data_process/         数据预处理相关内容
docs/                 论文、记录或补充文档
duet_repro/           复现项目公共代码
experiments/          标准化实验配置目录
legacy/               从 status1/status3 移出的旧训练与评估脚本
output/               统一实验输出目录
status1/              status1 配置和说明
status3/              status3 配置和说明
tools/                统一命令行入口
```

根目录只保留项目级文件、环境文件和基础权重：

```text
README.md
environment.yml
requirements.txt
yolo11n.pt
```

## 3. 统一调用方式

### Status1 主实验

```powershell
python tools\train.py --scenario status1 --config experiments\status1\train.yaml
python tools\train.py --scenario status1 --config experiments\status1\ref_voc.yaml
python tools\train.py --scenario status1 --config experiments\status1\ref_clipart.yaml
python tools\evaluate.py --scenario status1 --config experiments\status1\eval.yaml
```

只从 T2 继续训练：

```powershell
python tools\train.py --scenario status1 --config experiments\status1\train_t2_only.yaml
```

### Status3 主实验

```powershell
python tools\train.py --scenario status3 --config experiments\status3\train.yaml
python tools\train.py --scenario status3 --config experiments\status3\ref_daytime.yaml
python tools\train.py --scenario status3 --config experiments\status3\ref_night.yaml
python tools\evaluate.py --scenario status3 --config experiments\status3\eval.yaml
```

只从 T2 继续训练：

```powershell
python tools\train.py --scenario status3 --config experiments\status3\train_t2_only.yaml
```

### Ablation

生成或检查单个消融配置：

```powershell
python tools\run_ablation.py --scenario status1 --name 03_seqft_incremental_head_duet --materialize-only
```

运行单个消融：

```powershell
python tools\run_ablation.py --scenario status1 --name 03_seqft_incremental_head_duet
```

如果确实要按表格顺序跑完所有消融，再显式使用 `--all`：

```powershell
python tools\run_ablation.py --scenario status1 --all
```

当前统一消融入口只覆盖 `status1`。

## 4. 输出目录

所有训练和评估结果统一写入根目录 `output/`：

```text
output/status1/main
output/status1/ref_voc
output/status1/ref_clipart
output/status3/main
output/status3/t2_only
output/status3/ref_daytime
output/status3/ref_night
```

## 5. 已有实验结果记录

### Status1

主实验汇总：

```text
Avg RI = 86.88%
Avg GI = 44.33%
RAI    = 65.61%
```

分项：

```text
RI_VOC_1_10_after_T2      final=0.7200  reference=0.8288  ratio=86.88%
GI_VOC_11_20_at_T2        final=0.1415  reference=0.7968  ratio=17.76%
GI_Clipart_1_10_at_T2     final=0.2795  reference=0.3942  ratio=70.91%
```

Status1 消融记录：

```text
00_no_seqft                         未评估
01_seqft                            Avg RI=0.12%   Avg GI=13.22%  RAI=6.67%
02_seqft_incremental_head           Avg RI=21.02%  Avg GI=22.68%  RAI=21.85%
03_seqft_incremental_head_duet      Avg RI=69.34%  Avg GI=29.83%  RAI=49.58%
04_seqft_incremental_head_duet_distill Avg RI=88.62% Avg GI=41.69% RAI=65.15%
05_full                             Avg RI=89.23%  Avg GI=43.38%  RAI=66.31%
```

### Status3

主实验汇总：

```text
Avg RI = 80.15%
Avg GI = 38.15%
RAI    = 59.15%
```

分项：

```text
RI_DaytimeSunny_1_4_after_T2   final=0.3731  reference=0.4655  ratio=80.15%
GI_DaytimeSunny_5_7_at_T2      final=0.0798  reference=0.5150  ratio=15.49%
GI_NightSunny_1_4_at_T2        final=0.2802  reference=0.4608  ratio=60.80%
```

最终模型在 T2 数据集上的测试：

```text
Final_NightSunny_5_7_on_T2     mAP50=0.0949  ratio=9.49%
```

## 6. 说明

论文官方实现未随论文公开。本项目是课程复现版本，重点复现论文核心思想、实验组织和指标计算流程，而不是逐行复制作者私有代码。
