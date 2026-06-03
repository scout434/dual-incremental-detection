# Status1 消融实验说明

这组消融对应论文表中 `Seq FT / Incremental Head / DuET Module / L_Distill / L_DC`
这几项组件。为了节省时间，除 `00_no_seqft` 外，其余实验都复用已经训练好的
T1 checkpoint，然后只训练 T2。

## 前置文件

主实验的 T1 模型必须存在：

```text
status1/output/main/task_1_voc_1_10_best.pt
```

GI 分母参考模型也需要存在：

```text
status1/output/ref_voc/task_1_voc_11_20_best.pt
status1/output/ref_clipart/task_1_clipart_1_10_best.pt
```

如果这些文件不存在，先运行：

```powershell
python status1\train_duet.py --config status1\configs\train.yaml
python status1\train_duet.py --config status1\configs\ref_voc.yaml
python status1\train_duet.py --config status1\configs\ref_clipart.yaml
```

## 消融组别

```text
00_no_seqft
  Seq FT: no
  Incremental Head: no
  DuET Module: no
  L_Distill: no
  L_DC: no
  说明：不复用 T1，从 reference_full_head 开始训练 T2。

01_seqft
  Seq FT: yes
  Incremental Head: no
  DuET Module: no
  L_Distill: no
  L_DC: no
  说明：复用 T1，直接顺序微调 T2，旧类行不做回填保护。

02_seqft_incremental_head
  Seq FT: yes
  Incremental Head: yes
  DuET Module: no
  L_Distill: no
  L_DC: no
  说明：复用 T1，T2 后把旧类分类行从 T1 回填，新类分类行来自 T2。

03_seqft_incremental_head_duet
  Seq FT: yes
  Incremental Head: yes
  DuET Module: yes
  L_Distill: no
  L_DC: no
  说明：共享 backbone/neck 走 DuET 融合，head 行级回填。

04_seqft_incremental_head_duet_distill
  Seq FT: yes
  Incremental Head: yes
  DuET Module: yes
  L_Distill: yes
  L_DC: no
  说明：在 03 基础上加入旧模型蒸馏。

05_full
  Seq FT: yes
  Incremental Head: yes
  DuET Module: yes
  L_Distill: yes
  L_DC: yes
  说明：完整方法。
```

## 生成配置

只生成训练配置和评估配置，不开始训练：

```powershell
python status1\run_ablation.py --all --materialize-only
```

生成后会得到：

```text
status1/configs/ablations/00_no_seqft.yaml
status1/configs/ablations/00_no_seqft_eval.yaml
...
status1/configs/ablations/05_full.yaml
status1/configs/ablations/05_full_eval.yaml
status1/configs/ablations/index.json
```

## 单独运行某一组

只跑 `00_no_seqft`：

```powershell
python status1\run_ablation.py --name 00_no_seqft
python status1\eval_paper_metrics.py --plan status1\configs\ablations\00_no_seqft_eval.yaml
```

只跑 `01_seqft`：

```powershell
python status1\run_ablation.py --name 01_seqft
python status1\eval_paper_metrics.py --plan status1\configs\ablations\01_seqft_eval.yaml
```

只跑 `02_seqft_incremental_head`：

```powershell
python status1\run_ablation.py --name 02_seqft_incremental_head
python status1\eval_paper_metrics.py --plan status1\configs\ablations\02_seqft_incremental_head_eval.yaml
```

只跑 `03_seqft_incremental_head_duet`：

```powershell
python status1\run_ablation.py --name 03_seqft_incremental_head_duet
python status1\eval_paper_metrics.py --plan status1\configs\ablations\03_seqft_incremental_head_duet_eval.yaml
```

只跑 `04_seqft_incremental_head_duet_distill`：

```powershell
python status1\run_ablation.py --name 04_seqft_incremental_head_duet_distill
python status1\eval_paper_metrics.py --plan status1\configs\ablations\04_seqft_incremental_head_duet_distill_eval.yaml
```

只跑 `05_full`：

```powershell
python status1\run_ablation.py --name 05_full
python status1\eval_paper_metrics.py --plan status1\configs\ablations\05_full_eval.yaml
```

## 一次跑完全部消融

训练全部 6 组：

```powershell
python status1\run_ablation.py --all
```

然后依次评估：

```powershell
python status1\eval_paper_metrics.py --plan status1\configs\ablations\00_no_seqft_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\01_seqft_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\02_seqft_incremental_head_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\03_seqft_incremental_head_duet_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\04_seqft_incremental_head_duet_distill_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\05_full_eval.yaml
```

每组输出保存在：

```text
status1/output/ablations/<ablation_name>/metrics.json
status1/output/ablations/<ablation_name>/rai_metrics.json
```

## 指标

每组都计算：

```text
Avg RI = final 在 VOC[1:10] 上的 mAP / T1 模型在 VOC[1:10] 上的 mAP
Avg GI = mean(
  final 在 VOC[11:20] 上的 mAP / VOC[11:20] 参考模型 mAP,
  final 在 Clipart[1:10] 上的 mAP / Clipart[1:10] 参考模型 mAP
)
RAI = (Avg RI + Avg GI) / 2
```
