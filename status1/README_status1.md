# status1：论文 Pascal Two Phase 场景

本目录对应论文表格中的两阶段场景：

```text
T1: 1-10 from VOC
T2: 11-20 from Clipart
```

对应评估项：

```text
new:    mAP_new^T1(VOC[1:10]), mAP_new^T2(Clipart[11:20])
old:    mAP_old^T2(VOC[1:10])
unseen: mAP_unseen^T2(VOC[11:20]), mAP_unseen^T2(Clipart[1:10])
```

## 1. 先重新切干净数据

先进入 status1 目录：

```bash
cd /root/autodl-tmp/dual/dual-incremental-detection-master/status1
```

修改 `configs/prepare_pascal_2phase_slices.yaml` 里的：

```yaml
sources:
  voc: /root/autodl-tmp/voc_yolo/data.yaml
  clipart: /root/autodl-tmp/clipart_yolo/data.yaml
```

然后运行：

```bash
python prepare_paper_slices.py --config configs/prepare_pascal_2phase_slices.yaml
```

生成后运行检查脚本：

```bash
python check_status1_slices.py
```

检查结果必须满足：

```text
voc_1_10      只能出现类别 [0..9]
clipart_11_20 只能出现类别 [10..19]
voc_11_20     只能出现类别 [10..19]
clipart_1_10  只能出现类别 [0..9]
```

## 2. 跑主训练

```bash
python train_duet.py --config configs/train_pascal_2phase_full.yaml
```

训练时请看日志：

```text
T1 class_indices 必须是 [0,1,2,3,4,5,6,7,8,9]
T2 class_indices 必须是 [10,11,12,13,14,15,16,17,18,19]
```

如果 T1 被修正成 `[0..19]`，说明数据切片仍然不干净。

## 3. 跑 GI 分母参考模型

```bash
python train_duet.py --config configs/train_reference_voc_11_20.yaml
python train_duet.py --config configs/train_reference_clipart_1_10.yaml
```

## 4. 计算论文指标

```bash
python eval_paper_metrics.py --plan configs/paper_metrics_pascal_2phase_full.yaml
```

输出文件：

```text
outputs/status1_pascal_2phase_duet_yolo11n/paper_metrics_results.json
```

## 5. 这组代码和论文的对应关系

T1 使用普通检测训练。

T2 从 T1 merged checkpoint 继续训练，并接入：

```text
L = Ldet + lambda_distill * Ldistill + lambda_dc * LDC
```

T2 结束后执行 DuET task arithmetic：

```text
theta_merged = theta_ref + alpha_l * tau_old + beta_l * tau_curr
```

检测头不参与共享参数融合，而是按 `class_indices` 做类别通道切片保留。
