# status2：论文 Multi Phase 场景

本目录对应论文表格中的四阶段场景：

```text
Watercolor [1:3] -> Comic [4:6] -> Clipart [7:13] -> VOC [14:20]
```

运行前先把 `status2/configs/*.yaml` 里的 `data:` 路径改成你服务器上已有切片的真实路径。

主训练：

```bash
python status2/train_duet.py --config status2/configs/train_multiphase_full.yaml
```

GI 分母参考模型：

```bash
python status2/train_duet.py --config status2/configs/train_reference_watercolor_4_6.yaml
python status2/train_duet.py --config status2/configs/train_reference_comic_1_3.yaml
python status2/train_duet.py --config status2/configs/train_reference_clipart_1_6.yaml
python status2/train_duet.py --config status2/configs/train_reference_voc_1_13.yaml
```

计算论文指标：

```bash
python status2/eval_paper_metrics.py --plan status2/configs/paper_metrics_multiphase_full.yaml
```

如果你是在 `status2/` 目录内部运行，也可以省略 `status2/` 前缀：

```bash
python train_duet.py --config configs/train_multiphase_full.yaml
python eval_paper_metrics.py --plan configs/paper_metrics_multiphase_full.yaml
```
