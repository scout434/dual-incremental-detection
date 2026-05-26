# status3：论文 Diverse Weather Two Phase 场景

本目录对应论文表格中的两阶段天气场景：

```text
Daytime Sunny [1:4] -> Night Sunny [5:7]
```

运行前先把 `status3/configs/*.yaml` 里的 `data:` 路径改成你服务器上已有切片的真实路径。

主训练：

```bash
python status3/train_duet.py --config status3/configs/train_weather_2phase_full.yaml
```

GI 分母参考模型：

```bash
python status3/train_duet.py --config status3/configs/train_reference_daytime_sunny_5_7.yaml
python status3/train_duet.py --config status3/configs/train_reference_night_sunny_1_4.yaml
```

计算论文指标：

```bash
python status3/eval_paper_metrics.py --plan status3/configs/paper_metrics_weather_2phase_full.yaml
```

如果你是在 `status3/` 目录内部运行，也可以省略 `status3/` 前缀：

```bash
python train_duet.py --config configs/train_weather_2phase_full.yaml
python eval_paper_metrics.py --plan configs/paper_metrics_weather_2phase_full.yaml
```
