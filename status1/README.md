# Status1 DuET

This directory contains the consolidated Pascal two-phase DuET reproduction.

## Train

Run the main experiment:

```powershell
python status1\train_duet.py --config status1\configs\train.yaml
```

Reuse the completed T1 checkpoint and train only T2:

```powershell
python status1\train_duet.py --config status1\configs\train_t2_only.yaml
```

Train the two single-task reference models used as GI denominators:

```powershell
python status1\train_duet.py --config status1\configs\ref_voc.yaml
python status1\train_duet.py --config status1\configs\ref_clipart.yaml
```

## Evaluate

```powershell
python status1\eval_paper_metrics.py --plan status1\configs\eval.yaml
```

## Outputs

The main files are intentionally short and live below `status1/output/`:

```text
status1/output/main/task_1_voc_1_10_best.pt
status1/output/main/task_2_clipart_11_20_duet.pt
status1/output/main/metrics.json
status1/output/ref_voc/task_1_voc_11_20_best.pt
status1/output/ref_clipart/task_1_clipart_1_10_best.pt
```

`task_2_clipart_11_20_duet.pt` uses a single cumulative YOLO11n Detect head. Shared backbone/neck
parameters are merged with DuET task arithmetic. The previous Detect head is
kept as the template and the current task classification rows are inserted into
that head after training.
