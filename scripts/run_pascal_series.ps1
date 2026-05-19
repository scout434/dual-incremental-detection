$ErrorActionPreference = "Stop"

conda activate duet-repro
python train_ultralytics_duet.py --config configs/pascal_series_yolo.yaml

