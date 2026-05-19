# ============================================================
# DuET Pascal VOC 系列实验启动脚本 (PowerShell)
#
# 用途：自动化执行 DuET 类别增量目标检测的完整训练流程
#
# 依赖：
#   1. 已安装 conda 环境 duet-repro
#   2. 已准备好 Pascal VOC 增量数据集
#      （由 make_yolo_subsets.py 生成，位于 E:/datasets/duet_pascal_tasks/）
#
# 使用方法：
#   在项目根目录下运行：
#     .\scripts\run_pascal_series.ps1
#   或在 PowerShell 中执行：
#     & ".\scripts\run_pascal_series.ps1"
#
# 注意事项：
#   - 确保配置文件 configs/pascal_series_yolo.yaml 中的数据路径正确
#   - 训练可能需要数小时（取决于 GPU 性能）
#   - 输出将保存到 outputs/pascal_series_duet_yolo11n/
# ============================================================

# 设置错误处理策略：遇到任何错误立即停止脚本执行
$ErrorActionPreference = "Stop"

# 激活 conda 环境 duet-repro
# conda activate 会在当前 shell 中激活指定的 conda 环境
# 激活后可以使用该环境中的 Python、PyTorch 等工具
conda activate duet-repro

# 执行 DuET 训练脚本
# --config 参数指定实验配置文件路径
# 训练流程：
#   1. Task 1 (Base): 在预训练 yolo11n.pt 上训练 5 个基础类别
#   2. Task 2 (Inc 1): 合并 Task1 + 训练 Task2 增量类别
#   3. Task 3 (Inc 2): 合并 Task2 合并结果 + 训练 Task3 增量类别
#   4. Task 4 (Inc 3): 合并 Task3 合并结果 + 训练 Task4 增量类别
python train_ultralytics_duet.py --config configs/pascal_series_yolo.yaml

# 训练完成后输出提示信息（可选）
Write-Host "训练完成！请检查 outputs/pascal_series_duet_yolo11n/ 目录下的结果。"
