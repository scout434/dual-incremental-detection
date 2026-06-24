from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duet_repro.engines.scenario_runner import run_train
from duet_repro.experiments.registry import (
    SCENARIO_NAMES,
    TRAIN_VARIANTS,
    resolve_config_path,
    resolve_scenario,
    resolve_train_script,
)


def parse_args() -> argparse.Namespace:
    """解析统一训练入口的命令行参数。

    本脚本不直接写死 status1/status3 的训练细节，而是先接收场景名、
    配置文件名和少量覆盖参数，再交给 registry/scenario_runner 去定位
    真正的 YAML 和 legacy 训练脚本。
    """
    parser = argparse.ArgumentParser(description="Unified DuET scenario training entrypoint.")
    parser.add_argument("--scenario", required=True, choices=SCENARIO_NAMES)
    parser.add_argument("--variant", default="default", choices=TRAIN_VARIANTS)
    parser.add_argument("--config", default="train.yaml")
    parser.add_argument("--start-task", type=int)
    parser.add_argument("--previous-checkpoint")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 根据 --scenario 和 --config 找到当前实验场景对象。这里会处理诸如
    # "train.yaml"、"experiments/status1/train.yaml" 这类相对路径写法。
    scenario = resolve_scenario(args.scenario, config_or_plan=args.config)
    config_path = resolve_config_path(scenario, args.config)

    # 训练逻辑仍复用 legacy/status*/train_duet.py；统一入口只负责把命令行
    # 参数整理成 legacy 脚本 main() 能接受的形式。
    run_train(
        scenario,
        config_path,
        script_path=resolve_train_script(scenario, args.variant),
        start_task=args.start_task,
        previous_checkpoint=args.previous_checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

