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
    scenario = resolve_scenario(args.scenario, config_or_plan=args.config)
    config_path = resolve_config_path(scenario, args.config)
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

