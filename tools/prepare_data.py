from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duet_repro.engines.scenario_runner import run_prepare_data
from duet_repro.experiments.registry import SCENARIO_NAMES, resolve_prepare_plan
from duet_repro.utils.paths import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DuET data preparation entrypoint.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIO_NAMES)
    group.add_argument("--plan")
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-update-configs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = resolve_prepare_plan(args.scenario) if args.scenario else resolve_repo_path(args.plan)
    extra_args: list[str] = []
    if args.copy_files:
        extra_args.append("--copy-files")
    if args.overwrite:
        extra_args.append("--overwrite")
    if args.no_update_configs:
        extra_args.append("--no-update-configs")
    run_prepare_data(plan_path, extra_args=extra_args)


if __name__ == "__main__":
    main()

