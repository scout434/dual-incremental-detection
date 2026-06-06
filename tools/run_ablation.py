from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duet_repro.engines.scenario_runner import run_ablation
from duet_repro.experiments.registry import get_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified status1 ablation entrypoint.")
    parser.add_argument("--scenario", default="status1", choices=["status1"])
    parser.add_argument("--name", help="Run one ablation, for example 03_seqft_incremental_head_duet.")
    parser.add_argument("--all", action="store_true", help="Run all ablations in table order.")
    parser.add_argument("--materialize-only", action="store_true", help="Only write ablation configs/eval plans.")
    args = parser.parse_args()
    if not args.name and not args.all:
        parser.error("Use --name <ablation> for one ablation, or --all to run every ablation.")
    if args.name and args.all:
        parser.error("Use either --name <ablation> or --all, not both.")
    return args


def main() -> None:
    args = parse_args()
    extra_args: list[str] = []
    if args.name:
        extra_args.extend(["--name", args.name])
    if args.all:
        extra_args.append("--all")
    if args.materialize_only:
        extra_args.append("--materialize-only")
    run_ablation(get_scenario(args.scenario), extra_args=extra_args)


if __name__ == "__main__":
    main()
