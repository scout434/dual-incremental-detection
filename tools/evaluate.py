from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duet_repro.engines.scenario_runner import run_eval
from duet_repro.experiments.registry import SCENARIO_NAMES, resolve_plan_path, resolve_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DuET scenario evaluation entrypoint.")
    parser.add_argument("--scenario", required=True, choices=SCENARIO_NAMES)
    parser.add_argument("--config", "--plan", dest="plan", default="eval.yaml")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-alias", action="append", default=[], metavar="NAME=PATH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = resolve_scenario(args.scenario, config_or_plan=args.plan)
    plan_path = resolve_plan_path(scenario, args.plan)
    if args.checkpoint_alias:
        plan_path = write_plan_with_alias_overrides(plan_path, args.checkpoint_alias)

    extra_args: list[str] = []
    if args.output:
        extra_args.extend(["--output", args.output])
    if args.device:
        extra_args.extend(["--device", args.device])
    run_eval(scenario, plan_path, extra_args=extra_args)


def write_plan_with_alias_overrides(plan_path: Path, overrides: list[str]) -> Path:
    text = plan_path.read_text(encoding="utf-8")
    plan = yaml.safe_load(text) if plan_path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    plan.setdefault("checkpoint_aliases", {})
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--checkpoint-alias must use NAME=PATH, got: {item}")
        name, value = item.split("=", 1)
        plan["checkpoint_aliases"][name.strip()] = value.strip()

    output_dir = ROOT / "outputs" / "_tool_eval_plans"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{plan_path.stem}_alias_override.yaml"
    output_path.write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    main()

