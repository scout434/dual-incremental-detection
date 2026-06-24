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
    """解析统一评估入口参数。

    --config 和 --plan 指向的是同一个参数位：评估脚本内部都称为 plan。
    这样既兼容用户习惯的 --config eval.yaml，也兼容更准确的 --plan eval.yaml。
    """
    parser = argparse.ArgumentParser(description="Unified DuET scenario evaluation entrypoint.")
    parser.add_argument("--scenario", required=True, choices=SCENARIO_NAMES)
    parser.add_argument("--config", "--plan", dest="plan", default="eval.yaml")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-alias", action="append", default=[], metavar="NAME=PATH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 根据场景名和 plan 路径解析出完整的实验描述。resolve_plan_path 会优先
    # 查找 experiments/status*/ 下的 YAML，因此命令行里可以写短路径。
    scenario = resolve_scenario(args.scenario, config_or_plan=args.plan)
    plan_path = resolve_plan_path(scenario, args.plan)

    # checkpoint alias 用于临时把 eval.yaml 里的某个别名改成新权重路径，
    # 不会直接改原始配置文件，便于复现实验和临时测试。
    if args.checkpoint_alias:
        plan_path = write_plan_with_alias_overrides(plan_path, args.checkpoint_alias)

    extra_args: list[str] = []
    if args.output:
        extra_args.extend(["--output", args.output])
    if args.device:
        extra_args.extend(["--device", args.device])

    # 评估最终仍调用 legacy/status*/eval_paper_metrics.py，以保证指标口径
    # 和论文复现实验保持一致。
    run_eval(scenario, plan_path, extra_args=extra_args)


def write_plan_with_alias_overrides(plan_path: Path, overrides: list[str]) -> Path:
    """生成一份临时评估计划，用命令行 alias 覆盖原 plan 中的权重路径。"""
    text = plan_path.read_text(encoding="utf-8")
    plan = yaml.safe_load(text) if plan_path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    plan.setdefault("checkpoint_aliases", {})
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--checkpoint-alias must use NAME=PATH, got: {item}")
        name, value = item.split("=", 1)
        plan["checkpoint_aliases"][name.strip()] = value.strip()

    # 临时 plan 放在 outputs/_tool_eval_plans，避免污染 experiments 下的标准配置。
    output_dir = ROOT / "outputs" / "_tool_eval_plans"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{plan_path.stem}_alias_override.yaml"
    output_path.write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    main()

