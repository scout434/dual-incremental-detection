from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "duet_repro").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "duet_repro").exists():
    raise RuntimeError(f"Could not locate project root from {SCRIPT_DIR}")
CONFIG_DIR = PROJECT_ROOT / "experiments" / "status1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


BASE_T2_CONFIG = CONFIG_DIR / "train_t2_only.yaml"
BASE_EVAL_PLAN = CONFIG_DIR / "eval.yaml"


ABLATIONS: dict[str, dict[str, Any]] = {
    "00_no_seqft": {
        "title": "No Seq FT / No Incremental Head / No DuET / No losses",
        "seq_ft": False,
        "incremental_head": False,
        "duet_enabled": False,
        "distill_weight": 0.0,
        "dc_weight": 0.0,
    },
    "01_seqft": {
        "title": "Seq FT only",
        "seq_ft": True,
        "incremental_head": False,
        "duet_enabled": False,
        "distill_weight": 0.0,
        "dc_weight": 0.0,
    },
    "02_seqft_incremental_head": {
        "title": "Seq FT + Incremental Head",
        "seq_ft": True,
        "incremental_head": True,
        "duet_enabled": False,
        "distill_weight": 0.0,
        "dc_weight": 0.0,
    },
    "03_seqft_incremental_head_duet": {
        "title": "Seq FT + Incremental Head + DuET Module",
        "seq_ft": True,
        "incremental_head": True,
        "duet_enabled": True,
        "distill_weight": 0.0,
        "dc_weight": 0.0,
    },
    "04_seqft_incremental_head_duet_distill": {
        "title": "Seq FT + Incremental Head + DuET Module + Distill",
        "seq_ft": True,
        "incremental_head": True,
        "duet_enabled": True,
        "distill_weight": 0.01,
        "dc_weight": 0.0,
    },
    "05_full": {
        "title": "Seq FT + Incremental Head + DuET Module + Distill + DC",
        "seq_ft": True,
        "incremental_head": True,
        "duet_enabled": True,
        "distill_weight": 0.01,
        "dc_weight": 0.01,
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_train_config(name: str) -> tuple[dict[str, Any], Path]:
    spec = ABLATIONS[name]
    cfg = copy.deepcopy(load_yaml(BASE_T2_CONFIG))
    output_dir = f"output/status1/ablations/{name}"
    base_t1_checkpoint = cfg.get("resume", {}).get("previous_checkpoint")
    if not base_t1_checkpoint:
        raise ValueError(f"Missing resume.previous_checkpoint in {BASE_T2_CONFIG}")

    cfg["experiment"]["name"] = f"ablation_{name}"
    cfg["experiment"]["output_dir"] = output_dir
    cfg.setdefault("ablation", {})
    cfg["ablation"]["name"] = name
    cfg["ablation"]["title"] = spec["title"]
    cfg["ablation"]["seq_ft"] = bool(spec["seq_ft"])
    cfg["ablation"]["incremental_head"] = bool(spec["incremental_head"])

    cfg.setdefault("resume", {})
    cfg["resume"]["start_task"] = 2
    if spec["seq_ft"]:
        cfg["resume"]["previous_checkpoint"] = base_t1_checkpoint
        cfg["resume"].pop("allow_full_head_previous", None)
    else:
        cfg["resume"]["previous_checkpoint"] = f"{output_dir}/reference_full_head.pt"
        cfg["resume"]["allow_full_head_previous"] = True

    cfg.setdefault("duet", {})
    cfg["duet"]["enabled"] = bool(spec["duet_enabled"])
    cfg["duet"]["use_duet_module"] = bool(spec["duet_enabled"])
    cfg["duet"]["distill_weight"] = float(spec["distill_weight"])
    cfg["duet"]["dc_weight"] = float(spec["dc_weight"])
    cfg["duet"]["shared_key_exclude"] = ["model.23"]

    train_cfg_path = CONFIG_DIR / "ablations" / f"{name}.yaml"
    return cfg, train_cfg_path


def build_eval_plan(name: str, train_cfg: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    output_dir = train_cfg["experiment"]["output_dir"]
    plan = copy.deepcopy(load_yaml(BASE_EVAL_PLAN))
    base_t1_checkpoint = train_cfg.get("resume", {}).get("previous_checkpoint")
    if not base_t1_checkpoint:
        raise ValueError("Missing resume.previous_checkpoint in materialized train config.")

    plan["manifest"] = f"{output_dir}/eval_manifest.json"
    plan["output"] = f"{output_dir}/metrics.json"
    plan.setdefault("checkpoint_aliases", {})
    plan["checkpoint_aliases"].update(
        {
            "t1": base_t1_checkpoint,
            "task_1": base_t1_checkpoint,
            "voc_1_10": base_t1_checkpoint,
        }
    )
    eval_plan_path = CONFIG_DIR / "ablations" / f"{name}_eval.yaml"
    return plan, eval_plan_path


def materialize(name: str) -> tuple[Path, Path]:
    train_cfg, train_cfg_path = build_train_config(name)
    eval_plan, eval_plan_path = build_eval_plan(name, train_cfg)
    write_yaml(train_cfg_path, train_cfg)
    write_yaml(eval_plan_path, eval_plan)

    index_path = CONFIG_DIR / "ablations" / "index.json"
    index_payload = {
        key: {
            "title": value["title"],
            "train_config": f"experiments/status1/ablations/{key}.yaml",
            "eval_plan": f"experiments/status1/ablations/{key}_eval.yaml",
            "output_dir": f"output/status1/ablations/{key}",
            "switches": {
                "seq_ft": value["seq_ft"],
                "incremental_head": value["incremental_head"],
                "duet_module": value["duet_enabled"],
                "distill": value["distill_weight"] > 0,
                "dc": value["dc_weight"] > 0,
            },
        }
        for key, value in ABLATIONS.items()
    }
    index_path.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return train_cfg_path, eval_plan_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or materialize status1 ablation configs.")
    parser.add_argument("--name", choices=sorted(ABLATIONS), help="Ablation name to run.")
    parser.add_argument("--all", action="store_true", help="Run all ablations in table order.")
    parser.add_argument("--materialize-only", action="store_true", help="Only write configs/eval plans.")
    args = parser.parse_args()

    selected = sorted(ABLATIONS) if args.all else [args.name]
    if not selected or selected == [None]:
        parser.error("Use --name <ablation> or --all.")

    for name in selected:
        train_cfg_path, eval_plan_path = materialize(name)
        print(f"[ablation] {name}: {ABLATIONS[name]['title']}")
        print(f"[ablation] train config: {display_path(train_cfg_path)}")
        print(f"[ablation] eval plan:    {display_path(eval_plan_path)}")
        if args.materialize_only:
            continue

        from train_duet import main as train_main

        os.chdir(PROJECT_ROOT)
        train_main(train_cfg_path)


if __name__ == "__main__":
    main()
