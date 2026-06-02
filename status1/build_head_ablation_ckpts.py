from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from duet_repro.core.task_vectors import inject_state_dict_into_checkpoint, load_state_dict


DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/status1_pascal_2phase_duet_yolo11n_paperhead_t2/eval_manifest.json"
DEFAULT_PLAN = SCRIPT_DIR / "configs/paper_metrics_pascal_2phase_paperhead_t2.yaml"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, PROJECT_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def is_detect_key(key: str) -> bool:
    return key.startswith("model.23.")


def is_class_output_key(key: str) -> bool:
    lowered = key.lower()
    return "model.23.cv3." in lowered and (lowered.endswith(".2.weight") or lowered.endswith(".2.bias"))


def same_shape(state: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> bool:
    return key in state and tuple(state[key].shape) == tuple(value.shape)


def copy_exact_prefix(
    target: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
    *,
    skip_class_output: bool = True,
) -> list[str]:
    copied: list[str] = []
    for key, value in list(target.items()):
        if skip_class_output and is_class_output_key(key):
            continue
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if same_shape(source, key, value):
            target[key] = source[key].detach().clone().to(dtype=value.dtype)
            copied.append(key)
    return copied


def copy_incremental_class_rows(
    target: dict[str, torch.Tensor],
    old_state: dict[str, torch.Tensor],
    new_state: dict[str, torch.Tensor],
    *,
    old_indices: list[int],
    new_indices: list[int],
) -> dict[str, int]:
    old_rows = 0
    new_rows = 0
    class_keys = 0
    for key, value in list(target.items()):
        if not is_class_output_key(key):
            continue
        class_keys += 1
        merged = value.detach().clone()
        if key in old_state:
            for index in old_indices:
                if index < merged.shape[0] and index < old_state[key].shape[0]:
                    merged[index] = old_state[key][index].to(dtype=merged.dtype)
                    old_rows += 1
        if key in new_state:
            for index in new_indices:
                if index < merged.shape[0] and index < new_state[key].shape[0]:
                    merged[index] = new_state[key][index].to(dtype=merged.dtype)
                    new_rows += 1
        target[key] = merged

    if class_keys == 0:
        raise RuntimeError("No YOLO Detect class-output tensors were found.")
    return {"class_output_tensors": class_keys, "old_rows": old_rows, "new_rows": new_rows}


def write_variant_manifest_and_plan(
    *,
    variant_dir: Path,
    variant_name: str,
    source_manifest: dict[str, Any],
    source_plan: dict[str, Any],
    ckpt_path: Path,
) -> tuple[Path, Path]:
    manifest = copy.deepcopy(source_manifest)
    manifest["latest_checkpoint"] = str(ckpt_path)
    manifest["output_dir"] = str(variant_dir)
    manifest["ablation"] = {
        "name": variant_name,
        "source_manifest": str(DEFAULT_MANIFEST),
        "latest_checkpoint": str(ckpt_path),
    }
    manifest_path = variant_dir / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    plan = copy.deepcopy(source_plan)
    plan["manifest"] = str(manifest_path)
    plan["output"] = str(variant_dir / "paper_metrics_results.json")
    plan_path = variant_dir / "paper_metrics_plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return manifest_path, plan_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline Detect-head ablation checkpoints for the status1 paperhead T2 run."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.manifest)
    plan_path = resolve_project_path(args.plan)
    manifest = load_json(manifest_path)
    plan = load_yaml(plan_path)
    output_root = resolve_project_path(args.out_dir) if args.out_dir else resolve_project_path(manifest["output_dir"]) / "head_ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    history = manifest["history"]
    t1_entry = next(item for item in history if int(item["task_index"]) == 1)
    t2_entry = next(item for item in history if int(item["task_index"]) == 2)

    t1_ckpt = resolve_project_path(t1_entry["merged_checkpoint"])
    t2_trained_ckpt = resolve_project_path(t2_entry["trained_checkpoint"])
    final_ckpt = resolve_project_path(manifest["latest_checkpoint"])

    print("[inputs]")
    print(f"T1 checkpoint       : {t1_ckpt}")
    print(f"T2 trained checkpoint: {t2_trained_ckpt}")
    print(f"DuET final checkpoint: {final_ckpt}")
    print(f"output root         : {output_root}")

    t1_state = load_state_dict(t1_ckpt)
    t2_state = load_state_dict(t2_trained_ckpt)
    final_state = load_state_dict(final_ckpt)

    variants = {
        "bbox_dfl_t1": ("model.23.cv2.", "model.23.dfl."),
        "bbox_dfl_clsmid_t1": ("model.23.cv2.", "model.23.dfl.", "model.23.cv3."),
        "head_t1_new_rows_t2": ("model.23.",),
    }

    old_indices = [int(i) for i in t1_entry["class_indices"]]
    new_indices = [int(i) for i in t2_entry["class_indices"]]
    summaries: list[dict[str, Any]] = []

    for variant_name, prefixes in variants.items():
        variant_dir = output_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().clone() for key, value in final_state.items()}
        copied = copy_exact_prefix(state, t1_state, prefixes, skip_class_output=True)
        row_report = copy_incremental_class_rows(
            state,
            t1_state,
            t2_state,
            old_indices=old_indices,
            new_indices=new_indices,
        )

        ckpt_path = variant_dir / f"{variant_name}.pt"
        inject_state_dict_into_checkpoint(final_ckpt, state, ckpt_path)
        manifest_out, plan_out = write_variant_manifest_and_plan(
            variant_dir=variant_dir,
            variant_name=variant_name,
            source_manifest=manifest,
            source_plan=plan,
            ckpt_path=ckpt_path,
        )
        summary = {
            "variant": variant_name,
            "checkpoint": str(ckpt_path),
            "manifest": str(manifest_out),
            "plan": str(plan_out),
            "prefixes_from_t1": prefixes,
            "exact_head_tensors_copied_from_t1": len(copied),
            **row_report,
        }
        summaries.append(summary)
        print(f"\n[{variant_name}]")
        print(f"copied exact tensors from T1: {len(copied)}")
        print(f"class row report            : {row_report}")
        print(f"checkpoint                  : {ckpt_path}")
        print(f"eval plan                   : {plan_out}")

    summary_path = output_root / "head_ablation_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[done]")
    print(f"summary: {summary_path}")
    print("Run one of these from status1, for example:")
    print(f"python eval_paper_metrics.py --plan {summaries[-1]['plan']}")


if __name__ == "__main__":
    main()
