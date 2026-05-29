from __future__ import annotations

"""
Debug entry for status1 T2 only.

It reuses a finished T1 checkpoint and starts training from task 2
without rerunning VOC [1:10]. Loss weights can be overridden from CLI.

Examples:
  python debug_train_t2_from_t1.py --config configs/train_pascal_2phase_full.yaml --distill-weight 0 --dc-weight 0 --epochs 10 --skip-merge
  python debug_train_t2_from_t1.py --config configs/train_pascal_2phase_full.yaml --distill-weight 0.01 --dc-weight 0 --epochs 10 --skip-merge
  python debug_train_t2_from_t1.py --config configs/train_pascal_2phase_full.yaml --distill-weight 0 --dc-weight 0.01 --epochs 10 --skip-merge
  python debug_train_t2_from_t1.py --config configs/train_pascal_2phase_full.yaml --device 1 --epochs 100
"""

import argparse
import copy
import multiprocessing
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if not (PROJECT_ROOT / "duet_repro").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from duet_repro.core.task_vectors import inject_state_dict_into_checkpoint, load_state_dict, task_vector
from duet_repro.core.incremental_head import inject_incremental_head_checkpoint
from train_duet import (
    load_config,
    merge_state_dicts_with_duet_module,
    prepare_global_task_data,
    resolve_config_path,
    save_reference_checkpoint,
    seed_everything,
    setup_console_txt_log,
    train_one_task,
    write_experiment_state,
)


def resolve_existing_path(path: Path) -> Path:
    """Resolve a path against the normal project roots used by train_duet.py."""
    if path.is_absolute():
        return path.resolve()
    for base in (PROJECT_ROOT, SCRIPT_DIR, Path.cwd()):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def default_t1_checkpoint(output_dir: Path) -> Path:
    return resolve_existing_path(output_dir / "task_1_voc_1_10_best.pt")


def resolve_status1_config(config_path: str | Path) -> Path:
    """Prefer configs under status1, avoiding similarly named root configs."""
    path = Path(config_path)
    if path.is_absolute():
        return path.resolve()
    for base in (SCRIPT_DIR, Path.cwd(), PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return resolve_config_path(config_path)


def main() -> None:
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_pascal_2phase_full.yaml")
    parser.add_argument("--t1-ckpt", default=None, help="Finished T1 best/merged checkpoint.")
    parser.add_argument("--output-suffix", default="debug_t2", help="Suffix appended to experiment output_dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override T2 epochs.")
    parser.add_argument("--device", default=None, help="Override device, e.g. 0, 1, cpu.")
    parser.add_argument("--distill-weight", type=float, default=None, help="Override duet.distill_weight.")
    parser.add_argument("--dc-weight", type=float, default=None, help="Override duet.dc_weight.")
    parser.add_argument("--skip-merge", action="store_true", help="Train T2 only and skip DuET merge.")
    args = parser.parse_args()

    config_path = resolve_status1_config(args.config)
    os.chdir(PROJECT_ROOT)

    cfg = load_config(config_path)
    cfg = copy.deepcopy(cfg)
    seed_everything(int(cfg["experiment"].get("seed", 42)))

    base_output_dir = Path(cfg["experiment"]["output_dir"])
    t1_ckpt = resolve_existing_path(Path(args.t1_ckpt)) if args.t1_ckpt else default_t1_checkpoint(base_output_dir)
    if not t1_ckpt.exists():
        raise FileNotFoundError(f"T1 checkpoint not found: {t1_ckpt}")

    debug_output_dir = Path(f"{base_output_dir}_{args.output_suffix}")
    cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', 'status1')}_{args.output_suffix}"
    cfg["experiment"]["output_dir"] = str(debug_output_dir)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.distill_weight is not None:
        cfg["duet"]["distill_weight"] = float(args.distill_weight)
    if args.dc_weight is not None:
        cfg["duet"]["dc_weight"] = float(args.dc_weight)

    debug_output_dir.mkdir(parents=True, exist_ok=True)
    setup_console_txt_log(debug_output_dir)

    print("\n" + "=" * 72)
    print("[Status1 T2 Debug] Start from T2 only, reuse finished T1 checkpoint")
    print(f"config              : {args.config}")
    print(f"T1 checkpoint       : {t1_ckpt}")
    print(f"debug output_dir    : {debug_output_dir}")
    print(f"epochs              : {cfg['training'].get('epochs')}")
    print(f"device              : {cfg['training'].get('device')}")
    print(f"distill_weight      : {cfg['duet'].get('distill_weight')}")
    print(f"dc_weight           : {cfg['duet'].get('dc_weight')}")
    print(f"skip_merge          : {args.skip_merge}")
    print("=" * 72 + "\n")

    reference_ckpt = save_reference_checkpoint(cfg, debug_output_dir)
    reference_state = load_state_dict(reference_ckpt)
    old_state = load_state_dict(t1_ckpt)
    duet_cfg = cfg.get("duet", {})
    shared_key_exclude = tuple(duet_cfg.get("shared_key_exclude", []))

    tasks = cfg["tasks"]
    if len(tasks) < 2:
        raise ValueError("This debug script needs at least two tasks in config.")

    t1_task = tasks[0]
    t2_task = tasks[1]
    learned_indices = [int(i) for i in t1_task["class_indices"]]

    prepared_data = prepare_global_task_data(t2_task, cfg, debug_output_dir)
    current_indices = [int(i) for i in t2_task.get("resolved_class_indices", t2_task["class_indices"])]

    prev_tv = task_vector(reference_state, old_state, shared_key_exclude=shared_key_exclude)
    history = [
        {
            "task_index": 1,
            "task": t1_task["name"],
            "merged_checkpoint": str(t1_ckpt),
            "is_reused_for_t2_debug": True,
            "class_indices": learned_indices,
        }
    ]
    write_experiment_state(
        debug_output_dir,
        cfg,
        history,
        reference_ckpt=reference_ckpt,
        latest_ckpt=t1_ckpt,
    )

    print("\n" + "=" * 72)
    print(f"[Status1 T2 Debug] Training T2: {t2_task['name']}")
    print(f"source data.yaml    : {t2_task['data']}")
    print(f"prepared data.yaml  : {prepared_data}")
    print(f"old class indices   : {learned_indices}")
    print(f"new class indices   : {current_indices}")
    print("=" * 72 + "\n")

    trained_ckpt = train_one_task(
        t1_ckpt,
        t2_task,
        cfg,
        debug_output_dir,
        is_first=False,
        teacher_ckpt=t1_ckpt,
        prev_task_vector=prev_tv,
        task_vector_history=[prev_tv],
        reference_state=reference_state,
        old_class_indices=learned_indices,
    )
    print(f"[Status1 T2 Debug] T2 trained checkpoint: {trained_ckpt}")

    if args.skip_merge:
        latest_ckpt = trained_ckpt
        print("[Status1 T2 Debug] Skip DuET merge because --skip-merge is set.")
    else:
        print("[Status1 T2 Debug] Running DuET merge.")
        new_state = load_state_dict(trained_ckpt)
        merged_state, report = merge_state_dicts_with_duet_module(
            reference_state,
            old_state,
            new_state,
            gamma=float(duet_cfg.get("gamma", 0.1)),
            alpha_base=float(duet_cfg.get("alpha_base", 0.5)),
            shared_key_exclude=shared_key_exclude,
            per_layer_report=duet_cfg.get("verbose_merge", False),
        )
        latest_ckpt = debug_output_dir / f"task_2_{t2_task['name']}_duet.pt"
        inject_incremental_head_checkpoint(
            template_checkpoint_path=trained_ckpt,
            merged_shared_state=merged_state,
            old_checkpoint_path=t1_ckpt,
            new_checkpoint_path=trained_ckpt,
            output_path=latest_ckpt,
            old_class_indices=learned_indices,
            new_class_indices=current_indices,
            total_classes=int(cfg["detector"]["total_classes"]),
        )
        print(
            "[Status1 T2 Debug] merged_keys={0} skipped_keys={1}".format(
                report["merged_keys"],
                report["skipped_keys"],
            )
        )
        print(f"[Status1 T2 Debug] merged checkpoint: {latest_ckpt}")

    history.append(
        {
            "task_index": 2,
            "task": t2_task["name"],
            "data": t2_task["data"],
            "prepared_data": t2_task.get("prepared_data"),
            "label_space": t2_task.get("label_space"),
            "class_indices": current_indices,
            "trained_checkpoint": str(trained_ckpt),
            "merged_checkpoint": str(latest_ckpt),
            "debug_distill_weight": cfg["duet"].get("distill_weight"),
            "debug_dc_weight": cfg["duet"].get("dc_weight"),
        }
    )
    write_experiment_state(
        debug_output_dir,
        cfg,
        history,
        reference_ckpt=reference_ckpt,
        latest_ckpt=latest_ckpt,
    )

    print("\n[Status1 T2 Debug] Done.")
    print(f"[Status1 T2 Debug] trained checkpoint: {trained_ckpt}")
    print(f"[Status1 T2 Debug] latest checkpoint : {latest_ckpt}")


if __name__ == "__main__":
    main()
