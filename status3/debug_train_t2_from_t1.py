from __future__ import annotations

"""
只从 status3 的 T2 开始调试训练。

用途：
1. 复用已经训练好的 T1 checkpoint，不再浪费时间重跑 T1。
2. 通过命令行覆盖 distill_weight / dc_weight，逐步定位 T2 NaN/Inf 的来源。
3. T2 训练结束后仍执行一次 DuET merge，便于比较 trained checkpoint 和 merged checkpoint。

示例：
  # 先完全关闭 DuET loss，只看 T2 普通增量训练是否稳定
  python debug_train_t2_from_t1.py --config configs/train_weather_2phase_full.yaml --distill-weight 0 --dc-weight 0 --epochs 10

  # 只开蒸馏
  python debug_train_t2_from_t1.py --config configs/train_weather_2phase_full.yaml --distill-weight 0.01 --dc-weight 0 --epochs 10

  # 只开 DC
  python debug_train_t2_from_t1.py --config configs/train_weather_2phase_full.yaml --distill-weight 0 --dc-weight 0.01 --epochs 10
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
from train_duet import (
    merge_full_head_slices,
    merge_state_dicts_with_duet_module,
    prepare_global_task_data,
    resolve_config_path,
    seed_everything,
    setup_console_txt_log,
    load_config,
    save_reference_checkpoint,
    train_one_task,
    write_experiment_state,
)


def default_t1_checkpoint(output_dir: Path) -> Path:
    return output_dir / "task_1_daytime_sunny_1_4_best.pt"


def main() -> None:
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_weather_2phase_full.yaml")
    parser.add_argument("--t1-ckpt", default=None, help="已经跑好的 T1 merged/best checkpoint")
    parser.add_argument("--output-suffix", default="debug_t2", help="调试输出目录后缀，避免覆盖正式结果")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 T2 训练 epoch，建议先用 3/5/10 快速排查")
    parser.add_argument("--device", default=None, help="覆盖训练设备，例如 0、1、cpu")
    parser.add_argument("--distill-weight", type=float, default=None)
    parser.add_argument("--dc-weight", type=float, default=None)
    parser.add_argument("--skip-merge", action="store_true", help="只训练 T2，不做 DuET merge")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    cfg = load_config(resolve_config_path(args.config))
    cfg = copy.deepcopy(cfg)
    seed_everything(int(cfg["experiment"].get("seed", 42)))

    base_output_dir = Path(cfg["experiment"]["output_dir"])
    t1_ckpt = Path(args.t1_ckpt) if args.t1_ckpt else default_t1_checkpoint(base_output_dir)
    if not t1_ckpt.is_absolute():
        t1_ckpt = (script_dir / t1_ckpt).resolve()
    if not t1_ckpt.exists():
        raise FileNotFoundError(f"找不到 T1 checkpoint: {t1_ckpt}")

    debug_output_dir = Path(f"{base_output_dir}_{args.output_suffix}")
    cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', 'status3')}_{args.output_suffix}"
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
    print("[T2 Debug] 只从 T2 开始训练，不重跑 T1")
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
        raise ValueError("配置中至少需要两个 tasks，当前脚本固定调试 T2")

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
    print(f"[T2 Debug] 开始训练 T2: {t2_task['name']}")
    print(f"原始 data.yaml      : {t2_task['data']}")
    print(f"全局 data.yaml      : {prepared_data}")
    print(f"全局类别通道        : {current_indices}")
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
    print(f"[T2 Debug] T2 trained checkpoint: {trained_ckpt}")

    if args.skip_merge:
        latest_ckpt = trained_ckpt
        print("[T2 Debug] 已按 --skip-merge 跳过 DuET merge。")
    else:
        print("[T2 Debug] 执行 DuET merge，便于比较 trained 与 merged 后指标。")
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
        merged_state = merge_full_head_slices(
            merged_state,
            old_state,
            new_state,
            learned_indices=learned_indices,
            current_indices=current_indices,
        )
        latest_ckpt = debug_output_dir / f"task_2_{t2_task['name']}_duet.pt"
        inject_state_dict_into_checkpoint(trained_ckpt, merged_state, latest_ckpt)
        print(
            "[T2 Debug] merged_keys={0} skipped_keys={1}".format(
                report["merged_keys"],
                report["skipped_keys"],
            )
        )
        print(f"[T2 Debug] merged checkpoint: {latest_ckpt}")

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

    print("\n[T2 Debug] Done.")
    print(f"[T2 Debug] trained checkpoint: {trained_ckpt}")
    print(f"[T2 Debug] latest checkpoint : {latest_ckpt}")


if __name__ == "__main__":
    main()
