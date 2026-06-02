from __future__ import annotations

"""Train status1 T2 with a cumulative 20-class head reused from a finished T1.

This is the route for the protocol:

1. T1 remains a clean 10-class checkpoint for classes 0..9.
2. Before T2, build a standard cumulative 20-class YOLO Detect head.
3. Copy T1 rows 0..9 into the expanded student head.
4. Train T2 on global rows 10..19 while distilling student rows 0..9 from T1.
5. After T2, run DuET on shared parameters and keep a single standard 20-class
   Detect head whose old rows come from T1 and new rows come from trained T2.
"""

import argparse
import multiprocessing
from pathlib import Path

from train_duet import main as train_cumulative_head


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs/train_pascal_2phase_full.yaml"
DEFAULT_T1 = (
    PROJECT_ROOT
    / "outputs/status1_pascal_2phase_local_head_duet_yolo11n/task_1_voc_1_10_local_best.pt"
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), SCRIPT_DIR, PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--t1-ckpt", default=str(DEFAULT_T1), help="Finished 10-class T1 checkpoint.")
    parser.add_argument("--output-suffix", default=None, help="Optional suffix appended to configured output_dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override T2 epochs.")
    parser.add_argument("--device", default=None, help="Override device, e.g. 0, 1, cpu.")
    parser.add_argument("--distill-weight", type=float, default=None, help="Override duet.distill_weight.")
    parser.add_argument("--dc-weight", type=float, default=None, help="Override duet.dc_weight.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    t1_ckpt = resolve_path(args.t1_ckpt)
    if not t1_ckpt.exists():
        raise FileNotFoundError(f"T1 checkpoint not found: {t1_ckpt}")

    output_dir = None
    if args.output_suffix:
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output_dir = Path(f"{cfg['experiment']['output_dir']}_{args.output_suffix}")

    training_overrides = {}
    if args.epochs is not None:
        training_overrides["epochs"] = int(args.epochs)
    if args.device is not None:
        training_overrides["device"] = args.device

    duet_overrides = {}
    if args.distill_weight is not None:
        duet_overrides["distill_weight"] = float(args.distill_weight)
    if args.dc_weight is not None:
        duet_overrides["dc_weight"] = float(args.dc_weight)

    print("[Status1 T2] cumulative 20-class head + Incremental Head row concat")
    print(f"config        : {config_path}")
    print(f"T1 checkpoint : {t1_ckpt}")
    print("T2 init       : expand to rows 0..19, restore old rows 0..9 from T1")
    print("T2 distill    : compare student old rows 0..9 against T1 teacher rows 0..9")
    print("final head    : standard YOLO Detect head, old rows from T1, new rows from T2")
    train_cumulative_head(
        config_path,
        start_task=2,
        previous_checkpoint=t1_ckpt,
        output_dir_override=output_dir,
        training_overrides=training_overrides,
        duet_overrides=duet_overrides,
    )


if __name__ == "__main__":
    main()
