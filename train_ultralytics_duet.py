from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from duet_repro.core.task_vectors import (
    inject_state_dict_into_checkpoint,
    load_state_dict,
    merge_state_dicts,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_one_task(weights: str | Path, task: dict, cfg: dict, project_dir: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Please install ultralytics first: pip install ultralytics") from exc

    training = cfg["training"]
    model = YOLO(str(weights))
    model.train(
        data=task["data"],
        epochs=training.get("epochs", 50),
        imgsz=training.get("imgsz", 640),
        batch=training.get("batch", 16),
        workers=training.get("workers", 4),
        device=training.get("device", 0),
        optimizer=training.get("optimizer", "auto"),
        lr0=training.get("lr0", 0.01),
        project=str(project_dir),
        name=task["name"],
        exist_ok=True,
    )
    best = Path(model.trainer.best)
    if not best.exists():
        raise FileNotFoundError(f"Ultralytics did not produce best checkpoint: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"].get("seed", 42)))

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = output_dir / "runs"

    base_weights = cfg["detector"]["base_weights"]
    duet_cfg = cfg.get("duet", {})
    tasks = cfg["tasks"]

    history: list[dict] = []
    reference_ckpt = base_weights
    old_ckpt: str | Path | None = None
    current_weights: str | Path = base_weights

    for task_index, task in enumerate(tasks, start=1):
        print(f"\n[DuET] Training task {task_index}/{len(tasks)}: {task['name']}")
        trained_ckpt = train_one_task(current_weights, task, cfg, project_dir)

        if task_index == 1 or not duet_cfg.get("enabled", True):
            merged_ckpt = output_dir / f"task_{task_index}_{task['name']}_best.pt"
            inject_state_dict_into_checkpoint(trained_ckpt, load_state_dict(trained_ckpt), merged_ckpt)
        else:
            if old_ckpt is None:
                raise RuntimeError("old_ckpt should be available after the first task")
            reference = load_state_dict(reference_ckpt)
            old_state = load_state_dict(old_ckpt)
            new_state = load_state_dict(trained_ckpt)
            merged_state, report = merge_state_dicts(
                reference,
                old_state,
                new_state,
                alpha_old=float(duet_cfg.get("alpha_old", 1.0)),
                alpha_new=float(duet_cfg.get("alpha_new", 1.0)),
                shared_key_exclude=duet_cfg.get("shared_key_exclude", []),
            )
            merged_ckpt = output_dir / f"task_{task_index}_{task['name']}_duet.pt"
            inject_state_dict_into_checkpoint(trained_ckpt, merged_state, merged_ckpt)
            print(
                "[DuET] merged_keys={0} skipped_keys={1} incompatible={2}".format(
                    report.merged_keys,
                    report.skipped_keys,
                    len(report.incompatible_keys),
                )
            )

        history.append(
            {
                "task": task["name"],
                "trained_checkpoint": str(trained_ckpt),
                "merged_checkpoint": str(merged_ckpt),
            }
        )
        old_ckpt = merged_ckpt
        current_weights = merged_ckpt

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[DuET] Done. History saved to {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()

