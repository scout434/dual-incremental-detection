from __future__ import annotations

"""
Paper-faithful DuET training with task-local detection heads.

The existing train_duet.py keeps a fixed global class head during every task.
This entry point instead trains each task with only its local classes:

  T1: nc = len(C1)
  T2: nc = len(C2)
  ...

After each incremental task, shared parameters are merged with DuET task
arithmetic and YOLO classification output rows are physically concatenated into
a cumulative Detect head. This prevents new-task BCE negatives from updating
old task classification rows during training.
"""

import argparse
import atexit
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if not (PROJECT_ROOT / "duet_repro").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from duet_repro.core.duet_loss import create_duet_criterion
from duet_repro.core.duet_module import merge_state_dicts_with_duet_module
from duet_repro.core.task_vectors import StateDict, load_state_dict, task_vector
from train_duet import (
    IMAGE_SUFFIXES,
    TeeStream,
    canonical_class_name,
    copy_class_output_rows,
    detect_label_space,
    extract_checkpoint_names,
    infer_label_path,
    iter_split_images,
    link_or_copy_image,
    load_config,
    normalize_names,
    read_label_classes,
    resolve_split_sources,
    safe_task_name,
    torch_load_checkpoint,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_SHM_DISABLED"] = "1"


def setup_console_txt_log(output_dir: Path) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_local_head_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def close_log_file() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    atexit.register(close_log_file)
    print(f"[log] console output is mirrored to: {log_path}")
    return log_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_class_output_key(key: str) -> bool:
    lowered = key.lower()
    return "cv3" in lowered and (lowered.endswith(".2.weight") or lowered.endswith(".2.bias"))


def _key_matches_any(key: str, patterns: Iterable[str]) -> bool:
    lowered = key.lower()
    return any(str(pattern).lower() in lowered for pattern in patterns)


def default_detect_head_patterns(state: StateDict | None = None) -> tuple[str, ...]:
    if not state:
        return ("model.23",)
    layer_ids: set[int] = set()
    for key in state:
        if not key.startswith("model."):
            continue
        parts = key.split(".")
        if len(parts) > 2 and parts[1].isdigit():
            layer_ids.add(int(parts[1]))
    return (f"model.{max(layer_ids)}",) if layer_ids else ("model.23",)


def default_class_output_exclude(state: StateDict) -> tuple[str, ...]:
    return tuple(key.rsplit(".", 1)[0] for key in state if _is_class_output_key(key))


def resolve_task_class_indices(task: dict[str, Any], data_cfg: dict[str, Any], cfg: dict[str, Any]) -> list[int]:
    configured = [int(index) for index in task["class_indices"]]
    source_names_raw = data_cfg.get("names")
    if source_names_raw is None:
        return configured

    source_names = normalize_names(source_names_raw)
    if len(source_names) != len(configured):
        return configured

    global_names = normalize_names(cfg["detector"]["names"])
    name_to_global = {name: index for index, name in global_names.items()}
    derived: list[int] = []
    for local_index in range(len(source_names)):
        name = source_names[local_index]
        if name not in name_to_global:
            return configured
        derived.append(name_to_global[name])
    if derived != configured:
        print(f"[data] {task['name']} class_indices corrected by data.yaml names: {derived}")
    return derived


def remap_label_file_to_local(src: Path, dst: Path, class_indices: list[int], label_space: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return

    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(class_indices)}
    remapped_lines: list[str] = []
    for line_number, line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        items = line.strip().split()
        if not items:
            continue
        class_id = int(float(items[0]))
        if label_space == "global":
            if class_id not in global_to_local:
                raise ValueError(f"{src}:{line_number} global class {class_id} is outside {class_indices}")
            class_id = global_to_local[class_id]
        elif class_id >= len(class_indices):
            raise ValueError(f"{src}:{line_number} local class {class_id} is outside task nc={len(class_indices)}")
        items[0] = str(class_id)
        remapped_lines.append(" ".join(items))
    dst.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")


def remap_label_file_to_output(
    src: Path,
    dst: Path,
    class_indices: list[int],
    global_to_output: dict[int, int],
    label_space: str,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return

    local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(class_indices)}
    remapped_lines: list[str] = []
    for line_number, line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        items = line.strip().split()
        if not items:
            continue
        raw_id = int(float(items[0]))
        global_id = raw_id if label_space == "global" else local_to_global.get(raw_id)
        if global_id is None or global_id not in global_to_output:
            raise ValueError(f"{src}:{line_number} cannot map class {raw_id} into cumulative output space")
        items[0] = str(global_to_output[global_id])
        remapped_lines.append(" ".join(items))
    dst.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")


def _collect_split_records(data_yaml: Path, data_cfg: dict[str, Any]) -> tuple[dict[str, list[tuple[Path, Path]]], set[int]]:
    split_records: dict[str, list[tuple[Path, Path]]] = {}
    label_classes: set[int] = set()
    for split in ("train", "val", "test"):
        records: list[tuple[Path, Path]] = []
        for source in resolve_split_sources(data_yaml, data_cfg, split):
            records.extend(iter_split_images(source))
        if records:
            split_records[split] = records
            for image_path, _ in records:
                label_classes.update(read_label_classes(infer_label_path(image_path)))
    if "train" not in split_records:
        raise ValueError(f"{data_yaml} has no usable train image source")
    return split_records, label_classes


def prepare_task_local_data(task: dict[str, Any], cfg: dict[str, Any], output_dir: Path) -> Path:
    original_yaml = Path(task["data"])
    if not original_yaml.is_absolute():
        original_yaml = PROJECT_ROOT / original_yaml
    original_yaml = original_yaml.resolve()
    data_cfg = load_config(original_yaml)

    class_indices = resolve_task_class_indices(task, data_cfg, cfg)
    total_classes = int(cfg["detector"]["total_classes"])
    global_names = normalize_names(cfg["detector"]["names"])
    local_names = {idx: global_names[global_idx] for idx, global_idx in enumerate(class_indices)}

    split_records, label_classes = _collect_split_records(original_yaml, data_cfg)
    label_space = detect_label_space(label_classes, class_indices, task.get("labels_are_global", "auto"), total_classes)

    task_root = output_dir / "prepared_local_data" / safe_task_name(task["name"])
    if task_root.exists():
        shutil.rmtree(task_root)
    task_root.mkdir(parents=True, exist_ok=True)

    prepared_cfg: dict[str, Any] = {
        "path": str(task_root.resolve()),
        "train": "images/train",
        "nc": len(class_indices),
        "names": local_names,
        "global_class_indices": class_indices,
    }
    for split, records in split_records.items():
        prepared_cfg[split] = f"images/{split}"
        for image_path, relative_path in records:
            dst_image = task_root / "images" / split / relative_path
            dst_label = task_root / "labels" / split / relative_path.with_suffix(".txt")
            link_or_copy_image(image_path, dst_image)
            remap_label_file_to_local(infer_label_path(image_path), dst_label, class_indices, label_space)

    prepared_yaml = task_root / "data.yaml"
    prepared_yaml.write_text(yaml.safe_dump(prepared_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    task["local_data"] = str(prepared_yaml)
    task["label_space"] = label_space
    task["resolved_class_indices"] = class_indices
    print(f"[data] {task['name']}: train local nc={len(class_indices)}, labels={label_space}, classes={class_indices}")
    return prepared_yaml


def prepare_task_cumulative_eval_data(
    task: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
    learned_order: list[int],
) -> Path:
    original_yaml = Path(task["data"])
    if not original_yaml.is_absolute():
        original_yaml = PROJECT_ROOT / original_yaml
    original_yaml = original_yaml.resolve()
    data_cfg = load_config(original_yaml)

    class_indices = [int(i) for i in task["resolved_class_indices"]]
    global_names = normalize_names(cfg["detector"]["names"])
    output_names = {idx: global_names[global_idx] for idx, global_idx in enumerate(learned_order)}
    global_to_output = {global_idx: idx for idx, global_idx in enumerate(learned_order)}

    split_records, label_classes = _collect_split_records(original_yaml, data_cfg)
    label_space = str(task.get("label_space") or detect_label_space(
        label_classes, class_indices, task.get("labels_are_global", "auto"), int(cfg["detector"]["total_classes"])
    ))

    task_root = output_dir / "prepared_cumulative_eval_data" / safe_task_name(task["name"])
    if task_root.exists():
        shutil.rmtree(task_root)
    task_root.mkdir(parents=True, exist_ok=True)

    prepared_cfg: dict[str, Any] = {
        "path": str(task_root.resolve()),
        "train": "images/train",
        "nc": len(learned_order),
        "names": output_names,
        "global_class_indices": learned_order,
    }
    for split, records in split_records.items():
        prepared_cfg[split] = f"images/{split}"
        for image_path, relative_path in records:
            dst_image = task_root / "images" / split / relative_path
            dst_label = task_root / "labels" / split / relative_path.with_suffix(".txt")
            link_or_copy_image(image_path, dst_image)
            remap_label_file_to_output(
                infer_label_path(image_path),
                dst_label,
                class_indices,
                global_to_output,
                label_space,
            )

    prepared_yaml = task_root / "data.yaml"
    prepared_yaml.write_text(yaml.safe_dump(prepared_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return prepared_yaml


def prepare_extra_cumulative_eval_slice(
    slice_cfg: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
    learned_order: list[int],
) -> Path:
    """Prepare an evaluation-only slice in the final cumulative output order."""
    original_yaml = Path(slice_cfg["data"])
    if not original_yaml.is_absolute():
        original_yaml = PROJECT_ROOT / original_yaml
    original_yaml = original_yaml.resolve()
    data_cfg = load_config(original_yaml)

    class_indices = resolve_task_class_indices(slice_cfg, data_cfg, cfg)
    global_names = normalize_names(cfg["detector"]["names"])
    output_names = {idx: global_names[global_idx] for idx, global_idx in enumerate(learned_order)}
    global_to_output = {global_idx: idx for idx, global_idx in enumerate(learned_order)}

    missing = sorted(set(class_indices) - set(learned_order))
    if missing:
        raise ValueError(
            f"eval slice {slice_cfg['name']} contains classes not present in final output order: {missing}"
        )

    split_records, label_classes = _collect_split_records(original_yaml, data_cfg)
    label_space = detect_label_space(
        label_classes,
        class_indices,
        slice_cfg.get("labels_are_global", "auto"),
        int(cfg["detector"]["total_classes"]),
    )

    slice_root = output_dir / "prepared_cumulative_eval_data" / safe_task_name(slice_cfg["name"])
    if slice_root.exists():
        shutil.rmtree(slice_root)
    slice_root.mkdir(parents=True, exist_ok=True)

    prepared_cfg: dict[str, Any] = {
        "path": str(slice_root.resolve()),
        "train": "images/train",
        "nc": len(learned_order),
        "names": output_names,
        "global_class_indices": learned_order,
    }
    for split, records in split_records.items():
        prepared_cfg[split] = f"images/{split}"
        for image_path, relative_path in records:
            dst_image = slice_root / "images" / split / relative_path
            dst_label = slice_root / "labels" / split / relative_path.with_suffix(".txt")
            link_or_copy_image(image_path, dst_image)
            remap_label_file_to_output(
                infer_label_path(image_path),
                dst_label,
                class_indices,
                global_to_output,
                label_space,
            )

    prepared_yaml = slice_root / "data.yaml"
    prepared_yaml.write_text(yaml.safe_dump(prepared_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[eval-data] {slice_cfg['name']}: {prepared_yaml}")
    return prepared_yaml


def build_model_with_nc(cfg: dict[str, Any], nc: int, names: dict[int, str]):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    model_yaml = str(cfg["detector"].get("model_yaml", "yolo11n.yaml"))
    model_cfg = yaml_model_load(model_yaml)
    model_cfg["nc"] = int(nc)

    model = YOLO(model_yaml)
    model.model = DetectionModel(model_cfg, nc=int(nc), verbose=False)
    model.model.names = names
    model.model.task = "detect"
    model.model.args = {"model": model_yaml, "task": "detect"}
    model.model.yaml["nc"] = int(nc)
    if hasattr(model.model, "nc"):
        model.model.nc = int(nc)
    if model.ckpt is None:
        model.ckpt = {}
    model.overrides.pop("nc", None)
    return model


def load_pretrained_rows_and_shared(model, weights: str | Path) -> None:
    ckpt = torch_load_checkpoint(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        official_state = ckpt["model"].state_dict()
    elif hasattr(ckpt, "state_dict"):
        official_state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        official_state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")

    model_state = model.model.state_dict()
    target_names = normalize_names(model.model.names)
    pretrained_names = extract_checkpoint_names(ckpt)
    copied_exact = 0
    copied_rows = 0

    for key in list(model_state):
        if key not in official_state:
            continue
        pretrained_value = official_state[key]
        if model_state[key].shape == pretrained_value.shape:
            model_state[key] = pretrained_value.clone()
            copied_exact += 1
        elif _is_class_output_key(key) and pretrained_names:
            copied_rows += copy_class_output_rows(model_state[key], pretrained_value, target_names, pretrained_names)

    model.model.load_state_dict(model_state, strict=False)
    print(f"[init] loaded exact tensors={copied_exact}, class rows={copied_rows} from {weights}")


def save_base_local_checkpoint(cfg: dict[str, Any], output_dir: Path, nc: int, names: dict[int, str], tag: str) -> Path:
    path = output_dir / "local_references" / f"{safe_task_name(tag)}_nc{nc}.pt"
    if path.exists():
        return path
    model = build_model_with_nc(cfg, nc, names)
    load_pretrained_rows_and_shared(model, cfg["detector"]["base_weights"])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def make_task_init_checkpoint(
    cfg: dict[str, Any],
    task: dict[str, Any],
    output_dir: Path,
    *,
    previous_shared_state: StateDict | None,
    init_exclude: Iterable[str],
) -> Path:
    local_names = normalize_names(load_config(task["local_data"])["names"])
    nc = len(local_names)
    base_ckpt = save_base_local_checkpoint(cfg, output_dir, nc, local_names, task["name"])

    if previous_shared_state is None:
        return base_ckpt

    from ultralytics import YOLO

    model = YOLO(str(base_ckpt))
    model_state = model.model.state_dict()
    loaded = 0
    for key, value in model_state.items():
        source_value = previous_shared_state.get(key)
        if source_value is None:
            continue
        if value.shape != source_value.shape:
            continue
        if _key_matches_any(key, init_exclude):
            continue
        model_state[key] = source_value.detach().clone().to(dtype=value.dtype)
        loaded += 1
    model.model.load_state_dict(model_state, strict=False)

    init_path = output_dir / "task_initializers" / f"{safe_task_name(task['name'])}_init.pt"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(init_path)
    print(f"[init] {task['name']}: loaded {loaded} shared tensors from previous merged model")
    return init_path


def train_one_local_task(
    init_ckpt: str | Path,
    task: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    is_first: bool,
    reference_state: StateDict,
    task_vector_history: list[StateDict],
    shared_key_exclude: Iterable[str],
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(init_ckpt))
    training = cfg["training"]
    duet_cfg = cfg.get("duet", {})
    dc_weight = float(duet_cfg.get("dc_weight", 0.0)) if not is_first else 0.0
    requested_distill = float(duet_cfg.get("distill_weight", 0.0)) if not is_first else 0.0
    if requested_distill > 0:
        print(
            "[loss] local-head mode disables class/logit distillation by default because "
            "current-task local classes are semantically different from old-task local classes."
        )

    if dc_weight > 0 and task_vector_history:
        criterion = create_duet_criterion(
            model=model.model,
            teacher_model=None,
            distill_weight=0.0,
            dc_weight=dc_weight,
            reference_state=reference_state,
            shared_key_exclude=shared_key_exclude,
        )
        for vector in task_vector_history:
            criterion.record_task_vector(vector)

        def on_train_start(trainer):
            from ultralytics.utils.torch_utils import unwrap_model

            original_model = unwrap_model(trainer.model)
            if hasattr(original_model, "args"):
                criterion.hyp = original_model.args
            original_model.criterion = criterion
            if trainer.ema and hasattr(trainer.ema, "ema"):
                unwrap_model(trainer.ema.ema).criterion = criterion
            trainer.loss_names = list(trainer.loss_names) + ["dc_loss"]

        model.add_callback("on_train_start", on_train_start)

    model.train(
        data=str(task["local_data"]),
        epochs=training.get("epochs", 30),
        warmup_epochs=training.get("warmup_epochs", 5),
        batch=training.get("batch", 64),
        imgsz=training.get("imgsz", 640),
        device=training.get("device", 0),
        optimizer=training.get("optimizer", "AdamW"),
        cos_lr=training.get("cos_lr", True),
        lr0=training.get("lr0", 0.01),
        lrf=training.get("lrf", 0.01),
        weight_decay=training.get("weight_decay_first" if is_first else "weight_decay_incremental", 0.0005),
        freeze=training.get("freeze", 0),
        workers=training.get("workers", 0),
        project=str(output_dir / "runs"),
        name=task["name"],
        exist_ok=True,
    )
    best = Path(model.trainer.best)
    if not best.exists():
        raise FileNotFoundError(f"Ultralytics did not produce best checkpoint: {best}")
    return best


def concat_class_output_rows(
    target_state: StateDict,
    old_state: StateDict,
    new_state: StateDict,
    *,
    old_nc: int,
    new_nc: int,
) -> None:
    for key, target_value in list(target_state.items()):
        if not _is_class_output_key(key):
            continue
        old_value = old_state.get(key)
        new_value = new_state.get(key)
        if old_value is None or new_value is None:
            continue
        if old_value.shape[0] != old_nc or new_value.shape[0] != new_nc:
            continue
        if target_value.shape[0] != old_nc + new_nc:
            continue
        if old_value.dim() == new_value.dim() == target_value.dim() == 4:
            target_state[key] = torch.cat([old_value.detach(), new_value.detach()], dim=0).to(dtype=target_value.dtype)
        elif old_value.dim() == new_value.dim() == target_value.dim() == 1:
            target_state[key] = torch.cat([old_value.detach(), new_value.detach()], dim=0).to(dtype=target_value.dtype)


def save_cumulative_checkpoint(
    cfg: dict[str, Any],
    output_path: Path,
    *,
    cumulative_names: dict[int, str],
    merged_state: StateDict,
    old_state: StateDict | None,
    new_state: StateDict,
    old_nc: int,
    new_nc: int,
) -> StateDict:
    model = build_model_with_nc(cfg, len(cumulative_names), cumulative_names)
    load_pretrained_rows_and_shared(model, cfg["detector"]["base_weights"])
    target_state = model.model.state_dict()

    loaded = 0
    for key, value in list(target_state.items()):
        merged_value = merged_state.get(key)
        if merged_value is None or merged_value.shape != value.shape:
            continue
        target_state[key] = merged_value.detach().clone().to(dtype=value.dtype)
        loaded += 1

    if old_state is not None:
        concat_class_output_rows(target_state, old_state, new_state, old_nc=old_nc, new_nc=new_nc)

    model.model.load_state_dict(target_state, strict=False)
    model.model.names = cumulative_names
    if hasattr(model.model, "nc"):
        model.model.nc = len(cumulative_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"[merge] saved cumulative nc={len(cumulative_names)} checkpoint, loaded tensors={loaded}: {output_path}")
    return model.model.state_dict()


def write_experiment_state(
    output_dir: Path,
    cfg: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    reference_ckpt: str | Path,
    latest_ckpt: str | Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "mode": "local_head_physical_concat",
        "reference_checkpoint": str(reference_ckpt),
        "latest_checkpoint": str(latest_ckpt) if latest_ckpt is not None else None,
        "output_dir": str(output_dir),
        "detector": cfg.get("detector", {}),
        "training": cfg.get("training", {}),
        "duet": cfg.get("duet", {}),
        "tasks": cfg.get("tasks", []),
        "history": history,
        "metric_note": {
            "train_data": "Task-local nc data used for training the local head.",
            "cumulative_eval_data": "Labels remapped to the cumulative output order for merged checkpoints.",
        },
    }
    (output_dir / "eval_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, PROJECT_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (SCRIPT_DIR / path).resolve()


def main(config_path: str = "configs/train_local_head_pascal_2phase.yaml") -> None:
    multiprocessing.freeze_support()
    config_path = resolve_config_path(config_path)
    os.chdir(PROJECT_ROOT)

    cfg = load_config(config_path)
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_console_txt_log(output_dir)

    print("=" * 72)
    print("[local-head DuET] task-local training + physical class-row concat")
    for idx, task in enumerate(cfg["tasks"], start=1):
        print(f"T{idx}: {task['name']} | classes={task['class_indices']} | data={task['data']}")
    print("=" * 72)

    # Build one reference with the first task nc. Shared tensors are shape-compatible
    # across all local heads and cumulative heads for YOLO11n.
    first_task = cfg["tasks"][0]
    first_data_cfg = load_config(first_task["data"] if Path(first_task["data"]).is_absolute() else PROJECT_ROOT / first_task["data"])
    first_indices = resolve_task_class_indices(first_task, first_data_cfg, cfg)
    global_names = normalize_names(cfg["detector"]["names"])
    first_names = {idx: global_names[global_idx] for idx, global_idx in enumerate(first_indices)}
    reference_ckpt = save_base_local_checkpoint(cfg, output_dir, len(first_indices), first_names, "shared_reference")
    reference_state = load_state_dict(reference_ckpt)

    duet_cfg = cfg.get("duet", {})
    merge_exclude = tuple(duet_cfg.get("shared_key_exclude") or default_class_output_exclude(reference_state))
    init_exclude = tuple(duet_cfg.get("local_head_init_exclude") or default_detect_head_patterns(reference_state))
    print(f"[config] merge shared_key_exclude={merge_exclude}")
    print(f"[config] local_head_init_exclude={init_exclude}")

    history: list[dict[str, Any]] = []
    task_vector_history: list[StateDict] = []
    learned_order: list[int] = []
    previous_shared_state: StateDict | None = None
    old_ckpt: Path | None = None
    old_nc = 0
    latest_ckpt: Path | None = None

    write_experiment_state(output_dir, cfg, history, reference_ckpt=reference_ckpt, latest_ckpt=latest_ckpt)

    for task_index, task in enumerate(cfg["tasks"], start=1):
        prepare_task_local_data(task, cfg, output_dir)
        current_indices = [int(i) for i in task["resolved_class_indices"]]
        current_names = {idx: global_names[global_idx] for idx, global_idx in enumerate(current_indices)}
        init_ckpt = make_task_init_checkpoint(
            cfg,
            task,
            output_dir,
            previous_shared_state=previous_shared_state,
            init_exclude=init_exclude,
        )

        print("\n" + "=" * 72)
        print(f"[T{task_index}] train local head: {task['name']}")
        print(f"local data: {task['local_data']}")
        print(f"local nc/classes: {len(current_indices)} / {current_indices}")
        print("=" * 72)

        trained_ckpt = train_one_local_task(
            init_ckpt,
            task,
            cfg,
            output_dir,
            is_first=task_index == 1,
            reference_state=reference_state,
            task_vector_history=task_vector_history,
            shared_key_exclude=merge_exclude,
        )

        new_state = load_state_dict(trained_ckpt)
        learned_order.extend(current_indices)
        cumulative_names = {idx: global_names[global_idx] for idx, global_idx in enumerate(learned_order)}
        cumulative_eval_data = prepare_task_cumulative_eval_data(task, cfg, output_dir, learned_order)

        if task_index == 1 or not duet_cfg.get("enabled", True):
            merged_state = new_state
            merged_ckpt = output_dir / f"task_{task_index}_{task['name']}_local_best.pt"
            final_state = save_cumulative_checkpoint(
                cfg,
                merged_ckpt,
                cumulative_names=cumulative_names,
                merged_state=merged_state,
                old_state=None,
                new_state=new_state,
                old_nc=0,
                new_nc=len(current_indices),
            )
        else:
            assert previous_shared_state is not None and old_ckpt is not None
            old_state = load_state_dict(old_ckpt)
            merged_state, report = merge_state_dicts_with_duet_module(
                reference_state,
                old_state,
                new_state,
                gamma=float(duet_cfg.get("gamma", 0.1)),
                alpha_base=float(duet_cfg.get("alpha_base", 0.5)),
                shared_key_exclude=merge_exclude,
                per_layer_report=duet_cfg.get("verbose_merge", False),
            )
            print(f"[DuET] merged shared tensors={report['merged_keys']}, skipped={report['skipped_keys']}")
            merged_ckpt = output_dir / f"task_{task_index}_{task['name']}_duet_local_concat.pt"
            final_state = save_cumulative_checkpoint(
                cfg,
                merged_ckpt,
                cumulative_names=cumulative_names,
                merged_state=merged_state,
                old_state=old_state,
                new_state=new_state,
                old_nc=old_nc,
                new_nc=len(current_indices),
            )

        current_tv = task_vector(reference_state, final_state, shared_key_exclude=merge_exclude)
        task_vector_history.append(current_tv)
        previous_shared_state = final_state
        old_ckpt = merged_ckpt
        old_nc = len(learned_order)
        latest_ckpt = merged_ckpt

        history.append(
            {
                "task_index": task_index,
                "task": task["name"],
                "class_indices": current_indices,
                "learned_output_order": learned_order.copy(),
                "local_data": task["local_data"],
                "cumulative_eval_data": str(cumulative_eval_data),
                "label_space": task["label_space"],
                "init_checkpoint": str(init_ckpt),
                "trained_checkpoint": str(trained_ckpt),
                "merged_checkpoint": str(merged_ckpt),
                "is_first_task": task_index == 1,
            }
        )
        write_experiment_state(output_dir, cfg, history, reference_ckpt=reference_ckpt, latest_ckpt=latest_ckpt)
        print(f"[T{task_index}] done: {merged_ckpt}")

    extra_eval_slices: list[dict[str, Any]] = []
    for slice_cfg in cfg.get("eval_slices", []):
        prepared = prepare_extra_cumulative_eval_slice(slice_cfg, cfg, output_dir, learned_order)
        extra_eval_slices.append(
            {
                "name": slice_cfg["name"],
                "data": slice_cfg["data"],
                "class_indices": [int(i) for i in slice_cfg["class_indices"]],
                "cumulative_eval_data": str(prepared),
            }
        )
    if extra_eval_slices:
        (output_dir / "extra_eval_slices.json").write_text(
            json.dumps(extra_eval_slices, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[eval-data] extra eval slice manifest: {output_dir / 'extra_eval_slices.json'}")

    print(f"\n[local-head DuET] done. Final checkpoint: {latest_ckpt}")


def shape_check(config_path: str = "configs/train_local_head_pascal_2phase.yaml") -> None:
    """Build two local heads and a cumulative head without running training."""
    config_path = resolve_config_path(config_path)
    os.chdir(PROJECT_ROOT)
    cfg = load_config(config_path)
    output_dir = Path(cfg["experiment"]["output_dir"]) / "_shape_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    global_names = normalize_names(cfg["detector"]["names"])

    if len(cfg["tasks"]) < 2:
        raise ValueError("shape_check needs at least two tasks.")

    task1_indices = [int(i) for i in cfg["tasks"][0]["class_indices"]]
    task2_indices = [int(i) for i in cfg["tasks"][1]["class_indices"]]
    names1 = {i: global_names[idx] for i, idx in enumerate(task1_indices)}
    names2 = {i: global_names[idx] for i, idx in enumerate(task2_indices)}

    model1 = build_model_with_nc(cfg, len(task1_indices), names1)
    load_pretrained_rows_and_shared(model1, cfg["detector"]["base_weights"])
    model2 = build_model_with_nc(cfg, len(task2_indices), names2)
    load_pretrained_rows_and_shared(model2, cfg["detector"]["base_weights"])
    state1 = model1.model.state_dict()
    state2 = model2.model.state_dict()

    duet_cfg = cfg.get("duet", {})
    exclude = tuple(duet_cfg.get("shared_key_exclude") or default_class_output_exclude(state1))
    merged, report = merge_state_dicts_with_duet_module(
        state1,
        state1,
        state2,
        gamma=float(duet_cfg.get("gamma", 0.1)),
        alpha_base=float(duet_cfg.get("alpha_base", 0.5)),
        shared_key_exclude=exclude,
    )
    learned = task1_indices + task2_indices
    cumulative_names = {i: global_names[idx] for i, idx in enumerate(learned)}
    final_state = save_cumulative_checkpoint(
        cfg,
        output_dir / "shape_check_concat.pt",
        cumulative_names=cumulative_names,
        merged_state=merged,
        old_state=state1,
        new_state=state2,
        old_nc=len(task1_indices),
        new_nc=len(task2_indices),
    )
    class_output_shapes = {
        key: tuple(value.shape)
        for key, value in final_state.items()
        if _is_class_output_key(key)
    }
    print(f"[shape-check] merged shared tensors={report['merged_keys']}")
    print(f"[shape-check] class output shapes={class_output_shapes}")
    print(f"[shape-check] saved={output_dir / 'shape_check_concat.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_local_head_pascal_2phase.yaml")
    parser.add_argument("--shape-check", action="store_true", help="only verify local-head concat shapes; do not train")
    args = parser.parse_args()
    if args.shape_check:
        shape_check(args.config)
    else:
        main(args.config)
