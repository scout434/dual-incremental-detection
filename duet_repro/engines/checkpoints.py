from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

from duet_repro.core.task_vectors import load_state_dict
from duet_repro.modeling.heads import is_class_output_key


def project_relative_value(value, *, project_root: Path):
    """把配置/历史记录里的路径转换成项目相对路径。

    训练输出会写入 JSON/YAML；如果里面保存绝对路径，换机器后就很难复现。
    这里递归处理 dict/list/Path/string，让 manifest 尽量可移植。
    """
    if isinstance(value, dict):
        return {key: project_relative_value(item, project_root=project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [project_relative_value(item, project_root=project_root) for item in value]
    if isinstance(value, tuple):
        return [project_relative_value(item, project_root=project_root) for item in value]
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        raw = value.replace("\\", "/")
        # 兼容历史配置里出现的 status*/output 或 output/status* 路径写法，
        # 统一折叠成 output/status*/...，方便 README 和评估脚本引用。
        for marker, prefix in (
            ("status1/output/", project_root / "output" / "status1"),
            ("status3/output/", project_root / "output" / "status3"),
            ("output/status1/", project_root / "output" / "status1"),
            ("output/status3/", project_root / "output" / "status3"),
        ):
            if marker in raw:
                return str(Path("output") / prefix.name / Path(raw.split(marker, 1)[1])).replace("\\", "/")
        path = Path(value)
    else:
        return value

    try:
        return str(path.resolve().relative_to(project_root)).replace("\\", "/")
    except (OSError, ValueError):
        return str(value)


def write_experiment_state(
    output_dir: Path,
    cfg: dict,
    history: list[dict],
    *,
    reference_ckpt: str | Path,
    latest_ckpt: str | Path | None,
) -> None:
    """写出一次训练的配置、历史和评估清单。

    training_history.json 记录每个任务的训练结果；eval_manifest.json 给评估脚本
    和人工检查使用；resolved_config.yaml 保存最终展开后的训练配置。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path.cwd().resolve()
    portable_history = project_relative_value(history, project_root=project_root)
    portable_cfg = project_relative_value(cfg, project_root=project_root)

    (output_dir / "training_history.json").write_text(
        json.dumps(portable_history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "reference_checkpoint": project_relative_value(reference_ckpt, project_root=project_root),
        "latest_checkpoint": project_relative_value(latest_ckpt, project_root=project_root)
        if latest_ckpt is not None
        else None,
        "output_dir": project_relative_value(output_dir, project_root=project_root),
        "detector": portable_cfg.get("detector", {}),
        "training": portable_cfg.get("training", {}),
        "duet": portable_cfg.get("duet", {}),
        "tasks": portable_cfg.get("tasks", []),
        "history": portable_history,
        "metric_note": {
            # 这里把论文指标口径直接写入 manifest，避免之后只看 JSON 时忘记含义。
            "Avg_RI": "Final-task mAP on old classes / mAP when those classes were first learned.",
            "Avg_GI": "Merged-checkpoint mAP on unseen class/domain slices / reference mAP for those unseen slices.",
            "RAI": "(Avg_RI + Avg_GI) / 2.",
        },
    }
    (output_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(portable_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def resolve_config_path(config_path: str | Path, *, script_dir: Path, project_root: Path) -> Path:
    """解析训练配置路径，兼容当前目录、脚本目录和项目根目录。"""
    path = Path(config_path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), script_dir, project_root):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (script_dir / path).resolve()


def resolve_project_path(path: str | Path, *, project_root: Path) -> Path:
    """解析项目内路径；不存在时仍返回项目根目录下的候选路径。"""
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), project_root):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (project_root / path).resolve()


def validate_resume_checkpoint(
    checkpoint: str | Path,
    learned_indices: Iterable[int],
    *,
    project_root: Path,
    total_classes: int | None = None,
    allow_full_head: bool = False,
) -> Path:
    """校验断点续训权重的检测头类别数是否和当前任务匹配。

    增量训练时最容易拿错 checkpoint：例如用 20 类完整 head 去续 10 类阶段，
    或反过来。这里读取 cv3 分类输出层行数，提前报错，比训练到中途炸掉更好。
    """
    checkpoint = resolve_project_path(checkpoint, project_root=project_root)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")

    expected_rows = len([int(index) for index in learned_indices])
    state = load_state_dict(checkpoint)
    class_output_shapes = {
        key: tuple(value.shape)
        for key, value in state.items()
        if is_class_output_key(key)
    }
    if not class_output_shapes:
        raise ValueError(f"Resume checkpoint has no YOLO cv3 classification outputs: {checkpoint}")

    allowed_rows = {expected_rows}
    if allow_full_head and total_classes is not None:
        allowed_rows.add(int(total_classes))
    unexpected = {
        key: shape
        for key, shape in class_output_shapes.items()
        if not shape or shape[0] not in allowed_rows
    }
    if unexpected:
        raise ValueError(
            f"Resume checkpoint must have one of {sorted(allowed_rows)} cumulative class rows, "
            f"but found incompatible outputs: {unexpected}"
        )

    actual_rows = sorted({shape[0] for shape in class_output_shapes.values()})
    print(f"[resume] validated checkpoint head rows={actual_rows}: {checkpoint}")
    return checkpoint
