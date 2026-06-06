from __future__ import annotations

"""
DuET 论文指标评估脚本。

这个脚本用来计算 DuET 论文中的三个核心指标：
  Avg RI = mean(最终模型在旧任务上的 mAP / 该任务首次学习完成时的 mAP)
  Avg GI = mean(最终模型在未见类别或未见域上的 mAP / 对应参考模型的 mAP)
  RAI    = (Avg RI + Avg GI) / 2

脚本采用“评估计划文件”驱动，因为论文 Table S1 里的 old/new/unseen
评估切片需要你明确指定。训练脚本可以保存每个阶段的权重，但它无法自动知道
每个 unseen 切片的 data.yaml，也无法自动知道单独训练参考模型得到的 mAP。
"""

import argparse
import json
import os
import shutil
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
EVAL_DATA_CACHE: dict[str, Path] = {}
STATUS3_GLOBAL_NAMES = {idx: f"class_{idx + 1}" for idx in range(7)}


def normalize_project_path(value: str | Path | None) -> str | None:
    """Map old manifest paths to the current project layout."""
    if value is None:
        return None
    raw = str(value).replace("\\", "/")
    for marker in (
        "status1/output/",
        "status3/output/",
        "output/status1/",
        "output/status3/",
        "data/status1/",
        "data/status3/",
    ):
        if marker not in raw:
            continue
        suffix = raw.split(marker, 1)[1]
        if marker == "status1/output/":
            return str(PROJECT_ROOT / "output" / "status1" / Path(suffix))
        if marker == "status3/output/":
            return str(PROJECT_ROOT / "output" / "status3" / Path(suffix))
        if marker == "output/status1/":
            return str(PROJECT_ROOT / "output" / "status1" / Path(suffix))
        if marker == "output/status3/":
            return str(PROJECT_ROOT / "output" / "status3" / Path(suffix))
        if marker == "data/status1/":
            return str(PROJECT_ROOT / "data" / "status1" / Path(suffix))
        if marker == "data/status3/":
            return str(PROJECT_ROOT / "data" / "status3" / Path(suffix))

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def resolve_data_yaml_path(data_yaml: str | Path) -> Path:
    """Resolve a data.yaml path using cwd, status dir, and project root."""
    path = Path(data_yaml)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, PROJECT_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_dataset_root(data_yaml: Path, cfg: dict[str, Any]) -> Path:
    """Resolve YOLO dataset root."""
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_sources(data_yaml: Path, cfg: dict[str, Any], split: str) -> list[Path]:
    """Resolve a split entry to one or more filesystem paths."""
    value = cfg.get(split)
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    root = resolve_dataset_root(data_yaml, cfg)
    sources = []
    for item in values:
        source = Path(str(item))
        if not source.is_absolute():
            source = root / source
        sources.append(source.resolve())
    return sources


def iter_images(source: Path) -> list[tuple[Path, Path]]:
    """Return image path and relative path for a directory or txt image list."""
    if source.is_dir():
        images = [path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        return [(path, path.relative_to(source)) for path in sorted(images)]
    if source.is_file():
        records = []
        for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            raw = line.strip()
            if not raw:
                continue
            image_path = Path(raw)
            if not image_path.is_absolute():
                image_path = source.parent / image_path
            records.append((image_path, Path(f"{index:08d}_{image_path.name}")))
        return records
    return []


def infer_label_path(image_path: Path) -> Path:
    """Infer a YOLO label path from an image path."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def link_or_copy_image(src: Path, dst: Path) -> None:
    """Reuse image files with symlinks when possible, copy otherwise."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_local_label_to_global(src: Path, dst: Path, global_indices: list[int]) -> None:
    """Write a label file whose local ids are remapped to global ids."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return
    lines = []
    for line_number, line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        local_id = int(float(parts[0]))
        if local_id < 0 or local_id >= len(global_indices):
            raise ValueError(f"标签 {src}:{line_number} 的局部类别 {local_id} 超出 global_class_indices")
        parts[0] = str(int(global_indices[local_id]))
        lines.append(" ".join(parts))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_eval_data(data_yaml: str | Path, global_indices_override: list[int] | None = None) -> Path:
    """
    Convert local-label weather eval datasets to temporary global-label data.

    YOLO.val does not know train_duet.py's class_indices mapping. If data.yaml
    declares global_class_indices, remap labels before validation.
    """
    data_yaml_path = resolve_data_yaml_path(data_yaml)
    cache_key = str(data_yaml_path) + "::" + ",".join(str(i) for i in (global_indices_override or []))
    if cache_key in EVAL_DATA_CACHE:
        return EVAL_DATA_CACHE[cache_key]

    cfg = load_mapping(data_yaml_path)
    global_indices = global_indices_override or cfg.get("global_class_indices")
    if not global_indices:
        EVAL_DATA_CACHE[cache_key] = data_yaml_path
        return data_yaml_path

    global_indices = [int(index) for index in global_indices]
    eval_root = PROJECT_ROOT / "output" / "_eval_global_data" / data_yaml_path.parent.name
    if eval_root.exists():
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    prepared_cfg: dict[str, Any] = {
        "path": str(eval_root.resolve()),
        "nc": len(STATUS3_GLOBAL_NAMES),
        "names": STATUS3_GLOBAL_NAMES,
    }
    for split in ("train", "val", "test"):
        records: list[tuple[Path, Path]] = []
        for source in resolve_split_sources(data_yaml_path, cfg, split):
            records.extend(iter_images(source))
        if not records:
            continue
        prepared_cfg[split] = f"images/{split}"
        for image_path, relative_path in records:
            dst_image = eval_root / "images" / split / relative_path
            dst_label = eval_root / "labels" / split / relative_path.with_suffix(".txt")
            link_or_copy_image(image_path, dst_image)
            remap_local_label_to_global(infer_label_path(image_path), dst_label, global_indices)

    prepared_yaml = eval_root / "data.yaml"
    prepared_yaml.write_text(yaml.safe_dump(prepared_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    EVAL_DATA_CACHE[cache_key] = prepared_yaml
    return prepared_yaml


def resolve_plan_path(plan_path: str | Path) -> Path:
    """优先按当前工作目录、status2 目录、项目根目录依次解析评估计划路径。"""
    path = Path(plan_path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, PROJECT_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (SCRIPT_DIR / path).resolve()


def load_mapping(path: str | Path) -> dict[str, Any]:
    """读取 JSON 或 YAML 文件，并转换成字典。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def build_checkpoint_aliases(manifest: dict[str, Any] | None, plan: dict[str, Any]) -> dict[str, str]:
    """
    构造权重路径别名，方便在评估计划文件中直接引用。

    从 eval_manifest.json 自动生成的内置别名包括：
      reference -> 训练前保存的完整检测头参考权重
      latest/final -> 最后一个任务融合后的权重
      t1/task_1/<task_name> -> 第 1 个任务融合后的权重
      t1_trained/task_1_trained -> 第 1 个任务直接训练得到的 best.pt
    """
    aliases: dict[str, str] = {}

    if manifest:
        reference = manifest.get("reference_checkpoint")
        latest = manifest.get("latest_checkpoint")
        if reference:
            aliases["reference"] = str(normalize_project_path(reference))
        if latest:
            latest = str(normalize_project_path(latest))
            aliases["latest"] = latest
            aliases["final"] = latest

        for entry in manifest.get("history", []):
            idx = int(entry.get("task_index", len(aliases) + 1))
            task_name = str(entry.get("task", f"task_{idx}"))
            merged = entry.get("merged_checkpoint")
            trained = entry.get("trained_checkpoint")
            if merged:
                merged = str(normalize_project_path(merged))
                aliases[f"t{idx}"] = merged
                aliases[f"task_{idx}"] = merged
                aliases[task_name] = merged
            if trained:
                trained = str(normalize_project_path(trained))
                aliases[f"t{idx}_trained"] = trained
                aliases[f"task_{idx}_trained"] = trained
                aliases[f"{task_name}_trained"] = trained

    aliases.update({str(k): str(normalize_project_path(v)) for k, v in plan.get("checkpoint_aliases", {}).items()})
    return aliases


def resolve_checkpoint(value: str | Path, aliases: dict[str, str]) -> str:
    """把权重别名解析成真实路径；如果不是别名，就原样当成路径返回。"""
    raw = str(value)
    return aliases.get(raw, str(normalize_project_path(raw)))


def extract_map(metrics: Any, metric_name: str) -> float:
    """从 Ultralytics 的 DetMetrics 结果中提取 map50 或 map50_95。"""
    box = getattr(metrics, "box", metrics)
    if metric_name in {"map50", "mAP50", "mAP@0.5"}:
        return float(box.map50)
    if metric_name in {"map", "map50_95", "mAP50-95", "mAP@0.5:0.95"}:
        return float(box.map)
    raise ValueError(f"不支持指标 '{metric_name}'。请使用 map50 或 map50_95。")


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_yaml: str | Path,
    *,
    metric_name: str,
    device: str | int,
    imgsz: int | None = None,
    split: str = "val",
    global_indices: list[int] | None = None,
    prepare_global_labels: bool = True,
    cache: dict[tuple[str, str, str, str], float],
) -> float:
    """运行 YOLO 验证流程，并返回指定的 mAP 数值。"""
    checkpoint = str(checkpoint)
    if prepare_global_labels:
        data_yaml = str(prepare_eval_data(data_yaml, global_indices))
    else:
        data_yaml = str(resolve_data_yaml_path(data_yaml))
    key = (checkpoint, data_yaml, metric_name, str(imgsz or "default"))
    if key in cache:
        return cache[key]

    from ultralytics import YOLO

    model = YOLO(checkpoint)
    kwargs: dict[str, Any] = {
        "data": data_yaml,
        "device": device,
        "verbose": False,
        "plots": False,
        "split": split,
    }
    if imgsz is not None:
        kwargs["imgsz"] = imgsz

    metrics = model.val(**kwargs)
    value = extract_map(metrics, metric_name)
    cache[key] = value
    return value


def resolve_value(
    spec: dict[str, Any] | float | int,
    aliases: dict[str, str],
    *,
    metric_name: str,
    device: str | int,
    imgsz: int | None,
    split: str,
    cache: dict[tuple[str, str, str, str], float],
) -> dict[str, Any]:
    """
    解析一个指标项。

    一个指标项可以写成两种形式：
      value: 直接填写已经算好的 mAP，例如 {"value": 0.49}
      checkpoint + data: 让脚本自动运行 YOLO val，例如 {"checkpoint": "final", "data": ".../data.yaml"}
    """
    if isinstance(spec, (int, float)):
        return {"value": float(spec), "source": "literal"}

    if "value" in spec and spec["value"] is not None:
        return {"value": float(spec["value"]), "source": "literal"}

    checkpoint = spec.get("checkpoint")
    data_yaml = spec.get("data")
    if not checkpoint or not data_yaml:
        raise ValueError(f"指标项必须包含 value，或者同时包含 checkpoint 和 data：{spec}")

    resolved_checkpoint = resolve_checkpoint(checkpoint, aliases)
    value = evaluate_checkpoint(
        resolved_checkpoint,
        data_yaml,
        metric_name=metric_name,
        device=device,
        imgsz=imgsz,
        split=spec.get("split", split),
        global_indices=spec.get("global_class_indices"),
        prepare_global_labels=bool(spec.get("prepare_global_labels", True)),
        cache=cache,
    )
    return {
        "value": value,
        "checkpoint": resolved_checkpoint,
        "data": str(data_yaml),
        "source": "val",
    }


def compute_ratio(numerator: float, denominator: float, *, clamp_to_one: bool) -> float:
    """计算论文中的比例指标。默认不截断到 1，因为论文公式本身没有裁剪。"""
    if denominator <= 1e-12:
        return 0.0
    ratio = numerator / denominator
    return min(ratio, 1.0) if clamp_to_one else ratio


def evaluate_section(
    items: list[dict[str, Any]],
    aliases: dict[str, str],
    *,
    metric_name: str,
    device: str | int,
    imgsz: int | None,
    split: str,
    clamp_to_one: bool,
    cache: dict[tuple[str, str, str, str], float],
) -> tuple[list[float], list[dict[str, Any]]]:
    """评估一个指标分组，例如 retention 或 generalization。"""
    ratios: list[float] = []
    details: list[dict[str, Any]] = []

    for item in items:
        name = str(item.get("name", f"item_{len(details) + 1}"))
        numerator = resolve_value(
            item["numerator"],
            aliases,
            metric_name=metric_name,
            device=device,
            imgsz=imgsz,
            split=split,
            cache=cache,
        )
        denominator = resolve_value(
            item["denominator"],
            aliases,
            metric_name=metric_name,
            device=device,
            imgsz=imgsz,
            split=split,
            cache=cache,
        )
        ratio = compute_ratio(
            numerator["value"],
            denominator["value"],
            clamp_to_one=bool(item.get("clamp_to_one", clamp_to_one)),
        )
        ratios.append(ratio)
        details.append(
            {
                "name": name,
                "ratio": ratio,
                "ratio_percent": ratio * 100.0,
                "numerator": numerator,
                "denominator": denominator,
                "note": item.get("note", ""),
            }
        )

    return ratios, details


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 DuET 论文指标：Avg RI、Avg GI、RAI。")
    parser.add_argument("--plan", required=True, type=Path, help="指标评估计划文件，支持 YAML 或 JSON。")
    parser.add_argument("--output", type=Path, help="结果 JSON 保存路径；填写后会覆盖 plan.output。")
    parser.add_argument("--device", default=None, help="覆盖评估设备，例如 0 或 cpu。")
    args = parser.parse_args()

    args.plan = resolve_plan_path(args.plan)
    os.chdir(PROJECT_ROOT)

    plan = load_mapping(args.plan)
    manifest = None
    if plan.get("manifest"):
        manifest = load_mapping(plan["manifest"])

    aliases = build_checkpoint_aliases(manifest, plan)

    metric_name = str(plan.get("metric", "map50"))
    device = args.device if args.device is not None else plan.get("device", 0)
    imgsz = plan.get("imgsz")
    split = str(plan.get("split", "val"))
    clamp_to_one = bool(plan.get("clamp_to_one", False))
    cache: dict[tuple[str, str, str, str], float] = {}

    retention, retention_details = evaluate_section(
        plan.get("retention", []),
        aliases,
        metric_name=metric_name,
        device=device,
        imgsz=imgsz,
        split=split,
        clamp_to_one=clamp_to_one,
        cache=cache,
    )
    generalization, generalization_details = evaluate_section(
        plan.get("generalization", []),
        aliases,
        metric_name=metric_name,
        device=device,
        imgsz=imgsz,
        split=split,
        clamp_to_one=clamp_to_one,
        cache=cache,
    )

    avg_ri = sum(retention) / len(retention) if retention else 0.0
    avg_gi = sum(generalization) / len(generalization) if generalization else 0.0
    rai = (avg_ri + avg_gi) / 2.0

    result = {
        "metric": metric_name,
        "clamp_to_one": clamp_to_one,
        "avg_ri": avg_ri,
        "avg_gi": avg_gi,
        "rai": rai,
        "avg_ri_percent": avg_ri * 100.0,
        "avg_gi_percent": avg_gi * 100.0,
        "rai_percent": rai * 100.0,
        "retention": retention,
        "generalization": generalization,
        "retention_details": retention_details,
        "generalization_details": generalization_details,
        "checkpoint_aliases": aliases,
    }

    output = args.output or Path(plan.get("output", "paper_metrics_results.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    rai_payload = {"retention": retention, "generalization": generalization}
    (output.parent / "rai_metrics.json").write_text(
        json.dumps(rai_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n论文指标结果")
    print(f"Avg RI: {avg_ri:.4f} ({avg_ri * 100.0:.2f}%)")
    print(f"Avg GI: {avg_gi:.4f} ({avg_gi * 100.0:.2f}%)")
    print(f"RAI:    {rai:.4f} ({rai * 100.0:.2f}%)")
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()
