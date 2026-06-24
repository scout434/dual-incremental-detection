from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def load_config(path: str | Path) -> dict:
    """读取 YOLO data.yaml 配置。"""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_names(names: dict | list) -> dict[int, str]:
    """把 names 的 list/dict 两种写法统一成 {类别编号: 类别名}。"""
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def safe_task_name(name: str) -> str:
    """把任务名转换成安全的目录名。"""
    keep = []
    for char in str(name):
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(keep).strip("_") or "task"


def resolve_dataset_root(data_yaml: Path, data_cfg: dict) -> Path:
    """解析 data.yaml 中的 path 字段，得到数据集根目录。"""
    root = Path(data_cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_sources(data_yaml: Path, data_cfg: dict, split: str) -> list[Path]:
    """解析 train/val/test 字段为图片目录或图片列表文件路径。"""
    value = data_cfg.get(split)
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    root = resolve_dataset_root(data_yaml, data_cfg)
    sources = []
    for item in values:
        source = Path(str(item))
        if not source.is_absolute():
            source = root / source
        # 保留切片目录本身，不 resolve 到真实源目录；否则标签路径会从完整数据集推断，
        # 评估时就可能找不到对应 labels。
        sources.append(Path(os.path.abspath(source)))
    return sources


def iter_split_images(source: Path) -> list[tuple[Path, Path]]:
    """枚举一个 split 的图片，并保留用于输出目录复现的相对路径。"""
    if source.is_dir():
        images = [path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        return [(Path(os.path.abspath(path)), path.relative_to(source)) for path in sorted(images)]

    if source.is_file():
        records = []
        for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            raw = line.strip()
            if not raw:
                continue
            image_path = Path(raw)
            if not image_path.is_absolute():
                image_path = source.parent / image_path
            records.append((Path(os.path.abspath(image_path)), Path(f"{index:08d}_{image_path.name}")))
        return records

    raise FileNotFoundError(f"Cannot find image source declared by data.yaml: {source}")


def infer_label_path(image_path: Path) -> Path:
    """根据 YOLO 目录约定从图片路径推断标签路径。"""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_label_classes(label_path: Path) -> set[int]:
    """读取一个 YOLO txt 标签文件里出现过的类别编号。"""
    if not label_path.exists():
        return set()
    classes = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        items = line.strip().split()
        if items:
            classes.add(int(float(items[0])))
    return classes


def detect_label_space(
    label_classes: set[int],
    class_indices: list[int],
    mode: str,
    total_classes: int,
) -> str:
    """判断标签编号是局部类别空间还是全局类别空间。

    local 表示标签从 0 开始，只覆盖当前任务；global 表示标签编号已经对应
    detector.names 的全局编号。auto 会根据标签中出现的 class id 自动判断。
    """
    normalized_mode = str(mode).lower()
    if normalized_mode in {"global", "true", "yes", "1"}:
        return "global"
    if normalized_mode in {"local", "false", "no", "0"}:
        return "local"
    if normalized_mode != "auto":
        raise ValueError("labels_are_global must be auto, true/global, or false/local")

    if not label_classes:
        return "local"

    index_set = set(class_indices)
    local_ok = all(0 <= cls_id < len(class_indices) for cls_id in label_classes)
    global_ok = all(cls_id in index_set for cls_id in label_classes)
    global_like = all(0 <= cls_id < total_classes for cls_id in label_classes)
    if global_ok and not local_ok:
        return "global"
    if local_ok:
        return "local"
    if global_like:
        return "global"
    raise ValueError(
        f"Label class ids {sorted(label_classes)} are neither local ids for this task "
        f"nor valid global ids in [0, {total_classes - 1}]."
    )


def resolve_task_class_indices(task: dict, data_cfg: dict, cfg: dict) -> list[int]:
    """根据任务配置和 data.yaml names 推导当前任务对应的全局类别编号。"""
    configured = [int(index) for index in task["class_indices"]]
    source_names_raw = data_cfg.get("names")
    if source_names_raw is None:
        return configured

    source_names = normalize_names(source_names_raw)
    if len(source_names) != len(configured):
        return configured

    global_names = normalize_names(cfg["detector"]["names"])
    name_to_global = {name: index for index, name in global_names.items()}
    derived = []
    for local_index in range(len(source_names)):
        name = source_names[local_index]
        if name not in name_to_global:
            return configured
        derived.append(name_to_global[name])

    if derived != configured:
        print(f"[data] {task['name']} class_indices corrected from data.yaml names: {derived}")
    return derived


def link_or_copy_image(src: Path, dst: Path) -> None:
    """优先创建符号链接；Windows/权限不支持时退化为复制图片。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_label_file(
    src: Path,
    dst: Path,
    class_indices: list[int],
    label_space: str,
    global_to_output: dict[int, int] | None = None,
) -> None:
    """重写标签文件中的类别编号。

    训练时希望每个 prepared_data 都有明确的输出类别空间。这里会把局部标签转
    全局标签，再按 active_class_indices 映射为当前 YOLO head 的输出编号。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return

    remapped_lines = []
    for line_number, line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        items = line.strip().split()
        if not items:
            continue
        class_id = int(float(items[0]))
        if label_space == "local":
            if class_id >= len(class_indices):
                raise ValueError(f"Local class id {class_id} at {src}:{line_number} exceeds {class_indices}")
            class_id = class_indices[class_id]
        if global_to_output is not None:
            if class_id not in global_to_output:
                raise ValueError(
                    f"Global class id {class_id} at {src}:{line_number} is not in the active output space "
                    f"{sorted(global_to_output)}."
                )
            class_id = global_to_output[class_id]
        items[0] = str(class_id)
        remapped_lines.append(" ".join(items))

    dst.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")


def prepare_global_task_data(
    task: dict,
    cfg: dict,
    output_dir: Path,
    active_class_indices: Iterable[int] | None = None,
) -> Path:
    """为单个任务生成可直接喂给 Ultralytics 的 prepared_data/data.yaml。

    该函数完成三件事：收集图片、重映射标签、写出新的 data.yaml。这样每个
    增量任务都能在自己的类别输出空间中训练，同时保留全局类别信息用于评估。
    """
    original_yaml = Path(task["data"])
    if not original_yaml.is_absolute():
        original_yaml = Path.cwd() / original_yaml
    original_yaml = original_yaml.resolve()
    data_cfg = load_config(original_yaml)

    class_indices = resolve_task_class_indices(task, data_cfg, cfg)
    total_classes = int(cfg["detector"]["total_classes"])
    names = normalize_names(cfg["detector"]["names"])
    if len(names) != total_classes:
        raise ValueError("detector.names length must match detector.total_classes")
    if any(index < 0 or index >= total_classes for index in class_indices):
        raise ValueError(f"class_indices must be in [0, {total_classes - 1}]: {class_indices}")
    if active_class_indices is None:
        # 默认使用完整全局类别空间；增量阶段也可以传入已学习类别子集。
        active_class_indices = list(range(total_classes))
    active_class_indices = [int(index) for index in active_class_indices]
    if len(active_class_indices) != len(set(active_class_indices)):
        raise ValueError(f"active_class_indices cannot contain duplicates: {active_class_indices}")
    if any(index < 0 or index >= total_classes for index in active_class_indices):
        raise ValueError(f"active_class_indices must be in [0, {total_classes - 1}]: {active_class_indices}")
    if not set(class_indices).issubset(active_class_indices):
        raise ValueError(f"Task classes {class_indices} must be included in active output space {active_class_indices}")

    global_to_output = {global_index: output_index for output_index, global_index in enumerate(active_class_indices)}
    active_names = {output_index: names[global_index] for output_index, global_index in enumerate(active_class_indices)}

    prepared_root = output_dir / "prepared_data" / safe_task_name(task["name"])
    if prepared_root.exists():
        # 每次训练前重建 prepared_data，避免旧标签或旧软链接残留影响结果。
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[tuple[Path, Path]]] = {}
    label_classes: set[int] = set()
    for split in ("train", "val", "test"):
        records: list[tuple[Path, Path]] = []
        for source in resolve_split_sources(original_yaml, data_cfg, split):
            records.extend(iter_split_images(source))
        if records:
            split_records[split] = records
            for image_path, _ in records:
                label_classes.update(read_label_classes(infer_label_path(image_path)))

    if "train" not in split_records:
        raise ValueError(f"No train images found in {original_yaml}")

    label_space = detect_label_space(label_classes, class_indices, task.get("labels_are_global", "auto"), total_classes)
    if label_space == "global" and label_classes:
        # 如果标签本身已经是全局编号，则以实际标签为准修正 class_indices。
        actual_indices = sorted(label_classes)
        unexpected = sorted(set(actual_indices) - set(class_indices))
        if unexpected:
            print(f"[data] {task['name']} uses global labels outside configured class_indices: {unexpected}")
            print(f"[data] {task['name']} class_indices corrected from labels: {actual_indices}")
            class_indices = actual_indices

    if not set(class_indices).issubset(active_class_indices):
        raise ValueError(f"Resolved task classes {class_indices} must be included in {active_class_indices}")

    print(
        f"[data] {task['name']} label_space={label_space}; "
        f"writing cumulative {len(active_class_indices)}-class data.yaml"
    )

    prepared_cfg = {
        # Ultralytics data.yaml 中 path 使用绝对路径，避免从不同工作目录启动时找错数据。
        "path": str(prepared_root.resolve()),
        "train": "images/train",
        "nc": len(active_class_indices),
        "names": active_names,
        "global_class_indices": active_class_indices,
    }
    for split, records in split_records.items():
        prepared_cfg[split] = f"images/{split}"
        for image_path, relative_path in records:
            dst_image = prepared_root / "images" / split / relative_path
            dst_label = prepared_root / "labels" / split / relative_path.with_suffix(".txt")
            link_or_copy_image(image_path, dst_image)
            remap_label_file(
                infer_label_path(image_path),
                dst_label,
                class_indices,
                label_space,
                global_to_output,
            )

    if "channels" in data_cfg:
        prepared_cfg["channels"] = data_cfg["channels"]

    prepared_yaml = prepared_root / "data.yaml"
    prepared_yaml.write_text(yaml.safe_dump(prepared_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    task["prepared_data"] = str(prepared_yaml)
    task["label_space"] = label_space
    task["resolved_class_indices"] = class_indices
    task["active_class_indices"] = active_class_indices
    return prepared_yaml
