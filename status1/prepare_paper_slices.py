from __future__ import annotations

"""
为 DuET 论文实验准备 YOLO 数据切片。

你的原始 data.yaml 可能已经包含全局 20 类标签。论文里的任务却要求只训练或评估其中一部分：
  Pascal 两阶段示例：
    T1 训练 VOC [1:10]
    T2 训练 Clipart [11:20]
    GI 还需要评估 VOC [11:20] 和 Clipart [1:10]

这个脚本会从全量 YOLO 数据集中筛出指定类别，生成新的 data.yaml。
生成后的标签仍然保留全局类别编号，这样 train_duet.py 可以训练全局 20 类检测头。
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件。"""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def normalize_names(names: dict | list) -> dict[int, str]:
    """把 names 统一成 {int: str} 格式。"""
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def resolve_dataset_root(data_yaml: Path, data_cfg: dict[str, Any]) -> Path:
    """解析源数据集根目录。"""
    root = Path(data_cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_sources(data_yaml: Path, data_cfg: dict[str, Any], split: str) -> list[Path]:
    """解析 train/val/test 对应的图片目录或 txt 文件。"""
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
        # 不跟随软链接，避免从已经切好的数据再次切片时误读全量数据标签。
        sources.append(Path(os.path.abspath(source)))
    return sources


def iter_split_images(source: Path) -> list[tuple[Path, Path]]:
    """
    枚举一个 split 中的图片。

    返回 (原始图片路径, 输出数据集中的相对路径)。
    """
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

    raise FileNotFoundError(f"找不到图片来源: {source}")


def infer_label_path(image_path: Path) -> Path:
    """按 YOLO 常见目录结构，从图片路径推断标签路径。"""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def build_source_to_global_map(source_names: dict[int, str], global_names: dict[int, str]) -> dict[int, int]:
    """根据类别名建立“源数据类别编号 -> 全局类别编号”的映射。"""
    name_to_global = {name: index for index, name in global_names.items()}
    mapping = {}
    for source_index, name in source_names.items():
        if name in name_to_global:
            mapping[source_index] = name_to_global[name]
        else:
            mapping[source_index] = source_index
    return mapping


def filter_label_file(
    src_label: Path,
    dst_label: Path,
    *,
    keep_classes: set[int],
    source_to_global: dict[int, int],
) -> int:
    """
    过滤一个标签文件，只保留 keep_classes 中的全局类别。

    返回保留下来的目标框数量。
    """
    if not src_label.exists():
        return 0

    kept_lines = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        items = line.strip().split()
        if not items:
            continue
        source_cls = int(float(items[0]))
        global_cls = source_to_global.get(source_cls, source_cls)
        if global_cls not in keep_classes:
            continue
        items[0] = str(global_cls)
        kept_lines.append(" ".join(items))

    if not kept_lines:
        return 0

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return len(kept_lines)


def link_or_copy_image(src: Path, dst: Path, *, copy_images: bool) -> None:
    """把图片放到输出数据集；默认软链接，必要时复制。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_one_slice(
    *,
    name: str,
    source_yaml: Path,
    output_root: Path,
    keep_classes: set[int],
    global_names: dict[int, str],
    copy_images: bool,
) -> Path:
    """生成一个数据切片。"""
    source_cfg = load_yaml(source_yaml)
    source_names = normalize_names(source_cfg.get("names", global_names))
    source_to_global = build_source_to_global_map(source_names, global_names)

    slice_root = output_root / name
    if slice_root.exists():
        shutil.rmtree(slice_root)
    slice_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "val", "test"):
        records = []
        for source in resolve_split_sources(source_yaml, source_cfg, split):
            records.extend(iter_split_images(source))
        if not records:
            continue

        kept_images = 0
        kept_boxes = 0
        for image_path, relative_path in records:
            dst_image = slice_root / "images" / split / relative_path
            dst_label = slice_root / "labels" / split / relative_path.with_suffix(".txt")
            box_count = filter_label_file(
                infer_label_path(image_path),
                dst_label,
                keep_classes=keep_classes,
                source_to_global=source_to_global,
            )
            if box_count == 0:
                continue
            link_or_copy_image(image_path, dst_image, copy_images=copy_images)
            kept_images += 1
            kept_boxes += box_count

        summary[split] = {"images": kept_images, "instances": kept_boxes}

    data_yaml = slice_root / "data.yaml"
    data_cfg = {
        "path": str(slice_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(global_names),
        "names": global_names,
    }
    if (slice_root / "images" / "test").exists():
        data_cfg["test"] = "images/test"

    data_yaml.write_text(yaml.safe_dump(data_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[切片完成] {name}: {data_yaml}")
    for split, stat in summary.items():
        print(f"  - {split}: {stat['images']} images, {stat['instances']} instances")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 DuET 论文实验所需的 YOLO 数据切片。")
    parser.add_argument("--config", required=True, type=Path, help="切片配置 YAML。")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    global_names = normalize_names(cfg["global_names"])
    copy_images = bool(cfg.get("copy_images", False))
    sources = {name: Path(path).resolve() for name, path in cfg["sources"].items()}

    for item in cfg["slices"]:
        source_key = item["source"]
        if source_key not in sources:
            raise KeyError(f"切片 {item['name']} 引用了不存在的数据源: {source_key}")
        prepare_one_slice(
            name=item["name"],
            source_yaml=sources[source_key],
            output_root=output_root,
            keep_classes={int(index) for index in item["classes"]},
            global_names=global_names,
            copy_images=copy_images,
        )


if __name__ == "__main__":
    main()
