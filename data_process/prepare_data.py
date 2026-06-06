from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def split_value_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def find_zip(zip_root: Path, task: dict[str, Any]) -> Path:
    candidates = []
    if "zip" in task:
        candidates.append(str(task["zip"]))
    candidates.extend(str(item) for item in task.get("aliases", []) or [])
    if not candidates:
        candidates.append(f"{task['name']}.zip")

    for item in candidates:
        path = Path(item)
        if not path.is_absolute():
            path = zip_root / path
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing zip for task {task['name']} under {zip_root}. Tried: {candidates}")


def extract_zip(zip_path: Path, extract_root: Path, overwrite: bool) -> Path:
    target = extract_root / zip_path.stem
    marker = target / ".extract_ok"
    if marker.exists() and not overwrite:
        return target
    reset_dir(target)
    print(f"[extract] {zip_path} -> {target}")
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"Bad file in {zip_path}: {bad}")
        archive.extractall(target)
    marker.write_text("ok\n", encoding="utf-8")
    return target


def find_dataset_root(extracted: Path, task: dict[str, Any], data_yaml_name: str) -> Path:
    if task.get("dataset_root"):
        root = Path(task["dataset_root"])
        if not root.is_absolute():
            root = extracted / root
        if not root.exists():
            raise FileNotFoundError(f"Configured dataset_root does not exist: {root}")
        return root

    candidates = []
    for yaml_path in extracted.rglob(data_yaml_name):
        root = yaml_path.parent
        if (root / "images").exists() and (root / "labels").exists():
            candidates.append(root)
    if candidates:
        return sorted(candidates, key=lambda p: len(p.parts))[0]
    if (extracted / "images").exists() and (extracted / "labels").exists():
        return extracted
    raise FileNotFoundError(f"No YOLO dataset root found under {extracted}")


def resolve_split_paths(dataset_root: Path, data_cfg: dict[str, Any], split: str) -> list[Path]:
    root = Path(data_cfg.get("path", dataset_root))
    if not root.is_absolute():
        root = dataset_root / root
    paths = []
    for item in split_value_to_list(data_cfg.get(split)):
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        paths.append(path)
    return paths


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [Path(line) if Path(line).is_absolute() else source.parent / line for line in lines]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"Missing image source: {source}")


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files or sys.platform.startswith("win"):
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)


def normalize_names(names: Any, class_indices: list[int]) -> dict[int, str]:
    if names is None:
        return {idx: f"class_{idx + 1}" for idx in range(len(class_indices))}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def write_dataset_yaml(path: Path, root: Path, names: dict[int, str], class_indices: list[int]) -> None:
    cfg = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "nc": len(names),
        "names": names,
        "global_class_indices": class_indices,
        "label_mapping": "local_to_global",
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def yolo_box(width: int, height: int, xyxy: list[float]) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = xyxy
    x = ((xmin + xmax) / 2.0 - 1.0) / width
    y = ((ymin + ymax) / 2.0 - 1.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return x, y, w, h


def normalize_yolo_dataset(
    dataset_root: Path,
    out_root: Path,
    class_indices: list[int],
    copy_files: bool,
    data_yaml_name: str,
    train_split: str,
    val_split: str,
    names_override: Any = None,
) -> tuple[int, int, int, int]:
    data_yaml = dataset_root / data_yaml_name
    data_cfg = read_yaml(data_yaml) if data_yaml.exists() else {}
    names = normalize_names(names_override if names_override is not None else data_cfg.get("names"), class_indices)
    reset_dir(out_root)

    counts = {"train_images": 0, "train_instances": 0, "val_images": 0, "val_instances": 0}
    split_pairs = [(train_split, "train"), (val_split, "val")]

    for source_split, target_split in split_pairs:
        sources = resolve_split_paths(dataset_root, data_cfg, source_split)
        if not sources:
            fallback = dataset_root / "images" / source_split
            if target_split == "val" and not fallback.exists() and source_split == "val":
                fallback = dataset_root / "images" / "test"
            sources = [fallback]
        for source in sources:
            for image_path in iter_images(source):
                dst_image = out_root / "images" / target_split / image_path.name
                dst_label = out_root / "labels" / target_split / f"{image_path.stem}.txt"
                link_or_copy(image_path, dst_image, copy_files)
                src_label = infer_label_path(image_path)
                lines = []
                if src_label.exists():
                    lines = [line for line in src_label.read_text(encoding="utf-8").splitlines() if line.strip()]
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                key_img = "train_images" if target_split == "train" else "val_images"
                key_inst = "train_instances" if target_split == "train" else "val_instances"
                counts[key_img] += 1
                counts[key_inst] += len(lines)

    write_dataset_yaml(out_root / "data.yaml", out_root, names, class_indices)
    return counts["train_images"], counts["train_instances"], counts["val_images"], counts["val_instances"]


def update_yaml_data_paths(config_path: Path, data_paths: dict[str, Path]) -> bool:
    if not config_path.exists():
        return False
    cfg = read_yaml(config_path)
    changed = False

    for task in cfg.get("tasks", []) or []:
        name = task.get("name")
        if name in data_paths:
            task["data"] = str(data_paths[name].as_posix())
            changed = True

    def walk(node: Any) -> None:
        nonlocal changed
        if isinstance(node, dict):
            if "data" in node and isinstance(node["data"], str):
                lower = node["data"].replace("\\", "/").lower()
                for name, path in data_paths.items():
                    if f"/{name}/data.yaml" in lower or lower.endswith(f"{name}/data.yaml"):
                        node["data"] = str(path.as_posix())
                        changed = True
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(cfg)
    if changed:
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"[config] updated: {config_path}")
    return changed


def expand_config_paths(items: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        raw = str(item)
        if any(char in raw for char in "*?[]"):
            matches = sorted(PROJECT_ROOT.glob(raw.replace("\\", "/")))
            paths.extend(path for path in matches if path.is_file())
        else:
            paths.append(project_path(raw))
    return paths


def collect_zip_file_configs(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for key, value in plan.items():
        if key == "datasets" and isinstance(value, dict):
            for name, cfg in value.items():
                files[str(name)] = dict(cfg or {})
        elif key.endswith("_files") and isinstance(value, dict):
            for name, cfg in value.items():
                files[str(name)] = dict(cfg or {})
        elif key.endswith("_zips") and isinstance(value, dict):
            for name, cfg in value.items():
                if isinstance(cfg, dict):
                    files[str(name)] = dict(cfg)
                else:
                    files[str(name)] = {"zip": str(cfg)}
    return files


def collect_zip_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if isinstance(plan.get("tasks"), list):
        tasks.extend(dict(task) for task in plan["tasks"])
    for key, value in plan.items():
        if key.endswith("_tasks") and isinstance(value, list):
            tasks.extend(dict(task) for task in value)
    return tasks


def resolve_task_dataset(plan: dict[str, Any], task: dict[str, Any], file_configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    resolved = dict(task)
    dataset_name = task.get("dataset") or task.get("data_file") or task.get("file") or task.get("name")
    if dataset_name is None or dataset_name not in file_configs:
        for key in ("dataset_root", "data_yaml", "train_split", "val_split"):
            if key in plan and key not in resolved:
                resolved[key] = plan[key]
        return resolved

    dataset_cfg = dict(file_configs[str(dataset_name)] or {})
    dataset_cfg.setdefault("name", str(dataset_name))
    dataset_cfg.update(task)
    for key in ("dataset_root", "data_yaml", "train_split", "val_split"):
        if key in plan and key not in dataset_cfg:
            dataset_cfg[key] = plan[key]
    return dataset_cfg


def print_plan_slice_summary(name: str, root: Path, train_images: int, train_instances: int, val_images: int, val_instances: int) -> None:
    print(
        f"[{name}] root={root} train_images={train_images} train_instances={train_instances} "
        f"val_images={val_images} val_instances={val_instances}"
    )


def prepare_voc_clipart_plan(plan: dict[str, Any]) -> None:
    from scripts.prepare_server_status1_slices import (
        add_clipart_slice_records,
        add_voc_slice_records,
        print_dir_summary,
        prepare_clipart_full,
        prepare_voc_full,
        VOC_TRAIN_SPLITS,
        write_slice_yaml,
    )

    data_root = project_path(plan.get("data_root", "data"))
    zip_root = project_path(plan.get("zip_root", data_root / "downloads"))
    voc_raw_root = data_root / "VOC_raw"
    voc_root = data_root / "VOC"
    voc_slice_root = data_root / "pascal_voc_yolo"
    clipart_root = data_root / "clipart_yolo"
    clipart_slice_root = data_root / "clipart_paper_yolo"

    print(f"[voc_clipart] data_root={data_root}")
    print(f"[voc_clipart] zip_root={zip_root}")
    prepare_voc_full(zip_root, voc_raw_root, voc_root, voc_zips=plan.get("voc_zips"))
    prepare_clipart_full(zip_root, clipart_root, clipart_zip=str(plan.get("clipart_zip", "clipart.zip")))

    data_paths: dict[str, Path] = {}
    reset_dir(voc_slice_root)
    for task in plan.get("voc_tasks", []):
        name = str(task["name"])
        class_indices = [int(index) for index in task["class_indices"]]
        slice_root = voc_slice_root / name
        train_images = train_instances = 0
        for split in task.get("train_splits", VOC_TRAIN_SPLITS):
            images, instances = add_voc_slice_records(voc_root, slice_root, str(split), "train", class_indices)
            train_images += images
            train_instances += instances
        val_split = str(task.get("val_split", "test2007"))
        val_images, val_instances = add_voc_slice_records(voc_root, slice_root, val_split, "val", class_indices)
        write_slice_yaml(slice_root / "data.yaml", slice_root, class_indices)
        data_paths[name] = slice_root / "data.yaml"
        print_plan_slice_summary(name, slice_root, train_images, train_instances, val_images, val_instances)

    reset_dir(clipart_slice_root)
    for task in plan.get("clipart_tasks", []):
        name = str(task["name"])
        class_indices = [int(index) for index in task["class_indices"]]
        slice_root = clipart_slice_root / name
        train_images, train_instances = add_clipart_slice_records(
            clipart_root,
            slice_root,
            str(task.get("train_split", "train")),
            "train",
            class_indices,
        )
        val_images, val_instances = add_clipart_slice_records(
            clipart_root,
            slice_root,
            str(task.get("val_split", "test")),
            "val",
            class_indices,
        )
        write_slice_yaml(slice_root / "data.yaml", slice_root, class_indices)
        data_paths[name] = slice_root / "data.yaml"
        print_plan_slice_summary(name, slice_root, train_images, train_instances, val_images, val_instances)

    if not plan.get("skip_summaries", False):
        for path in data_paths.values():
            print_dir_summary(path.parent)

    if plan.get("update_configs"):
        for config_path in expand_config_paths(plan.get("update_configs", [])):
            update_yaml_data_paths(config_path, data_paths)

    print("[done] VOC/Clipart datasets are ready.")


def prepare_zip_plan(plan: dict[str, Any], copy_files: bool, overwrite: bool, update_configs: bool) -> None:
    zip_root = project_path(plan["zip_root"])
    output_root = project_path(plan["output_root"])
    extract_root = project_path(plan.get("extract_root", output_root.parent / "_raw_extract"))
    output_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    file_configs = collect_zip_file_configs(plan)
    tasks = collect_zip_tasks(plan)
    data_paths: dict[str, Path] = {}
    for task in tasks:
        task = resolve_task_dataset(plan, task, file_configs)
        name = str(task["name"])
        class_indices = [int(index) for index in task["class_indices"]]
        data_yaml_name = str(task.get("data_yaml", plan.get("data_yaml", "data.yaml")))
        train_split = str(task.get("train_split", plan.get("train_split", "train")))
        val_split = str(task.get("val_split", plan.get("val_split", "val")))
        zip_path = find_zip(zip_root, task)
        extracted = extract_zip(zip_path, extract_root, overwrite)
        dataset_root = find_dataset_root(extracted, task, data_yaml_name)
        output_name = str(task.get("output", name))
        out_root = output_root / output_name
        train_images, train_instances, val_images, val_instances = normalize_yolo_dataset(
            dataset_root,
            out_root,
            class_indices,
            copy_files=copy_files,
            data_yaml_name=data_yaml_name,
            train_split=train_split,
            val_split=val_split,
            names_override=task.get("names"),
        )
        data_paths[name] = out_root / "data.yaml"
        print(
            f"[task] {name}: train_images={train_images} train_instances={train_instances} "
            f"val_images={val_images} val_instances={val_instances}"
        )

    if update_configs:
        for config_path in expand_config_paths(plan.get("update_configs", []) or []):
            update_yaml_data_paths(config_path, data_paths)

    print(f"[done] datasets are ready under: {output_root}")


def collect_named_zip_configs(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    zips: dict[str, dict[str, Any]] = {}
    for key, value in plan.items():
        if key.endswith("_zip") and isinstance(value, (str, Path)):
            zips[key[: -len("_zip")]] = {"zip": str(value)}
        elif key.endswith("_zips") and isinstance(value, dict):
            for name, cfg in value.items():
                zips[str(name)] = dict(cfg or {}) if isinstance(cfg, dict) else {"zip": str(cfg)}
    return zips


def find_voc_root(extracted: Path, configured_root: Any = None) -> Path:
    if configured_root:
        root = Path(str(configured_root))
        if not root.is_absolute():
            root = extracted / root
        if not root.exists():
            raise FileNotFoundError(f"Configured VOC root does not exist: {root}")
        return root
    candidates = []
    for path in extracted.rglob("*"):
        if path.is_dir() and (path / "Annotations").exists() and (path / "JPEGImages").exists() and (path / "ImageSets" / "Main").exists():
            candidates.append(path)
    if candidates:
        return sorted(candidates, key=lambda p: len(p.parts))[0]
    raise FileNotFoundError(f"No VOC-style root found under {extracted}")


def image_size(image_path: Path) -> tuple[int, int]:
    data = image_path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index < len(data):
            while index < len(data) and data[index] == 0xFF:
                index += 1
            marker = data[index]
            index += 1
            block_len = int.from_bytes(data[index : index + 2], "big")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height = int.from_bytes(data[index + 3 : index + 5], "big")
                width = int.from_bytes(data[index + 5 : index + 7], "big")
                return width, height
            index += block_len
    raise ValueError(f"Unsupported image format or missing size: {image_path}")


def parse_voc_xml_lines(xml_path: Path, image_path: Path, class_names: list[str], keep_indices: list[int]) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is not None:
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
    else:
        width, height = image_size(image_path)
    local_index = {global_id: local_id for local_id, global_id in enumerate(keep_indices)}
    keep = set(keep_indices)
    lines: list[str] = []
    for obj in root.iter("object"):
        cls_name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", "0") or 0)
        if difficult == 1 or cls_name not in class_names:
            continue
        global_id = class_names.index(cls_name)
        if global_id not in keep:
            continue
        box = obj.find("bndbox")
        xyxy = [float(box.findtext(tag)) for tag in ("xmin", "xmax", "ymin", "ymax")]
        lines.append(" ".join([str(local_index[global_id]), *(f"{value:.6f}" for value in yolo_box(width, height, xyxy))]))
    return lines


def build_voc_zip_slice(
    voc_root: Path,
    out_root: Path,
    task: dict[str, Any],
    class_names: list[str],
    copy_files: bool,
) -> tuple[int, int, int, int]:
    class_indices = [int(index) for index in task["class_indices"]]
    reset_dir(out_root)
    counts = {"train_images": 0, "train_instances": 0, "val_images": 0, "val_instances": 0}
    split_pairs = [(str(task.get("train_split", "train")), "train"), (str(task.get("val_split", "test")), "val")]

    for source_split, target_split in split_pairs:
        split_file = voc_root / "ImageSets" / "Main" / f"{source_split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for image_id in image_ids:
            xml_path = voc_root / "Annotations" / f"{image_id}.xml"
            image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
            if not xml_path.exists() or not image_path.exists():
                continue
            lines = parse_voc_xml_lines(xml_path, image_path, class_names, class_indices)
            if not lines:
                continue
            dst_image = out_root / "images" / target_split / image_path.name
            dst_label = out_root / "labels" / target_split / f"{image_path.stem}.txt"
            link_or_copy(image_path, dst_image, copy_files)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            key_img = "train_images" if target_split == "train" else "val_images"
            key_inst = "train_instances" if target_split == "train" else "val_instances"
            counts[key_img] += 1
            counts[key_inst] += len(lines)

    names = {local_id: class_names[global_id] for local_id, global_id in enumerate(class_indices)}
    write_dataset_yaml(out_root / "data.yaml", out_root, names, class_indices)
    return counts["train_images"], counts["train_instances"], counts["val_images"], counts["val_instances"]


def prepare_voc_zip_slices_plan(plan: dict[str, Any], copy_files: bool, overwrite: bool, update_configs: bool) -> None:
    zip_root = project_path(plan["zip_root"])
    data_root = project_path(plan.get("data_root", "data"))
    output_root = project_path(plan.get("output_root", data_root))
    extract_root = project_path(plan.get("extract_root", data_root / "raw"))
    class_names = [str(name) for name in plan["class_names"]]
    output_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    zip_configs = collect_named_zip_configs(plan)
    tasks = collect_zip_tasks(plan)
    data_paths: dict[str, Path] = {}
    extracted_roots: dict[str, Path] = {}

    for task in tasks:
        name = str(task["name"])
        zip_cfg = zip_configs.get(name)
        if zip_cfg is None:
            raise KeyError(f"Task {name} has no matching zip config. Add it to a *_zip or *_zips block.")
        zip_path = find_zip(zip_root, {"name": name, **zip_cfg})
        extracted = extract_zip(zip_path, extract_root, overwrite)
        cache_key = str(zip_path.resolve())
        if cache_key not in extracted_roots:
            extracted_roots[cache_key] = find_voc_root(extracted, zip_cfg.get("voc_root") or zip_cfg.get("dataset_root") or plan.get("dataset_root"))
        voc_root = extracted_roots[cache_key]
        out_root = output_root / str(task.get("output", name))
        train_images, train_instances, val_images, val_instances = build_voc_zip_slice(
            voc_root,
            out_root,
            task,
            class_names,
            copy_files=copy_files,
        )
        data_paths[name] = out_root / "data.yaml"
        print_plan_slice_summary(name, out_root, train_images, train_instances, val_images, val_instances)

    if update_configs:
        for config_path in expand_config_paths(plan.get("update_configs", []) or []):
            update_yaml_data_paths(config_path, data_paths)

    print(f"[done] VOC zip slices are ready under: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven data preparation.")
    parser.add_argument("--plan", required=True, help="YAML file describing how to split zip datasets.")
    parser.add_argument("--copy-files", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract zip files.")
    parser.add_argument("--no-update-configs", action="store_true", help="Do not rewrite train/eval YAML data paths.")
    args = parser.parse_args()

    plan_path = project_path(args.plan)
    plan = read_yaml(plan_path)
    kind = str(plan.get("kind", "zip_yolo"))

    if kind == "voc_clipart":
        prepare_voc_clipart_plan(plan)
    elif kind == "voc_zip_slices":
        prepare_voc_zip_slices_plan(
            plan,
            copy_files=args.copy_files,
            overwrite=args.overwrite,
            update_configs=not args.no_update_configs,
        )
    elif kind == "zip_yolo":
        prepare_zip_plan(
            plan,
            copy_files=args.copy_files,
            overwrite=args.overwrite,
            update_configs=not args.no_update_configs,
        )
    else:
        raise ValueError(f"Unsupported data process kind: {kind}")


if __name__ == "__main__":
    main()
