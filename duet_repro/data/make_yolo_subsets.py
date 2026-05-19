"""
YOLO 数据集子集划分脚本 (终极优雅版 - 基于 data.yaml 路径解析)

核心改进：
1. 完全依赖 src-yaml 中的 path, train, val 字段定位源数据。
2. 自动推导标签路径（将图像路径中的 'images' 替换为 'labels'）。
3. 保留原始类别名称。
4. 支持累积类别空间映射 (CIL)。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm


# 支持的图像文件扩展名列表
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_tasks(raw: str) -> list[list[int]]:
    """解析任务配置字符串。"""
    return [[int(x) for x in part.split(",") if x.strip()] for part in raw.split("|")]


def load_dataset_config(yaml_path: Path) -> dict:
    """
    加载并验证数据集配置文件。
    返回包含绝对路径的结构化配置。
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if 'path' not in data:
        raise ValueError("data.yaml must contain a 'path' field.")
    if 'train' not in data or 'val' not in data:
        raise ValueError("data.yaml must contain 'train' and 'val' fields.")

    base_path = Path(data['path'])

    # 处理相对路径和绝对路径
    train_img_dir = base_path / data['train'] if not Path(data['train']).is_absolute() else Path(data['train'])
    val_img_dir = base_path / data['val'] if not Path(data['val']).is_absolute() else Path(data['val'])

    # 推导标签目录：将图像路径中的 'images' 替换为 'labels'
    # 这是一个强大的约定，覆盖了 99% 的 YOLO 数据集
    def get_label_dir(img_dir: Path) -> Path:
        # 尝试替换最后一个 'images' 出现的位置
        parts = img_dir.parts
        try:
            # 找到 'images' 在路径中的索引
            idx = [p.lower() for p in parts].index('images')
            # 构建新路径：将 images 替换为 labels
            new_parts = list(parts)
            new_parts[idx] = 'labels'
            return Path(*new_parts)
        except ValueError:
            # 如果路径中没有 'images'，尝试另一种常见结构：
            # 如果 train 是 "train/images"，则 label 是 "train/labels"
            # 这里我们简单地将路径末尾的文件夹名如果是 images 则替换，否则追加 ../labels?
            # 为了安全，如果找不到 images，我们假设 labels 在与 images 同级的目录下
            # 例如: root/data/train -> root/data/labels (如果原图在 root/data/train/images 这种嵌套情况外)

            #  fallback: 假设目录结构是 parallel
            # 如果 img_dir 是 .../something/train, 尝试 .../something/labels
            parent = img_dir.parent
            label_dir = parent / "labels" / img_dir.name
            if label_dir.exists():
                return label_dir

            # 如果还是不行，报错
            raise FileNotFoundError(
                f"Could not automatically infer label directory for {img_dir}. "
                f"Expected 'images' in path to replace with 'labels', or parallel 'labels' folder."
            )

    try:
        train_lbl_dir = get_label_dir(train_img_dir)
        val_lbl_dir = get_label_dir(val_img_dir)
    except Exception as e:
        print(f"[Warning] Auto-detection of label path failed: {e}")
        print("[Info] Please ensure your dataset follows the standard structure where 'labels' replaces 'images' in the path.")
        raise

    names = data.get('names', {})
    # 确保 names key 为 int
    names = {int(k): v for k, v in names.items()}

    return {
        'train_img': train_img_dir,
        'train_lbl': train_lbl_dir,
        'val_img': val_img_dir,
        'val_lbl': val_lbl_dir,
        'names': names
    }


def find_image(image_dir: Path, stem: str) -> Path | None:
    """根据标签文件名查找对应的图像文件。"""
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_global_class_mapping(all_tasks_classes: list[list[int]]) -> tuple[dict[int, int], list[int]]:
    """构建从【原始类别ID】到【全局累积类别ID】的映射表。"""
    seen_classes = []
    global_map = {}
    for task_classes in all_tasks_classes:
        for cls in task_classes:
            if cls not in seen_classes:
                seen_classes.append(cls)
                global_map[cls] = len(seen_classes) - 1
    return global_map, seen_classes


def filter_label(
    src_label: Path,
    dst_label: Path,
    allowed_original_classes: set[int],
    global_class_map: dict[int, int],
) -> bool:
    """过滤标签文件，并将类别 ID 映射为全局累积 ID。"""
    kept: list[str] = []
    try:
        lines = src_label.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            original_cls = int(float(parts[0]))
        except ValueError:
            continue

        if original_cls not in allowed_original_classes:
            continue

        if original_cls in global_class_map:
            new_cls = global_class_map[original_cls]
        else:
            continue

        kept.append(" ".join([str(new_cls), *parts[1:]]))

    if not kept:
        return False

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return True


def process_split(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    allowed_set: set[int],
    global_class_map: dict[int, int],
    copy_images: bool,
    split_name: str,
    task_id: int
):
    """处理单个拆分（train 或 val）的文件复制和标签过滤。"""
    if not src_lbl_dir.exists():
        print(f"[Warning] Source label directory does not exist: {src_lbl_dir}")
        return

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(src_lbl_dir.glob("*.txt"))

    for src_label in tqdm(labels, desc=f"Task {task_id} {split_name}", leave=False):
        dst_label = dst_lbl_dir / src_label.name

        has_label = filter_label(src_label, dst_label, allowed_set, global_class_map)

        if not has_label:
            continue

        image = find_image(src_img_dir, src_label.stem)
        if image is None:
            continue

        dst_image = dst_img_dir / image.name
        if copy_images:
            shutil.copy2(image, dst_image)
        else:
            if not dst_image.exists():
                dst_image.symlink_to(image.resolve())


def build_task(
    dst_root: Path,
    task_id: int,
    original_classes: list[int],
    copy_images: bool,
    global_class_map: dict[int, int],
    all_seen_classes_ordered: list[int],
    original_names: dict[int, str],
    src_dirs: dict
) -> None:
    """构建单个任务的数据子集。"""
    task_root = dst_root / f"task_{task_id}"

    # 1. 处理 Train Split
    process_split(
        src_img_dir=src_dirs['train_img'],
        src_lbl_dir=src_dirs['train_lbl'],
        dst_img_dir=task_root / "images" / "train",
        dst_lbl_dir=task_root / "labels" / "train",
        allowed_set=set(original_classes),
        global_class_map=global_class_map,
        copy_images=copy_images,
        split_name="train",
        task_id=task_id
    )

    # 2. 处理 Val Split
    process_split(
        src_img_dir=src_dirs['val_img'],
        src_lbl_dir=src_dirs['val_lbl'],
        dst_img_dir=task_root / "images" / "val",
        dst_lbl_dir=task_root / "labels" / "val",
        allowed_set=set(original_classes),
        global_class_map=global_class_map,
        copy_images=copy_images,
        split_name="val",
        task_id=task_id
    )

    # 3. 生成 data.yaml
    names = {}
    for orig_cls in all_seen_classes_ordered:
        global_id = global_class_map[orig_cls]
        class_name = original_names.get(orig_cls, f"class_{orig_cls}")
        names[global_id] = class_name

    data_yaml = {
        "path": str(task_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    (task_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[Info] Task {task_id} generated. Total classes (cumulative): {len(names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO CIL Dataset Splitter")
    parser.add_argument("--src-yaml", required=True, type=Path, help="源数据集的 data.yaml 路径")
    parser.add_argument("--dst-root", required=True, type=Path, help="目标根目录")
    parser.add_argument("--tasks", required=True, help="任务配置，如 '0,1|2,3|4'")
    parser.add_argument("--copy-images", action="store_true", help="复制图像而非符号链接")

    args = parser.parse_args()

    # 1. 加载源数据集配置（路径和名称）
    print(f"[Info] Loading dataset config from {args.src_yaml}...")
    src_dirs = load_dataset_config(args.src_yaml)
    print(f"[Info] Found Train Img: {src_dirs['train_img']}")
    print(f"[Info] Found Train Lbl: {src_dirs['train_lbl']}")

    original_names = src_dirs['names']
    print(f"[Info] Loaded {len(original_names)} class names.")

    # 2. 解析任务
    tasks = parse_tasks(args.tasks)

    # 3. 预计算全局映射
    global_class_map, seen_ordered = get_global_class_mapping(tasks)
    print(f"[Info] Global Class Mapping: {global_class_map}")

    # 4. 逐个构建任务
    current_seen_ordered = []

    for index, original_classes in enumerate(tasks, start=1):
        for cls in original_classes:
            if cls not in current_seen_ordered:
                current_seen_ordered.append(cls)

        build_task(
            dst_root=args.dst_root,
            task_id=index,
            original_classes=original_classes,
            copy_images=args.copy_images,
            global_class_map=global_class_map,
            all_seen_classes_ordered=current_seen_ordered,
            original_names=original_names,
            src_dirs=src_dirs
        )


if __name__ == "__main__":
    main()
