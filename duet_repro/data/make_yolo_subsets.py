from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_tasks(raw: str) -> list[list[int]]:
    return [[int(x) for x in part.split(",") if x.strip()] for part in raw.split("|")]


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def filter_label(src_label: Path, dst_label: Path, allowed_classes: set[int], remap: bool) -> bool:
    kept: list[str] = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        if cls not in allowed_classes:
            continue
        if remap:
            ordered = sorted(allowed_classes)
            cls = ordered.index(cls)
        kept.append(" ".join([str(cls), *parts[1:]]))

    if not kept:
        return False

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return True


def build_task(
    src_root: Path,
    dst_root: Path,
    task_id: int,
    classes: list[int],
    splits: list[str],
    copy_images: bool,
    remap: bool,
) -> None:
    task_root = dst_root / f"task_{task_id}"
    for split in splits:
        src_image_dir = src_root / "images" / split
        src_label_dir = src_root / "labels" / split
        dst_image_dir = task_root / "images" / split
        dst_label_dir = task_root / "labels" / split
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        labels = sorted(src_label_dir.glob("*.txt"))
        for src_label in tqdm(labels, desc=f"task {task_id} {split}"):
            dst_label = dst_label_dir / src_label.name
            has_label = filter_label(src_label, dst_label, set(classes), remap)
            if not has_label:
                continue
            image = find_image(src_image_dir, src_label.stem)
            if image is None:
                continue
            dst_image = dst_image_dir / image.name
            if copy_images:
                shutil.copy2(image, dst_image)
            else:
                if not dst_image.exists():
                    dst_image.symlink_to(image)

    names = {i: f"class_{i}" for i in (range(len(classes)) if remap else classes)}
    data_yaml = {
        "path": str(task_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (task_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, type=Path)
    parser.add_argument("--dst-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--tasks", required=True, help="Example: '0,1,2|3,4,5'")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--remap", action="store_true", help="Remap class ids inside each task.")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    for index, classes in enumerate(tasks, start=1):
        build_task(args.src_root, args.dst_root, index, classes, args.splits, args.copy_images, args.remap)


if __name__ == "__main__":
    main()

