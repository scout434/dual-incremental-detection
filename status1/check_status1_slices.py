from __future__ import annotations

"""
检查 status1 的四个论文切片是否干净。

用途：
  python check_status1_slices.py

如果 voc_1_10 里出现了 10-19 类，或者 clipart_11_20 里出现了 0-9 类，
脚本会直接报错。这样可以避免把错误数据拿去训练 100 epochs。
"""

from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


CHECK_ITEMS = [
    (
        "voc_1_10",
        Path("data/pascal_voc_yolo/voc_1_10/data.yaml"),
        set(range(0, 10)),
    ),
    (
        "clipart_11_20",
        Path("data/clipart_paper_yolo/clipart_11_20/data.yaml"),
        set(range(0, 10)),
    ),
    (
        "voc_11_20",
        Path("data/pascal_voc_yolo/voc_11_20/data.yaml"),
        set(range(0, 10)),
    ),
    (
        "clipart_1_10",
        Path("data/clipart_paper_yolo/clipart_1_10/data.yaml"),
        set(range(0, 10)),
    ),
]


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YOLO data.yaml。"""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_root(data_yaml: Path, cfg: dict[str, Any]) -> Path:
    """解析 data.yaml 中的 path 字段。"""
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_path(data_yaml: Path, cfg: dict[str, Any], split: str) -> Path | None:
    """解析 train/val 对应的图片目录。"""
    value = cfg.get(split)
    if value is None:
        return None
    value_path = Path(str(value))
    if value_path.is_absolute():
        return value_path
    return (resolve_root(data_yaml, cfg) / value_path).resolve()


def infer_label_path(image_path: Path) -> Path:
    """从 YOLO 图片路径推断标签路径。"""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def iter_images(image_dir: Path) -> list[Path]:
    """列出一个 split 下的全部图片。"""
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def count_label_classes(data_yaml: Path) -> dict[str, Counter[int]]:
    """统计每个 split 中出现的类别编号。"""
    cfg = load_yaml(data_yaml)
    result: dict[str, Counter[int]] = {}
    for split in ("train", "val"):
        image_dir = resolve_split_path(data_yaml, cfg, split)
        if image_dir is None:
            continue
        counter: Counter[int] = Counter()
        for image_path in iter_images(image_dir):
            label_path = infer_label_path(image_path)
            if not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                items = line.strip().split()
                if items:
                    counter[int(float(items[0]))] += 1
        result[split] = counter
    return result


def main() -> None:
    """执行四个切片的类别检查。"""
    has_error = False
    for name, data_yaml, allowed in CHECK_ITEMS:
        print(f"\n{name}: {data_yaml}")
        if not data_yaml.exists():
            print("  [错误] data.yaml 不存在，请先运行 prepare_paper_slices.py")
            has_error = True
            continue

        split_counters = count_label_classes(data_yaml)
        for split, counter in split_counters.items():
            found = set(counter)
            unexpected = sorted(found - allowed)
            print(f"  {split}: classes={sorted(found)}, instances={sum(counter.values())}")
            if unexpected:
                print(f"  [错误] {split} 出现了不属于该论文切片的类别: {unexpected}")
                has_error = True

    if has_error:
        raise SystemExit("\nstatus1 数据切片检查失败，请先修正数据再训练。")

    print("\nstatus1 数据切片检查通过，可以开始训练。")


if __name__ == "__main__":
    main()
