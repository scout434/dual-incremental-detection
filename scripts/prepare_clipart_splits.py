from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "data" / "downloads" / "clipart.zip"
FULL_OUT = PROJECT_ROOT / "data" / "clipart_yolo"
SLICE_OUT = PROJECT_ROOT / "data" / "clipart_paper_yolo"

PAPER_NAMES = [
    "bicycle",
    "bird",
    "car",
    "cat",
    "dog",
    "person",
    "aeroplane",
    "boat",
    "bottle",
    "bus",
    "chair",
    "cow",
    "diningtable",
    "horse",
    "motorbike",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def yolo_box(width: int, height: int, box: list[float]) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = box
    x = ((xmin + xmax) / 2.0 - 1.0) / width
    y = ((ymin + ymax) / 2.0 - 1.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return x, y, w, h


def read_ids(zf: zipfile.ZipFile, split: str) -> list[str]:
    data = zf.read(f"clipart/ImageSets/Main/{split}.txt").decode("utf-8")
    return [line.strip() for line in data.splitlines() if line.strip()]


def convert_xml(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    lines: list[str] = []

    for obj in root.iter("object"):
        name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", "0") or 0)
        if difficult == 1 or name not in PAPER_NAMES:
            continue
        box = obj.find("bndbox")
        xyxy = [float(box.findtext(tag)) for tag in ("xmin", "xmax", "ymin", "ymax")]
        cls_id = PAPER_NAMES.index(name)
        lines.append(" ".join([str(cls_id), *(f"{value:.6f}" for value in yolo_box(width, height, xyxy))]))
    return lines


def write_yaml(path: Path, root: Path, names: list[str], train: str, val: str, test: str | None = None) -> None:
    cfg = {
        "path": str(root.resolve()),
        "train": train,
        "val": val,
        "nc": len(names),
        "names": {idx: name for idx, name in enumerate(names)},
    }
    if test is not None:
        cfg["test"] = test
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def prepare_full_clipart() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"找不到 Clipart 压缩包: {ZIP_PATH}")

    if FULL_OUT.exists():
        shutil.rmtree(FULL_OUT)
    FULL_OUT.mkdir(parents=True, exist_ok=True)

    skipped_images: list[str] = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        split_map = {"train": read_ids(zf, "train"), "test": read_ids(zf, "test")}
        for split, image_ids in split_map.items():
            image_dir = FULL_OUT / "images" / split
            label_dir = FULL_OUT / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            for index, image_id in enumerate(image_ids, start=1):
                image_name = f"{image_id}.jpg"
                xml_name = f"{image_id}.xml"
                try:
                    with zf.open(f"clipart/JPEGImages/{image_name}") as src, (image_dir / image_name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                except zipfile.BadZipFile:
                    (image_dir / image_name).unlink(missing_ok=True)
                    skipped_images.append(f"{split}/{image_name}")
                    continue
                lines = convert_xml(zf.read(f"clipart/Annotations/{xml_name}"))
                (label_dir / f"{image_id}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8",
                )
                if index % 100 == 0 or index == len(image_ids):
                    print(f"[Clipart] {split}: {index}/{len(image_ids)}")

    write_yaml(FULL_OUT / "data.yaml", FULL_OUT, PAPER_NAMES, "images/train", "images/test", "images/test")
    if skipped_images:
        print("[Clipart] 跳过损坏图片:")
        for item in skipped_images:
            print(f"  - {item}")


def label_classes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    classes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            classes.add(int(float(parts[0])))
    return classes


def copy_slice(name: str, indices: list[int]) -> tuple[int, int, int, int]:
    dst_root = SLICE_OUT / name
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    train_images = val_images = train_instances = val_instances = 0
    keep = set(indices)
    local_index = {global_id: local_id for local_id, global_id in enumerate(indices)}

    for src_split, dst_split in (("train", "train"), ("test", "val")):
        src_image_dir = FULL_OUT / "images" / src_split
        src_label_dir = FULL_OUT / "labels" / src_split
        dst_image_dir = dst_root / "images" / dst_split
        dst_label_dir = dst_root / "labels" / dst_split
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for label_path in sorted(src_label_dir.glob("*.txt")):
            lines = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts and int(float(parts[0])) in keep:
                    global_id = int(float(parts[0]))
                    parts[0] = str(local_index[global_id])
                    lines.append(" ".join(parts))
            if not lines:
                continue

            image_path = src_image_dir / f"{label_path.stem}.jpg"
            shutil.copy2(image_path, dst_image_dir / image_path.name)
            (dst_label_dir / label_path.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            if dst_split == "train":
                train_images += 1
                train_instances += len(lines)
            else:
                val_images += 1
                val_instances += len(lines)

    write_yaml(
        dst_root / "data.yaml",
        dst_root,
        [PAPER_NAMES[index] for index in indices],
        "images/train",
        "images/val",
        "images/val",
    )
    cfg = yaml.safe_load((dst_root / "data.yaml").read_text(encoding="utf-8"))
    cfg["global_class_indices"] = indices
    (dst_root / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return train_images, train_instances, val_images, val_instances


def count_full() -> tuple[int, int]:
    train = len(list((FULL_OUT / "images" / "train").glob("*.jpg")))
    test = len(list((FULL_OUT / "images" / "test").glob("*.jpg")))
    return train, test


def main() -> None:
    prepare_full_clipart()
    train, test = count_full()
    print(f"\n[Clipart full] train_images={train} val_images={test}")

    SLICE_OUT.mkdir(parents=True, exist_ok=True)
    slices = {
        "clipart_1_10": list(range(0, 10)),
        "clipart_11_20": list(range(10, 20)),
        "clipart_1_20": list(range(0, 20)),
    }
    for name, indices in slices.items():
        train_images, train_instances, val_images, val_instances = copy_slice(name, indices)
        print(
            f"[{name}] train_images={train_images} train_instances={train_instances} "
            f"val_images={val_images} val_instances={val_instances}"
        )


if __name__ == "__main__":
    main()
