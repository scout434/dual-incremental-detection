from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import yaml


ULTRALYTICS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

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

VOC_ZIPS = [
    "VOCtrainval_06-Nov-2007.zip",
    "VOCtest_06-Nov-2007.zip",
    "VOCtrainval_11-May-2012.zip",
]

VOC_SPLITS = [
    ("2012", "train"),
    ("2012", "val"),
    ("2007", "train"),
    ("2007", "val"),
    ("2007", "test"),
]

VOC_TRAIN_SPLITS = ["train2012", "train2007", "val2012", "val2007"]

STATUS1_SLICES = {
    "1_10": list(range(0, 10)),
    "11_20": list(range(10, 20)),
    "1_20": list(range(0, 20)),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_yaml(path: Path, root: Path, names: list[str], train: str | list[str], val: str | list[str]) -> None:
    cfg = {
        "path": str(root.resolve()),
        "train": train,
        "val": val,
        "test": val,
        "nc": len(names),
        "names": {idx: name for idx, name in enumerate(names)},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_slice_yaml(path: Path, root: Path, global_indices: list[int]) -> None:
    cfg = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "nc": len(global_indices),
        "names": {idx: PAPER_NAMES[global_id] for idx, global_id in enumerate(global_indices)},
        "global_class_indices": global_indices,
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


def parse_voc_xml(xml_path: Path, names: list[str]) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    lines: list[str] = []

    for obj in root.iter("object"):
        cls_name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", "0") or 0)
        if difficult == 1 or cls_name not in names:
            continue
        box = obj.find("bndbox")
        xyxy = [float(box.findtext(tag)) for tag in ("xmin", "xmax", "ymin", "ymax")]
        cls_id = names.index(cls_name)
        lines.append(" ".join([str(cls_id), *(f"{value:.6f}" for value in yolo_box(width, height, xyxy))]))
    return lines


def extract_zip_once(zip_path: Path, dst: Path) -> None:
    marker = dst / ".extract_markers" / zip_path.stem
    if marker.exists():
        print(f"[extract] skip existing: {zip_path.name}")
        return
    print(f"[extract] {zip_path.name} -> {dst}")
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"Bad file in {zip_path}: {bad}")
        archive.extractall(dst)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def prepare_voc_full(download_dir: Path, raw_root: Path, voc_root: Path, voc_zips: list[str] | None = None) -> None:
    print("\n========== VOC full -> YOLO ==========")
    for name in (voc_zips or VOC_ZIPS):
        zip_path = download_dir / name
        require_file(zip_path)
        extract_zip_once(zip_path, raw_root)

    vocdevkit = raw_root / "VOCdevkit"
    reset_dir(voc_root / "images")
    reset_dir(voc_root / "labels")

    split_image_counts: dict[str, int] = {}
    split_instance_counts: dict[str, int] = {}
    for year, split in VOC_SPLITS:
        split_name = f"{split}{year}"
        ids_path = vocdevkit / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
        image_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        image_out = voc_root / "images" / split_name
        label_out = voc_root / "labels" / split_name
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        instances = 0
        for index, image_id in enumerate(image_ids, start=1):
            src_image = vocdevkit / f"VOC{year}" / "JPEGImages" / f"{image_id}.jpg"
            dst_image = image_out / f"{image_id}.jpg"
            shutil.copy2(src_image, dst_image)
            lines = parse_voc_xml(vocdevkit / f"VOC{year}" / "Annotations" / f"{image_id}.xml", ULTRALYTICS_NAMES)
            (label_out / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            instances += len(lines)
            if index % 1000 == 0 or index == len(image_ids):
                print(f"[VOC convert] {split_name}: {index}/{len(image_ids)}")

        split_image_counts[split_name] = len(image_ids)
        split_instance_counts[split_name] = instances
        print(f"[VOC full] {split_name}: images={len(image_ids)} instances={instances}")

    write_yaml(
        voc_root / "data.yaml",
        voc_root,
        ULTRALYTICS_NAMES,
        ["images/train2012", "images/train2007", "images/val2012", "images/val2007"],
        ["images/test2007"],
    )
    print(
        "[VOC full] train_images={0} val_images={1}".format(
            sum(split_image_counts[name] for name in VOC_TRAIN_SPLITS),
            split_image_counts["test2007"],
        )
    )


def read_yolo_lines(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            rows.append((int(float(parts[0])), parts[1]))
    return rows


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def add_voc_slice_records(
    voc_root: Path,
    out_root: Path,
    source_split: str,
    target_split: str,
    global_indices: list[int],
) -> tuple[int, int]:
    keep = set(global_indices)
    local_index = {global_id: local_id for local_id, global_id in enumerate(global_indices)}
    image_count = 0
    instance_count = 0
    source_images = voc_root / "images" / source_split
    source_labels = voc_root / "labels" / source_split

    for label_path in sorted(source_labels.glob("*.txt")):
        remapped: list[str] = []
        for ultra_id, rest in read_yolo_lines(label_path):
            class_name = ULTRALYTICS_NAMES[ultra_id]
            paper_id = PAPER_NAMES.index(class_name)
            if paper_id in keep:
                remapped.append(f"{local_index[paper_id]} {rest}")
        if not remapped:
            continue

        stem = f"{source_split}_{label_path.stem}"
        copy_image(source_images / f"{label_path.stem}.jpg", out_root / "images" / target_split / f"{stem}.jpg")
        label_dst = out_root / "labels" / target_split / f"{stem}.txt"
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("\n".join(remapped) + "\n", encoding="utf-8")
        image_count += 1
        instance_count += len(remapped)

    return image_count, instance_count


def build_voc_slices(voc_root: Path, out_root: Path) -> None:
    print("\n========== VOC paper slices, local labels ==========")
    reset_dir(out_root)
    for suffix, global_indices in STATUS1_SLICES.items():
        slice_root = out_root / f"voc_{suffix}"
        train_images = train_instances = 0
        for split in VOC_TRAIN_SPLITS:
            images, instances = add_voc_slice_records(voc_root, slice_root, split, "train", global_indices)
            train_images += images
            train_instances += instances
        val_images, val_instances = add_voc_slice_records(voc_root, slice_root, "test2007", "val", global_indices)
        write_slice_yaml(slice_root / "data.yaml", slice_root, global_indices)
        print(
            f"[voc_{suffix}] train_images={train_images} train_instances={train_instances} "
            f"val_images={val_images} val_instances={val_instances}"
        )


def read_clipart_ids(zf: zipfile.ZipFile, split: str) -> list[str]:
    data = zf.read(f"clipart/ImageSets/Main/{split}.txt").decode("utf-8")
    return [line.strip() for line in data.splitlines() if line.strip()]


def parse_clipart_xml(zf: zipfile.ZipFile, image_id: str) -> list[str]:
    root = ET.fromstring(zf.read(f"clipart/Annotations/{image_id}.xml"))
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    lines: list[str] = []
    for obj in root.iter("object"):
        cls_name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", "0") or 0)
        if difficult == 1 or cls_name not in PAPER_NAMES:
            continue
        box = obj.find("bndbox")
        xyxy = [float(box.findtext(tag)) for tag in ("xmin", "xmax", "ymin", "ymax")]
        cls_id = PAPER_NAMES.index(cls_name)
        lines.append(" ".join([str(cls_id), *(f"{value:.6f}" for value in yolo_box(width, height, xyxy))]))
    return lines


def prepare_clipart_full(download_dir: Path, clipart_root: Path, clipart_zip: str = "clipart.zip") -> None:
    print("\n========== Clipart full -> YOLO ==========")
    zip_path = download_dir / clipart_zip
    require_file(zip_path)
    reset_dir(clipart_root)

    skipped: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        split_map = {"train": read_clipart_ids(zf, "train"), "test": read_clipart_ids(zf, "test")}
        for split, image_ids in split_map.items():
            image_out = clipart_root / "images" / split
            label_out = clipart_root / "labels" / split
            image_out.mkdir(parents=True, exist_ok=True)
            label_out.mkdir(parents=True, exist_ok=True)
            instances = 0
            copied = 0

            for index, image_id in enumerate(image_ids, start=1):
                image_name = f"{image_id}.jpg"
                try:
                    with zf.open(f"clipart/JPEGImages/{image_name}") as src, (image_out / image_name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                except zipfile.BadZipFile:
                    (image_out / image_name).unlink(missing_ok=True)
                    skipped.append(f"{split}/{image_name}")
                    continue

                lines = parse_clipart_xml(zf, image_id)
                (label_out / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                instances += len(lines)
                copied += 1
                if index % 100 == 0 or index == len(image_ids):
                    print(f"[Clipart convert] {split}: {index}/{len(image_ids)}")

            print(f"[Clipart full] {split}: images={copied} instances={instances}")

    write_yaml(clipart_root / "data.yaml", clipart_root, PAPER_NAMES, "images/train", "images/test")
    if skipped:
        print("[Clipart full] skipped broken images:")
        for item in skipped:
            print(f"  - {item}")


def add_clipart_slice_records(
    clipart_root: Path,
    out_root: Path,
    source_split: str,
    target_split: str,
    global_indices: list[int],
) -> tuple[int, int]:
    keep = set(global_indices)
    local_index = {global_id: local_id for local_id, global_id in enumerate(global_indices)}
    image_count = 0
    instance_count = 0
    source_images = clipart_root / "images" / source_split
    source_labels = clipart_root / "labels" / source_split

    for label_path in sorted(source_labels.glob("*.txt")):
        remapped: list[str] = []
        for global_id, rest in read_yolo_lines(label_path):
            if global_id in keep:
                remapped.append(f"{local_index[global_id]} {rest}")
        if not remapped:
            continue

        copy_image(source_images / f"{label_path.stem}.jpg", out_root / "images" / target_split / f"{label_path.stem}.jpg")
        label_dst = out_root / "labels" / target_split / label_path.name
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("\n".join(remapped) + "\n", encoding="utf-8")
        image_count += 1
        instance_count += len(remapped)

    return image_count, instance_count


def build_clipart_slices(clipart_root: Path, out_root: Path) -> None:
    print("\n========== Clipart paper slices, local labels ==========")
    reset_dir(out_root)
    for suffix, global_indices in STATUS1_SLICES.items():
        slice_root = out_root / f"clipart_{suffix}"
        train_images, train_instances = add_clipart_slice_records(clipart_root, slice_root, "train", "train", global_indices)
        val_images, val_instances = add_clipart_slice_records(clipart_root, slice_root, "test", "val", global_indices)
        write_slice_yaml(slice_root / "data.yaml", slice_root, global_indices)
        print(
            f"[clipart_{suffix}] train_images={train_images} train_instances={train_instances} "
            f"val_images={val_images} val_instances={val_instances}"
        )


def count_instances(label_dir: Path) -> int:
    total = 0
    if not label_dir.exists():
        return 0
    for label_path in label_dir.glob("*.txt"):
        total += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def print_dir_summary(root: Path) -> None:
    print(f"\n========== Directory summary: {root} ==========")
    if not root.exists():
        print("missing")
        return
    for image_dir in sorted((root / "images").glob("*")) if (root / "images").exists() else []:
        if not image_dir.is_dir():
            continue
        image_count = sum(1 for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        label_dir = root / "labels" / image_dir.name
        label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        instance_count = count_instances(label_dir)
        print(f"{image_dir.name}: images={image_count} labels={label_count} instances={instance_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=None,
        help="Output root. Default: the directory where this script is located.",
    )
    parser.add_argument("--download-dir", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve() if args.data_root else script_dir
    download_dir = Path(args.download_dir).resolve() if args.download_dir else data_root / "downloads"
    print(f"[config] data_root={data_root}")
    print(f"[config] download_dir={download_dir}")
    print("[config] no download, no zip size check")

    voc_raw_root = data_root / "VOC_raw"
    voc_root = data_root / "VOC"
    voc_slice_root = data_root / "pascal_voc_yolo"
    clipart_root = data_root / "clipart_yolo"
    clipart_slice_root = data_root / "clipart_paper_yolo"

    prepare_voc_full(download_dir, voc_raw_root, voc_root)
    build_voc_slices(voc_root, voc_slice_root)
    prepare_clipart_full(download_dir, clipart_root)
    build_clipart_slices(clipart_root, clipart_slice_root)

    for path in [
        voc_root,
        voc_slice_root / "voc_1_10",
        voc_slice_root / "voc_11_20",
        voc_slice_root / "voc_1_20",
        clipart_root,
        clipart_slice_root / "clipart_1_10",
        clipart_slice_root / "clipart_11_20",
        clipart_slice_root / "clipart_1_20",
    ]:
        print_dir_summary(path)

    print("\n[done] status1 server datasets are ready.")


if __name__ == "__main__":
    main()
