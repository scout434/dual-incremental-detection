from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"

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

ZIPS = {
    "VOCtrainval_06-Nov-2007.zip": {
        "url": f"{ASSETS_URL}/VOCtrainval_06-Nov-2007.zip",
        "size": 469_982_116,
    },
    "VOCtest_06-Nov-2007.zip": {
        "url": f"{ASSETS_URL}/VOCtest_06-Nov-2007.zip",
        "size": 451_614_552,
    },
    "VOCtrainval_11-May-2012.zip": {
        "url": f"{ASSETS_URL}/VOCtrainval_11-May-2012.zip",
        "size": 2_059_061_419,
    },
}

SLICES = {
    "voc_1_10": set(range(0, 10)),
    "voc_11_20": set(range(10, 20)),
    "voc_1_20": set(range(0, 20)),
}


def download_file(url: str, target: Path, expected_size: int | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        print(f"[download] exists: {target}")
        return
    if target.exists():
        print(f"[download] removing incomplete file: {target} ({target.stat().st_size} bytes)")
        target.unlink()

    print(f"[download] {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length", expected_size or 0))
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total * 100:5.1f}% {done / 1024 / 1024:,.0f}MB/{total / 1024 / 1024:,.0f}MB", end="")
        print()


def extract_zip(zip_path: Path, voc_root: Path) -> None:
    marker = voc_root / ".extracted" / zip_path.stem
    if marker.exists():
        print(f"[extract] complete, skip: {zip_path.name}")
        return
    print(f"[extract] {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(voc_root / "images")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def convert_box(size: tuple[int, int], box: list[float]) -> tuple[float, float, float, float]:
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_label(vocdevkit: Path, labels_path: Path, year: str, image_id: str) -> int:
    annotation = vocdevkit / f"VOC{year}" / "Annotations" / f"{image_id}.xml"
    root = ET.parse(annotation).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    lines = []
    for obj in root.iter("object"):
        cls = obj.findtext("name")
        difficult = int(obj.findtext("difficult", default="0"))
        if cls not in ULTRALYTICS_NAMES or difficult == 1:
            continue
        xmlbox = obj.find("bndbox")
        coords = [float(xmlbox.findtext(x)) for x in ("xmin", "xmax", "ymin", "ymax")]
        box = convert_box((width, height), coords)
        cls_id = ULTRALYTICS_NAMES.index(cls)
        lines.append(" ".join(str(value) for value in (cls_id, *box)))

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def move_or_copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.move(str(src), str(dst))


def build_ultralytics_voc(voc_root: Path) -> None:
    vocdevkit = voc_root / "images" / "VOCdevkit"
    total_images: dict[str, int] = {}
    total_instances: dict[str, int] = {}

    for year, split in (("2012", "train"), ("2012", "val"), ("2007", "train"), ("2007", "val"), ("2007", "test")):
        split_file = vocdevkit / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
        image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        image_out = voc_root / "images" / f"{split}{year}"
        label_out = voc_root / "labels" / f"{split}{year}"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        instances = 0
        for index, image_id in enumerate(image_ids, start=1):
            src = vocdevkit / f"VOC{year}" / "JPEGImages" / f"{image_id}.jpg"
            dst = image_out / f"{image_id}.jpg"
            move_or_copy_image(src, dst)
            instances += convert_label(vocdevkit, label_out / f"{image_id}.txt", year, image_id)
            if index % 1000 == 0:
                print(f"[convert] {split}{year}: {index}/{len(image_ids)}")

        total_images[f"{split}{year}"] = len(image_ids)
        total_instances[f"{split}{year}"] = instances
        print(f"[convert] {split}{year}: images={len(image_ids)} instances={instances}")

    data = {
        "path": str(voc_root.resolve()),
        "train": ["images/train2012", "images/train2007", "images/val2012", "images/val2007"],
        "val": ["images/test2007"],
        "test": ["images/test2007"],
        "names": {idx: name for idx, name in enumerate(ULTRALYTICS_NAMES)},
    }
    (voc_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"[ultralytics] train_images={sum(total_images[k] for k in ('train2012', 'train2007', 'val2012', 'val2007'))}")
    print(f"[ultralytics] val_images={total_images['test2007']}")


def parse_yolo_line(line: str) -> tuple[int, str]:
    parts = line.strip().split(maxsplit=1)
    return int(parts[0]), parts[1]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def add_slice_records(voc_root: Path, out_root: Path, source_split: str, target_split: str, keep: set[int]) -> tuple[int, int]:
    source_images = voc_root / "images" / source_split
    source_labels = voc_root / "labels" / source_split
    image_count = 0
    instance_count = 0
    local_index = {global_id: local_id for local_id, global_id in enumerate(sorted(keep))}

    for label_path in sorted(source_labels.glob("*.txt")):
        remapped = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ultra_id, rest = parse_yolo_line(line)
            class_name = ULTRALYTICS_NAMES[ultra_id]
            paper_id = PAPER_NAMES.index(class_name)
            if paper_id not in keep:
                continue
            remapped.append(f"{local_index[paper_id]} {rest}")
        if not remapped:
            continue
        image_src = source_images / f"{label_path.stem}.jpg"
        image_dst = out_root / "images" / target_split / f"{source_split}_{label_path.name}".replace(".txt", ".jpg")
        label_dst = out_root / "labels" / target_split / f"{source_split}_{label_path.name}"
        link_or_copy(image_src, image_dst)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("\n".join(remapped) + "\n", encoding="utf-8")
        image_count += 1
        instance_count += len(remapped)
    return image_count, instance_count


def build_paper_slices(voc_root: Path, output_root: Path) -> None:
    for name, keep in SLICES.items():
        out_root = output_root / name
        if out_root.exists():
            shutil.rmtree(out_root)
        train_images = 0
        train_instances = 0
        for source_split in ("train2012", "train2007", "val2012", "val2007"):
            images, instances = add_slice_records(voc_root, out_root, source_split, "train", keep)
            train_images += images
            train_instances += instances
        val_images, val_instances = add_slice_records(voc_root, out_root, "test2007", "val", keep)
        data = {
            "path": str(out_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/val",
            "nc": len(sorted(keep)),
            "names": {idx: PAPER_NAMES[global_id] for idx, global_id in enumerate(sorted(keep))},
            "global_class_indices": sorted(keep),
        }
        (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(
            f"[slice] {name}: train_images={train_images} train_instances={train_instances} "
            f"val_images={val_images} val_instances={val_instances}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", default="data/VOC")
    parser.add_argument("--download-dir", default="data/downloads")
    parser.add_argument("--slices-root", default="data/pascal_voc_yolo")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    voc_root = Path(args.voc_root)
    download_dir = Path(args.download_dir)
    voc_root.mkdir(parents=True, exist_ok=True)

    for filename, meta in ZIPS.items():
        zip_path = download_dir / filename
        if not args.skip_download:
            download_file(meta["url"], zip_path, meta["size"])
        extract_zip(zip_path, voc_root)

    build_ultralytics_voc(voc_root)
    build_paper_slices(voc_root, Path(args.slices_root))
    print("[done]")


if __name__ == "__main__":
    main()
