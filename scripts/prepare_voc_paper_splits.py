from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


VOC_CLASSES = [
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

DOWNLOADS = {
    "VOCtrainval_06-Nov-2007.tar": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "size": 460032000,
    },
    "VOCtest_06-Nov-2007.tar": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "size": 451020800,
    },
    "VOCtrainval_11-May-2012.tar": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "size": 1999639040,
    },
}

TASKS = {
    "voc_1_10": list(range(0, 10)),
    "voc_11_20": list(range(10, 20)),
    "voc_1_20": list(range(0, 20)),
}


def download(url: str, target: Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_size:
        print(f"[download] exists: {target}")
        return
    if target.exists():
        print(
            f"[download] incomplete file found, redownloading: "
            f"{target} ({target.stat().st_size} / {expected_size} bytes)"
        )
        target.unlink()
    print(f"[download] {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
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


def extract_tar(tar_path: Path, raw_dir: Path) -> None:
    voc2007 = raw_dir / "VOCdevkit" / "VOC2007"
    voc2012 = raw_dir / "VOCdevkit" / "VOC2012"
    if tar_path.name == "VOCtrainval_06-Nov-2007.tar":
        if (voc2007 / "ImageSets" / "Main" / "trainval.txt").exists():
            print(f"[extract] complete, skip: {voc2007} trainval")
            return
    elif tar_path.name == "VOCtest_06-Nov-2007.tar":
        if (voc2007 / "ImageSets" / "Main" / "test.txt").exists():
            print(f"[extract] complete, skip: {voc2007} test")
            return
    elif tar_path.name == "VOCtrainval_11-May-2012.tar":
        if all((voc2012 / subdir).exists() for subdir in ("Annotations", "ImageSets", "JPEGImages")):
            print(f"[extract] complete, skip: {voc2012}")
            return
    print(f"[extract] {tar_path.name}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_dir)


def read_split_ids(voc_root: Path, split: str) -> list[str]:
    path = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_annotation(xml_path: Path, keep_indices: set[int]) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    labels = []

    for obj in root.findall("object"):
        difficult = int(obj.findtext("difficult", default="0"))
        if difficult == 1:
            continue
        name = obj.findtext("name")
        if name not in VOC_CLASSES:
            continue
        class_id = VOC_CLASSES.index(name)
        if class_id not in keep_indices:
            continue
        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        x = ((xmin + xmax) / 2.0) / width
        y = ((ymin + ymax) / 2.0) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        labels.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return labels


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def add_records(
    *,
    voc_root: Path,
    ids: list[str],
    out_root: Path,
    split: str,
    keep_indices: set[int],
    prefix: str,
    copy_images: bool,
) -> tuple[int, int]:
    image_count = 0
    instance_count = 0
    for image_id in ids:
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        labels = parse_annotation(xml_path, keep_indices)
        if not labels:
            continue
        image_src = voc_root / "JPEGImages" / f"{image_id}.jpg"
        image_name = f"{prefix}_{image_id}.jpg"
        label_name = f"{prefix}_{image_id}.txt"
        link_or_copy(image_src, out_root / "images" / split / image_name, copy_images)
        label_dst = out_root / "labels" / split / label_name
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("\n".join(labels) + "\n", encoding="utf-8")
        image_count += 1
        instance_count += len(labels)
    return image_count, instance_count


def write_data_yaml(out_root: Path) -> None:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "nc": len(VOC_CLASSES),
        "names": {idx: name for idx, name in enumerate(VOC_CLASSES)},
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def prepare_task(raw_dir: Path, output_dir: Path, task_name: str, keep: list[int], copy_images: bool) -> None:
    out_root = output_dir / task_name
    if out_root.exists():
        shutil.rmtree(out_root)
    keep_indices = set(keep)

    voc2007 = raw_dir / "VOCdevkit" / "VOC2007"
    voc2012 = raw_dir / "VOCdevkit" / "VOC2012"

    train_2007 = read_split_ids(voc2007, "trainval")
    train_2012 = read_split_ids(voc2012, "trainval")
    val_2007 = read_split_ids(voc2007, "test")

    train_07 = add_records(
        voc_root=voc2007,
        ids=train_2007,
        out_root=out_root,
        split="train",
        keep_indices=keep_indices,
        prefix="voc2007",
        copy_images=copy_images,
    )
    train_12 = add_records(
        voc_root=voc2012,
        ids=train_2012,
        out_root=out_root,
        split="train",
        keep_indices=keep_indices,
        prefix="voc2012",
        copy_images=copy_images,
    )
    val = add_records(
        voc_root=voc2007,
        ids=val_2007,
        out_root=out_root,
        split="val",
        keep_indices=keep_indices,
        prefix="voc2007test",
        copy_images=copy_images,
    )
    write_data_yaml(out_root)
    print(
        f"[task] {task_name}: "
        f"train_images={train_07[0] + train_12[0]} train_instances={train_07[1] + train_12[1]} "
        f"val_images={val[0]} val_instances={val[1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"F:\DuET_Datasets\pascal_voc")
    parser.add_argument("--output", default=r"F:\DuET_Datasets\pascal_voc_yolo")
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    raw_dir = root / "raw"
    download_dir = Path(args.download_dir) if args.download_dir else root / "downloads"
    output_dir = Path(args.output)

    if not args.skip_download:
        for name, meta in DOWNLOADS.items():
            download(meta["url"], download_dir / name, meta["size"])

    for name in DOWNLOADS:
        extract_tar(download_dir / name, raw_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for task_name, keep in TASKS.items():
        prepare_task(raw_dir, output_dir, task_name, keep, args.copy_images)

    print(f"[done] data yaml files are under: {output_dir}")


if __name__ == "__main__":
    main()
