from __future__ import annotations

"""
把 dual-incremental-detection/train_new1.py 中已经跑通的训练方式，
接入当前项目的 DuET 复现逻辑。

这个脚本和原 train_ultralytics_duet.py 的区别：
1. 训练前始终构建一个“全量 20 类检测头”的 YOLO 模型。
2. 第一阶段只从 yolo11n.pt 加载 backbone/neck，检测头保持 20 类结构。
3. 后续阶段沿用上一轮 DuET 合并后的 checkpoint 继续训练。
4. 增量阶段训练时接入 duet_loss.py 中的蒸馏损失和方向一致性损失。
5. 每轮训练结束后用 duet_module.py 做共享参数动态融合，并按 class_indices
   把旧任务/新任务的分类头切片写回全量 20 类检测头。

适用场景：
你的数据 yaml 可能每个任务只含一部分类别，但最终希望模型一直保持 20 类输出。
"""

import argparse
import atexit
from datetime import datetime
import json
import multiprocessing
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

# 允许脚本放在项目根目录或 status3/ 这类子目录中运行。
# 注意：Python 包名不能包含横杠，所以不要写成
# from dual-incremental-detection-master.xxx import ...
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if not (PROJECT_ROOT / "duet_repro").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from duet_repro.core.duet_module import merge_state_dicts_with_duet_module
from duet_repro.core.task_vectors import (
    StateDict,
    inject_state_dict_into_checkpoint,
    load_state_dict,
    task_vector,
    torch_load_checkpoint,
)
from train_ultralytics_duet import DuETDetectionTrainer


# Windows + PyTorch/Ultralytics 常见兼容项，继承 train_new1.py 的稳定设置。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_SHM_DISABLED"] = "1"


class TeeStream:
    """把控制台输出同时写入原始终端和日志文件。"""

    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, text: str) -> int:
        """终端完整显示，日志文件过滤动态进度条。"""
        self.console_stream.write(text)
        for piece in text.splitlines(True):
            if not self._is_dynamic_progress(piece):
                self.log_stream.write(piece)
        return len(text)

    @staticmethod
    def _is_dynamic_progress(text: str) -> bool:
        """
        过滤 tqdm/Ultralytics 的动态进度条。

        这些输出通常带有回车符或清行控制符，保存到 txt 后会非常冗长；
        过滤后仍保留每轮最终的 mAP 表格和普通日志。
        """
        return "\r" in text or "\x1b[K" in text

    def flush(self) -> None:
        """同时刷新终端和 txt 文件，避免长训练时日志滞后。"""
        self.console_stream.flush()
        self.log_stream.flush()

    def isatty(self) -> bool:
        """保持 tqdm/Ultralytics 对终端类型的判断。"""
        return self.console_stream.isatty()

    def __getattr__(self, name: str):
        """其他终端属性继续交给原始输出流。"""
        return getattr(self.console_stream, name)


def setup_console_txt_log(output_dir: Path) -> Path:
    """
    把本次训练的控制台输出同步保存成 txt。

    保存位置：
      output_dir/logs/train_YYYYmmdd_HHMMSS.txt
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def close_log_file() -> None:
        """程序结束时恢复输出流并关闭日志文件。"""
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    atexit.register(close_log_file)
    print(f"[日志] 本次训练控制台输出将同步保存到: {log_path}")
    return log_path


def seed_everything(seed: int) -> None:
    """固定随机种子，方便多阶段实验复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    """读取实验 yaml 配置。"""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_names(names: dict | list) -> dict[int, str]:
    """把 yaml 中的类别名统一转换为 Ultralytics 需要的 {int: str} 格式。"""
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
REFERENCE_INIT_VERSION = "pretrained_full_head_v2"


def safe_task_name(name: str) -> str:
    """把任务名转换成适合作为目录名的短字符串。"""
    keep = []
    for char in str(name):
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(keep).strip("_") or "task"


def resolve_dataset_root(data_yaml: Path, data_cfg: dict) -> Path:
    """解析 YOLO data.yaml 中的 path 字段，得到数据集根目录。"""
    root = Path(data_cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_sources(data_yaml: Path, data_cfg: dict, split: str) -> list[Path]:
    """
    解析 train/val/test 对应的图片来源。

    Ultralytics 支持 split 写成单个目录、txt 文件或路径列表，这里统一转成 Path 列表。
    """
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
        # 不能用 Path.resolve()，否则图片目录或图片文件是软链接时会跳回全量数据集，
        # 后续 infer_label_path() 就会误读全量 labels，而不是当前切片 labels。
        sources.append(Path(os.path.abspath(source)))
    return sources


def iter_split_images(source: Path) -> list[tuple[Path, Path]]:
    """
    枚举一个 split 中的图片。

    返回值是 (原始图片路径, 在准备后数据集中的相对路径)。目录输入会保留子目录结构；
    txt 输入会按行读取图片路径，并用序号避免同名图片互相覆盖。
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

    raise FileNotFoundError(f"找不到 data.yaml 中声明的图片来源: {source}")


def infer_label_path(image_path: Path) -> Path:
    """
    根据 YOLO 目录约定从图片路径推断标签路径。

    常见结构是 images/train/xxx.jpg 对应 labels/train/xxx.txt。
    """
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_label_classes(label_path: Path) -> set[int]:
    """读取一个 YOLO 标签文件中的类别编号。"""
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
    """
    判断原始标签是局部编号还是全局编号。

    local 表示标签里的 0、1、2... 是“当前任务内部类别下标”，需要映射到 class_indices；
    global 表示标签已经是 detector.names 中的全局类别下标，可以直接使用。

    如果 auto 模式下发现标签编号超过当前任务类别数，但仍落在 total_classes 范围内，
    就按 global 处理。这正是服务器报错中“标签 13 出现在 10 类任务里”的情况。
    """
    normalized_mode = str(mode).lower()
    if normalized_mode in {"global", "true", "yes", "1"}:
        return "global"
    if normalized_mode in {"local", "false", "no", "0"}:
        return "local"
    if normalized_mode != "auto":
        raise ValueError("labels_are_global 只能填写 auto、true/global 或 false/local")

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
        f"标签类别编号 {sorted(label_classes)} 既不是当前任务局部编号，"
        f"也没有落在全局 [0, {total_classes - 1}] 范围内。"
    )


def resolve_task_class_indices(task: dict, data_cfg: dict, cfg: dict) -> list[int]:
    """
    解析当前任务真正对应的全局类别通道。

    优先使用任务配置里的 class_indices；如果当前 data.yaml 的 names 数量正好等于
    当前任务类别数，则再按类别名核对一次，避免 data.yaml 的类别顺序和手写下标不一致。
    """
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
        print(
            f"[数据准备] {task['name']} 的 class_indices 与 data.yaml names 顺序不一致，"
            f"已按类别名修正为: {derived}"
        )
    return derived


def link_or_copy_image(src: Path, dst: Path) -> None:
    """优先用软链接复用图片；如果系统不允许创建软链接，就退回到复制图片。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_label_file(src: Path, dst: Path, class_indices: list[int], label_space: str) -> None:
    """
    把当前任务标签写入全局类别空间。

    如果原标签是 local，第 0 类会被映射到 class_indices[0]；
    如果原标签已经是 global，则只复制类别编号和框坐标。
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
                raise ValueError(
                    f"标签 {src}:{line_number} 的局部类别 {class_id} 超出 class_indices 范围 {class_indices}"
                )
            class_id = class_indices[class_id]
        items[0] = str(class_id)
        remapped_lines.append(" ".join(items))

    dst.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")


def prepare_global_task_data(task: dict, cfg: dict, output_dir: Path) -> Path:
    """
    为当前任务生成全局类别空间下的 data.yaml。

    这是为了和论文的 Incremental Head 思路对齐：每个阶段的模型都保持同一个全局类别头，
    当前任务的标签被映射到它负责的全局通道，后续才能按 class_indices 做检测头切片保留。
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
        raise ValueError("detector.names 的数量必须和 detector.total_classes 一致")
    if any(index < 0 or index >= total_classes for index in class_indices):
        raise ValueError(f"class_indices 必须落在 [0, {total_classes - 1}] 范围内: {class_indices}")

    prepared_root = output_dir / "prepared_data" / safe_task_name(task["name"])
    if prepared_root.exists():
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
        raise ValueError(f"{original_yaml} 中没有可用的 train 图片路径")

    label_space = detect_label_space(label_classes, class_indices, task.get("labels_are_global", "auto"), total_classes)
    if label_space == "global" and label_classes:
        actual_indices = sorted(label_classes)
        unexpected = sorted(set(actual_indices) - set(class_indices))
        if unexpected:
            print(
                f"[数据准备] {task['name']} 的标签已经是全局编号，"
                f"且出现了配置 class_indices 之外的类别 {unexpected}。"
            )
            print(f"[数据准备] {task['name']} 已按标签实际出现的全局类别修正为: {actual_indices}")
            class_indices = actual_indices

    print(f"[数据准备] {task['name']} 标签编号模式: {label_space}，将写入全局 {total_classes} 类 data.yaml")

    prepared_cfg = {
        "path": str(prepared_root.resolve()),
        "train": "images/train",
        "nc": total_classes,
        "names": names,
    }
    for split, records in split_records.items():
        prepared_cfg[split] = f"images/{split}"
        for image_path, relative_path in records:
            dst_image = prepared_root / "images" / split / relative_path
            dst_label = prepared_root / "labels" / split / relative_path.with_suffix(".txt")
            link_or_copy_image(image_path, dst_image)
            remap_label_file(infer_label_path(image_path), dst_label, class_indices, label_space)

    if "channels" in data_cfg:
        prepared_cfg["channels"] = data_cfg["channels"]

    prepared_yaml = prepared_root / "data.yaml"
    prepared_yaml.write_text(
        yaml.safe_dump(prepared_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    task["prepared_data"] = str(prepared_yaml)
    task["label_space"] = label_space
    task["resolved_class_indices"] = class_indices
    return prepared_yaml


def build_full_head_model(cfg: dict):
    """
    构建全量类别检测头模型。

    train_new1.py 的核心做法是：不用 YOLO(data.yaml) 自动推断当前任务类别数，
    而是手动把模型 nc 固定为 total_classes=20。这样每个阶段训练时，模型输出
    维度始终是全局 20 类，后续才能按类别切片做无样本融合。
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    detector = cfg["detector"]
    model_yaml = str(detector.get("model_yaml", "yolo11n.yaml"))
    names = normalize_names(detector["names"])
    total_classes = int(detector["total_classes"])

    model_cfg = yaml_model_load(model_yaml)
    model_cfg["nc"] = total_classes

    model = YOLO(model_yaml)
    model.model = DetectionModel(model_cfg, nc=total_classes, verbose=False)
    model.model.names = names
    model.model.task = "detect"
    model.model.args = {"model": model_yaml, "task": "detect"}
    model.model.yaml["nc"] = total_classes
    model.overrides.pop("nc", None)
    if model.ckpt is None:
        model.ckpt = {}
    if hasattr(model.model, "nc"):
        model.model.nc = total_classes

    return model


CLASS_NAME_ALIASES = {
    "airplane": "aeroplane",
    "aeroplane": "aeroplane",
    "motorcycle": "motorbike",
    "motorbike": "motorbike",
    "diningtable": "diningtable",
    "dining table": "diningtable",
    "pottedplant": "pottedplant",
    "potted plant": "pottedplant",
    "couch": "sofa",
    "sofa": "sofa",
    "tv": "tvmonitor",
    "tvmonitor": "tvmonitor",
}


def canonical_class_name(name: str) -> str:
    """把 COCO/VOC 中写法不同但语义相同的类别名归一化。"""
    compact = " ".join(str(name).lower().replace("_", " ").split())
    if compact in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[compact]
    return "".join(ch for ch in compact if ch.isalnum())


def extract_checkpoint_names(ckpt) -> dict[int, str]:
    """从 Ultralytics checkpoint 中读取预训练类别名。"""
    candidate = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    names = getattr(candidate, "names", None)
    if names is None and isinstance(ckpt, dict):
        names = ckpt.get("names")
    if names is None:
        return {}
    return normalize_names(names)


def copy_class_output_rows(
    target_value: torch.Tensor,
    pretrained_value: torch.Tensor,
    target_names: dict[int, str],
    pretrained_names: dict[int, str],
) -> int:
    """
    将 COCO 预训练分类输出层中可匹配的类别通道拷贝到当前全局 head。

    YOLO11n.pt 的分类输出通常是 80 类，而论文 Pascal 场景需要 20 类。
    整层形状不一致时不能直接 load_state_dict，但可以按类别名拷贝对应行。
    """
    if target_value.dim() == 4 and pretrained_value.dim() == 4:
        if target_value.shape[1:] != pretrained_value.shape[1:]:
            return 0
    elif target_value.dim() == 1 and pretrained_value.dim() == 1:
        pass
    else:
        return 0

    source_by_name = {canonical_class_name(name): idx for idx, name in pretrained_names.items()}
    copied = 0
    for target_idx, target_name in target_names.items():
        source_idx = source_by_name.get(canonical_class_name(target_name))
        if source_idx is None:
            continue
        if target_idx >= target_value.shape[0] or source_idx >= pretrained_value.shape[0]:
            continue
        target_value[target_idx] = pretrained_value[source_idx].to(target_value.device, dtype=target_value.dtype)
        copied += 1
    return copied


def load_pretrained_full_head_weights(model, weights: str | Path) -> None:
    """
    从 YOLO 预训练权重初始化当前全局检测器。

    论文 Algorithm 1 是从预训练 detector 权重 theta_0 出发。这里会：
      1. 加载所有形状完全一致的参数，包括 box head 和 dfl；
      2. 对形状不一致的 cv3 最终分类输出层，按类别名拷贝可对应的通道。
    """
    ckpt = torch_load_checkpoint(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        official_state = ckpt["model"].state_dict()
    elif hasattr(ckpt, "state_dict"):
        official_state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        official_state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")

    model_state = model.model.state_dict()
    target_names = normalize_names(model.model.names)
    pretrained_names = extract_checkpoint_names(ckpt)
    copied_exact = 0
    copied_class_rows = 0

    for key in list(model_state):
        if key not in official_state:
            continue
        pretrained_value = official_state[key]
        if model_state[key].shape == pretrained_value.shape:
            model_state[key] = pretrained_value.clone()
            copied_exact += 1
            continue
        if _is_class_output_key(key) and pretrained_names:
            copied_class_rows += copy_class_output_rows(
                model_state[key],
                pretrained_value,
                target_names,
                pretrained_names,
            )
    model.model.load_state_dict(model_state, strict=False)
    print(
        f"[DuET reference] loaded exact tensors={copied_exact}, "
        f"class-output rows copied={copied_class_rows} from {weights}"
    )


def save_reference_checkpoint(cfg: dict, output_dir: Path) -> Path:
    """
    保存 DuET task vector 的 reference checkpoint。

    DuET 的 task vector 是 theta_task - theta_reference。这里的 reference 不是
    原始 yolo11n.pt，而是“20 类全量检测头 + 预训练 backbone/neck”的同结构模型。
    这样后续每个任务 checkpoint 都能和 reference 按相同 shape 做向量计算。
    """
    reference_path = output_dir / "reference_full_head.pt"
    marker_path = output_dir / "reference_full_head.init_version"
    total_classes = int(cfg["detector"]["total_classes"])
    marker_ok = marker_path.exists() and marker_path.read_text(encoding="utf-8").strip() == REFERENCE_INIT_VERSION
    if reference_path.exists() and marker_ok and checkpoint_has_full_class_head(reference_path, total_classes):
        return reference_path
    if reference_path.exists():
        print(f"[DuET reference] 发现旧 reference 初始化版本不匹配，正在重新生成: {reference_path}")

    model = build_full_head_model(cfg)
    load_pretrained_full_head_weights(model, cfg["detector"]["base_weights"])
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(reference_path)
    marker_path.write_text(REFERENCE_INIT_VERSION + "\n", encoding="utf-8")
    return reference_path


def configure_full_names(model, cfg: dict) -> None:
    """加载已有 checkpoint 后，重新绑定全局类别名和 total_classes。"""
    names = normalize_names(cfg["detector"]["names"])
    total_classes = int(cfg["detector"]["total_classes"])
    model.model.names = names
    if hasattr(model.model, "yaml"):
        model.model.yaml["nc"] = total_classes
    if hasattr(model.model, "args") and isinstance(model.model.args, dict):
        # 这里不能把 nc 写入 args。Ultralytics 会把 args 当训练参数校验，nc 不是合法训练参数。
        model.model.args.pop("nc", None)
    if hasattr(model, "overrides") and isinstance(model.overrides, dict):
        model.overrides.pop("nc", None)
    if hasattr(model.model, "nc"):
        model.model.nc = total_classes


def checkpoint_has_full_class_head(path: Path, total_classes: int) -> bool:
    """检查已有 reference checkpoint 是否真的是全局类别检测头。"""
    try:
        state = load_state_dict(path)
    except Exception:
        return False
    class_head_keys = [key for key in state if _is_class_output_key(key)]
    if not class_head_keys:
        return False
    return all(state[key].shape[0] == total_classes for key in class_head_keys)


def train_one_task(
    weights: str | Path,
    task: dict,
    cfg: dict,
    output_dir: Path,
    *,
    is_first: bool,
    teacher_ckpt: str | Path | None,
    prev_task_vector: StateDict | None,
    task_vector_history: list[StateDict],
    reference_state: StateDict,
    old_class_indices: Iterable[int] | None = None,
) -> Path:
    """
    训练单个阶段。

    第一阶段：
    - 从 reference_full_head.pt 加载 20 类全量头模型；
    - 不使用 teacher。

    增量阶段：
    - 从上一轮 DuET merged checkpoint 加载；
    - 使用 old_ckpt 作为 teacher；
    - 在 on_train_start 回调中把 Ultralytics 默认 loss 替换为 DuETDetectionLoss。
    """
    from duet_repro.core.duet_loss import create_duet_criterion
    from ultralytics import YOLO

    prepared_data = Path(task.get("prepared_data") or prepare_global_task_data(task, cfg, output_dir))

    if is_first:
        # 第一阶段从 reference_full_head.pt 开始，确保结构就是全局 20 类检测头。
        model = YOLO(str(weights))
        configure_full_names(model, cfg)
    else:
        # 后续阶段从上一轮合并模型继续训练，但仍保持全局 20 类 names/nc。
        model = YOLO(str(weights))
        configure_full_names(model, cfg)

    training = cfg["training"]
    duet_cfg = cfg.get("duet", {})
    distill_weight = float(duet_cfg.get("distill_weight", 0.0))
    dc_weight = float(duet_cfg.get("dc_weight", 0.0))

    if teacher_ckpt is not None and (distill_weight > 0 or dc_weight > 0):
        # teacher 是上一轮 merged checkpoint，用于旧知识蒸馏。
        teacher = YOLO(str(teacher_ckpt)).model
        teacher.to(training.get("device", 0))
        criterion = create_duet_criterion(
            model=model.model,
            teacher_model=teacher,
            distill_weight=distill_weight,
            dc_weight=dc_weight,
            distill_temperature=float(duet_cfg.get("distill_temperature", 2.0)),
            old_class_indices=list(old_class_indices) if old_class_indices is not None else None,
        )
        if prev_task_vector is not None:
            # prev_task_vector 用作方向一致性损失的当前历史方向。
            criterion.set_curr_task_vector(prev_task_vector)
        for vector in task_vector_history:
            # 历史 task vectors 用于 LDC 判断新旧更新方向是否冲突。
            criterion.record_task_vector(vector)

        def on_train_start(trainer):
            """Ultralytics 真正创建 trainer 后，再替换 criterion。"""
            from ultralytics.utils.torch_utils import unwrap_model

            original_model = unwrap_model(trainer.model)
            if hasattr(original_model, "args"):
                criterion.hyp = original_model.args
            original_model.criterion = criterion
            if trainer.ema and hasattr(trainer.ema, "ema"):
                # EMA 模型也要挂同一个 criterion，避免训练过程中仍走默认 loss。
                unwrap_model(trainer.ema.ema).criterion = criterion
            loss_names = list(trainer.loss_names)
            if distill_weight > 0:
                loss_names.append("distill_loss")
            if dc_weight > 0:
                loss_names.append("dc_loss")
            trainer.loss_names = loss_names

        model.add_callback("on_train_start", on_train_start)

    model.train(
        # 这里基本继承 train_new1.py 的训练超参：
        # AdamW、cos_lr、warmup、不同阶段 weight_decay 都从配置读取。
        data=str(prepared_data),
        epochs=training.get("epochs", 30),
        warmup_epochs=training.get("warmup_epochs", 5),
        batch=training.get("batch", 64),
        imgsz=training.get("imgsz", 640),
        device=training.get("device", 0),
        optimizer=training.get("optimizer", "AdamW"),
        cos_lr=training.get("cos_lr", True),
        lr0=training.get("lr0", 0.01),
        lrf=training.get("lrf", 0.01),
        weight_decay=training.get("weight_decay_first" if is_first else "weight_decay_incremental", 0.0005),
        freeze=training.get("freeze", 0),
        workers=training.get("workers", 0),
        project=str(output_dir / "runs"),
        name=task["name"],
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    if not best.exists():
        raise FileNotFoundError(f"Ultralytics did not produce best checkpoint: {best}")
    return best


def _is_class_output_key(key: str) -> bool:
    """判断 state_dict key 是否是 YOLO cv3 分类输出层的 weight/bias。"""
    lowered = key.lower()
    return "cv3" in lowered and (lowered.endswith(".2.weight") or lowered.endswith(".2.bias"))


def merge_full_head_slices(
    merged_state: StateDict,
    old_state: StateDict,
    new_state: StateDict,
    learned_indices: Iterable[int],
    current_indices: Iterable[int],
) -> StateDict:
    """
    在全量 20 类检测头里保留各任务学到的分类头切片。

    merge_state_dicts_with_duet_module 会对共享参数做动态融合；检测头属于任务特定参数，
    对 train_new1.py 这种“全量头 + 任务类别切片”的训练方式，不能简单整层采用新任务头，
    否则旧任务类别通道会被新阶段训练覆盖。

    learned_indices: 之前任务已经学过的全局类别下标。
    current_indices: 当前任务正在学习的全局类别下标。
    """
    learned = sorted({int(i) for i in learned_indices})
    current = sorted({int(i) for i in current_indices})
    result = {key: value.detach().clone() for key, value in merged_state.items()}

    for key, value in list(result.items()):
        if not _is_class_output_key(key):
            continue

        if key in old_state:
            # 旧任务类别通道来自上一轮 merged checkpoint。
            for idx in learned:
                if idx < value.shape[0] and idx < old_state[key].shape[0]:
                    value[idx] = old_state[key][idx].to(value.device, dtype=value.dtype)

        if key in new_state:
            # 当前任务类别通道来自本轮训练后的 checkpoint。
            for idx in current:
                if idx < value.shape[0] and idx < new_state[key].shape[0]:
                    value[idx] = new_state[key][idx].to(value.device, dtype=value.dtype)

        result[key] = value

    return result


def print_run_guide(cfg: dict) -> None:
    """
    把配置中的任务顺序打印成直观的 T1/T2/T3... 主控流程。

    这样代码仍然保持“改 yaml 就能换实验场景”，但运行时能像 train_new1.py 那样
    明确看到当前到底会先跑哪个阶段、每个阶段用哪个 data.yaml、对应全局类别通道是谁。
    """
    print("\n" + "=" * 72)
    print("【运行说明】当前脚本会按下面顺序执行：")
    print("  第一部分：顺序训练每个任务，增量阶段会自动接入 DuET 蒸馏/DC loss")
    print("  第二部分：每个任务训练结束后，立刻执行 DuET Module 动态融合")
    print("  第三部分：检测头按 class_indices 做通道切片保留，避免旧类被覆盖")
    print("-" * 72)
    for idx, task in enumerate(cfg["tasks"], start=1):
        class_indices = ", ".join(str(i) for i in task.get("class_indices", []))
        first_flag = "是，使用 yolo11n.pt 初始化 backbone/neck" if idx == 1 else "否，使用上一阶段 merged checkpoint"
        print(f"T{idx}: {task['name']}")
        print(f"    data.yaml      : {task['data']}")
        print(f"    class_indices  : [{class_indices}]")
        print(f"    是否第一阶段   : {first_flag}")
    print("=" * 72 + "\n")


def write_experiment_state(
    output_dir: Path,
    cfg: dict,
    history: list[dict],
    *,
    reference_ckpt: str | Path,
    latest_ckpt: str | Path | None,
) -> None:
    """
    保存训练过程元信息，保证后续能追溯并计算论文指标。

    论文中的 Avg RI / Avg GI / RAI 不是只看最终模型即可，还需要知道：
    1. 每个任务第一次学习完成时的 checkpoint；
    2. 每个阶段 DuET 合并后的 checkpoint；
    3. 每个任务的 data.yaml 和全局类别通道 class_indices；
    4. task vector 使用的 reference checkpoint。

    所以这里每完成一个阶段就写一次，长训练中断后也能保留已完成阶段记录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "reference_checkpoint": str(reference_ckpt),
        "latest_checkpoint": str(latest_ckpt) if latest_ckpt is not None else None,
        "output_dir": str(output_dir),
        "detector": cfg.get("detector", {}),
        "training": cfg.get("training", {}),
        "duet": cfg.get("duet", {}),
        "tasks": cfg.get("tasks", []),
        "history": history,
        "metric_note": {
            "Avg_RI": "Final-task mAP on old classes / mAP when those classes were first learned.",
            "Avg_GI": "Merged-checkpoint mAP on unseen class/domain slices / reference mAP for those unseen slices.",
            "RAI": "(Avg_RI + Avg_GI) / 2.",
        },
    }
    (output_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def resolve_config_path(config_path: str | Path) -> Path:
    """优先按当前工作目录、status2 目录、项目根目录依次解析配置路径。"""
    path = Path(config_path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, PROJECT_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (SCRIPT_DIR / path).resolve()


def main(config_path: str = "configs/train_weather_2phase_full.yaml") -> None:
    multiprocessing.freeze_support()
    # 保证相对路径如 yolo11n.pt、outputs/ 都以项目根目录为基准。
    config_path = resolve_config_path(config_path)
    os.chdir(PROJECT_ROOT)

    cfg = load_config(config_path)
    seed_everything(int(cfg["experiment"].get("seed", 42)))

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_console_txt_log(output_dir)

    print_run_guide(cfg)

    # =========================================================================
    # 【第一部分：准备 DuET reference 模型】
    # reference = 20 类全量头 + 预训练 backbone/neck。
    # 后续所有 task vector 都是：当前任务模型 - reference。
    # =========================================================================
    reference_ckpt = save_reference_checkpoint(cfg, output_dir)
    reference_state = load_state_dict(reference_ckpt)

    duet_cfg = cfg.get("duet", {})
    shared_key_exclude = tuple(duet_cfg.get("shared_key_exclude", []))
    if float(duet_cfg.get("dc_weight", 0.0)) > 0 or float(duet_cfg.get("distill_weight", 0.0)) > 0:
        from ultralytics.models.yolo.detect.train import DetectionTrainer

        # patch optimizer_step：每次参数更新后刷新当前 task vector，
        # 供 DuETDetectionLoss 中的方向一致性损失使用。
        DuETDetectionTrainer(
            base_trainer_cls=DetectionTrainer,
            reference_state=reference_state,
            shared_key_exclude=shared_key_exclude,
        )

    history: list[dict] = []
    task_vector_history: list[StateDict] = []
    learned_indices: list[int] = []
    current_weights: str | Path = reference_ckpt
    old_ckpt: str | Path | None = None
    write_experiment_state(
        output_dir,
        cfg,
        history,
        reference_ckpt=reference_ckpt,
        latest_ckpt=old_ckpt,
    )

    # =========================================================================
    # 【第二部分：增量顺序训练 + 每阶段 DuET 合并】
    # 你想跑论文第一个场景/第二个场景时，不需要改这里的 Python 逻辑；
    # 只需要在 configs/train_new1_duet_yolo.yaml 中替换 tasks 列表。
    # =========================================================================
    for task_index, task in enumerate(cfg["tasks"], start=1):
        prepared_data = prepare_global_task_data(task, cfg, output_dir)
        # class_indices 是当前任务类别在全局 20 类头中的通道位置。
        current_indices = [int(i) for i in task.get("resolved_class_indices", task.get("class_indices", []))]
        print("\n" + "=" * 72)
        print(f"【阶段 T{task_index}】开始训练：{task['name']}")
        print(f"原始 data.yaml: {task['data']}")
        print(f"全局 data.yaml: {prepared_data}")
        print(f"全局类别通道 class_indices: {current_indices}")
        print("=" * 72)

        prev_tv = None
        if old_ckpt is not None:
            # 上一轮 merged 模型相对 reference 的 task vector。
            prev_tv = task_vector(reference_state, load_state_dict(old_ckpt), shared_key_exclude=shared_key_exclude)

        trained_ckpt = train_one_task(
            current_weights,
            task,
            cfg,
            output_dir,
            is_first=task_index == 1,
            teacher_ckpt=old_ckpt,
            prev_task_vector=prev_tv,
            task_vector_history=task_vector_history,
            reference_state=reference_state,
            old_class_indices=learned_indices,
        )

        if task_index == 1 or not duet_cfg.get("enabled", True):
            # 第一阶段没有旧任务，直接把训练后的模型作为 merged checkpoint。
            print(f"【阶段 T{task_index}】第一阶段无旧模型，跳过 DuET merge，直接保存 best checkpoint。")
            merged_state = load_state_dict(trained_ckpt)
            merged_ckpt = output_dir / f"task_{task_index}_{task['name']}_best.pt"
        else:
            # 增量阶段：共享层走 DuET Module 动态融合。
            print(f"【阶段 T{task_index}】进入 DuET 融合：共享层动态融合 + 检测头通道切片保留。")
            old_state = load_state_dict(old_ckpt)
            new_state = load_state_dict(trained_ckpt)
            merged_state, report = merge_state_dicts_with_duet_module(
                reference_state,
                old_state,
                new_state,
                gamma=float(duet_cfg.get("gamma", 0.1)),
                alpha_base=float(duet_cfg.get("alpha_base", 0.5)),
                shared_key_exclude=shared_key_exclude,
                per_layer_report=duet_cfg.get("verbose_merge", False),
            )
            # 检测头分类输出层按全局类别通道切片，保留旧类并写入新类。
            merged_state = merge_full_head_slices(
                merged_state,
                old_state,
                new_state,
                learned_indices=learned_indices,
                current_indices=current_indices,
            )
            print(
                "[DuET train_new1] merged_keys={0} skipped_keys={1}".format(
                    report["merged_keys"],
                    report["skipped_keys"],
                )
            )
            merged_ckpt = output_dir / f"task_{task_index}_{task['name']}_duet.pt"

        inject_state_dict_into_checkpoint(trained_ckpt, merged_state, merged_ckpt)

        # 记录已经学习过的全局类别通道，供下一轮 head slice 合并使用。
        learned_indices = sorted(set(learned_indices) | set(current_indices))
        current_task_tv = task_vector(reference_state, merged_state, shared_key_exclude=shared_key_exclude)
        task_vector_history.append(current_task_tv)
        current_weights = merged_ckpt
        old_ckpt = merged_ckpt
        print(f"【阶段 T{task_index}】完成。merged checkpoint: {merged_ckpt}")

        history.append(
            {
                "task_index": task_index,
                "task": task["name"],
                "data": task["data"],
                "prepared_data": task.get("prepared_data"),
                "label_space": task.get("label_space"),
                "class_indices": current_indices,
                "learned_indices": learned_indices,
                "trained_checkpoint": str(trained_ckpt),
                "merged_checkpoint": str(merged_ckpt),
                "is_first_task": task_index == 1,
            }
        )
        write_experiment_state(
            output_dir,
            cfg,
            history,
            reference_ckpt=reference_ckpt,
            latest_ckpt=old_ckpt,
        )

    history_path = output_dir / "training_history.json"
    print(f"\n[DuET train_new1] Done. History saved to {history_path}")
    print(f"[DuET train_new1] Final checkpoint: {old_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_weather_2phase_full.yaml")
    args = parser.parse_args()
    main(args.config)
