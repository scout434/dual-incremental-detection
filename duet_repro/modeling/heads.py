from __future__ import annotations

from pathlib import Path
from typing import Iterable

from duet_repro.core.task_vectors import StateDict, load_state_dict


def is_class_output_key(key: str) -> bool:
    """判断 state_dict 的某个 key 是否对应 YOLO cv3 分类输出层。"""
    lowered = key.lower()
    return "cv3" in lowered and (lowered.endswith(".2.weight") or lowered.endswith(".2.bias"))


def checkpoint_has_full_class_head(path: Path, total_classes: int) -> bool:
    """检查 checkpoint 的分类输出行数是否已经覆盖完整全局类别数。"""
    state = load_state_dict(path)
    class_head_keys = [key for key in state if is_class_output_key(key)]
    if not class_head_keys:
        return False
    return all(state[key].shape[0] == total_classes for key in class_head_keys)


def merge_full_head_slices(
    merged_state: StateDict,
    old_state: StateDict,
    new_state: StateDict,
    learned_indices: Iterable[int],
    current_indices: Iterable[int],
) -> StateDict:
    """把旧类别和当前类别的分类 head 行合并到同一个完整 head 中。

    merged_state 通常来自 DuET 共享层融合结果；old_state 提供旧类别行，
    new_state 提供当前任务类别行。函数会逐行拷贝 cv3 分类输出，使最终
    checkpoint 同时保留旧类别和新类别的检测能力。
    """
    learned = sorted({int(i) for i in learned_indices})
    current = sorted({int(i) for i in current_indices})
    overlap = sorted(set(learned) & set(current))
    if overlap:
        raise ValueError(f"Incremental classes must be disjoint, but these indices were repeated: {overlap}")

    result = {key: value.detach().clone() for key, value in merged_state.items()}

    preserved_detect_tensors = 0
    for key, value in list(result.items()):
        if not key.startswith("model.23.") or is_class_output_key(key):
            continue
        # 非分类输出的 Detect 参数，如 box/dfl 分支，优先保留旧模型中兼容的参数，
        # 这样可以减少旧类别定位能力在增量合并时被覆盖。
        if key in old_state and old_state[key].shape == value.shape:
            result[key] = old_state[key].detach().clone().to(value.device, dtype=value.dtype)
            preserved_detect_tensors += 1

    copied_old_rows = 0
    copied_current_rows = 0
    class_output_keys = 0

    for key, value in list(result.items()):
        if not is_class_output_key(key):
            continue
        class_output_keys += 1

        # 旧类别行来自旧模型，当前任务类别行来自新模型；两者写入同一个全局 head。
        if key in old_state:
            for idx in learned:
                if idx < value.shape[0] and idx < old_state[key].shape[0]:
                    value[idx] = old_state[key][idx].to(value.device, dtype=value.dtype)
                    copied_old_rows += 1

        if key in new_state:
            for idx in current:
                if idx < value.shape[0] and idx < new_state[key].shape[0]:
                    value[idx] = new_state[key][idx].to(value.device, dtype=value.dtype)
                    copied_current_rows += 1

        result[key] = value

    if class_output_keys == 0:
        raise ValueError("No YOLO cv3 classification output tensors were found during Incremental Head merge.")
    expected_old_rows = class_output_keys * len(learned)
    expected_current_rows = class_output_keys * len(current)
    if copied_old_rows != expected_old_rows or copied_current_rows != expected_current_rows:
        raise ValueError(
            "Incremental Head merge copied an unexpected number of rows: "
            f"old={copied_old_rows}/{expected_old_rows}, current={copied_current_rows}/{expected_current_rows}"
        )
    print(
        f"[Incremental Head] preserved Detect tensors={preserved_detect_tensors}, old rows={copied_old_rows}, "
        f"inserted current rows={copied_current_rows}"
    )
    return result
