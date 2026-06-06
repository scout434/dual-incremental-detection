from __future__ import annotations

from pathlib import Path
from typing import Iterable

from duet_repro.core.task_vectors import StateDict, load_state_dict


def is_class_output_key(key: str) -> bool:
    lowered = key.lower()
    return "cv3" in lowered and (lowered.endswith(".2.weight") or lowered.endswith(".2.bias"))


def checkpoint_has_full_class_head(path: Path, total_classes: int) -> bool:
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
