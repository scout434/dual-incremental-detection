from __future__ import annotations

"""
DuET 论文指标评估脚本。

这个脚本用来计算 DuET 论文中的三个核心指标：
  Avg RI = mean(最终模型在旧任务上的 mAP / 该任务首次学习完成时的 mAP)
  Avg GI = mean(最终模型在未见类别或未见域上的 mAP / 对应参考模型的 mAP)
  RAI    = (Avg RI + Avg GI) / 2

脚本采用“评估计划文件”驱动，因为论文 Table S1 里的 old/new/unseen
评估切片需要你明确指定。训练脚本可以保存每个阶段的权重，但它无法自动知道
每个 unseen 切片的 data.yaml，也无法自动知道单独训练参考模型得到的 mAP。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if not (PROJECT_ROOT / "duet_repro").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_plan_path(plan_path: str | Path) -> Path:
    """优先按当前工作目录、status2 目录、项目根目录依次解析评估计划路径。"""
    path = Path(plan_path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, PROJECT_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (SCRIPT_DIR / path).resolve()


def load_mapping(path: str | Path) -> dict[str, Any]:
    """读取 JSON 或 YAML 文件，并转换成字典。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def build_checkpoint_aliases(manifest: dict[str, Any] | None, plan: dict[str, Any]) -> dict[str, str]:
    """
    构造权重路径别名，方便在评估计划文件中直接引用。

    从 eval_manifest.json 自动生成的内置别名包括：
      reference -> 训练前保存的完整检测头参考权重
      latest/final -> 最后一个任务融合后的权重
      t1/task_1/<task_name> -> 第 1 个任务融合后的权重
      t1_trained/task_1_trained -> 第 1 个任务直接训练得到的 best.pt
    """
    aliases: dict[str, str] = {}

    if manifest:
        reference = manifest.get("reference_checkpoint")
        latest = manifest.get("latest_checkpoint")
        if reference:
            aliases["reference"] = str(reference)
        if latest:
            aliases["latest"] = str(latest)
            aliases["final"] = str(latest)

        for entry in manifest.get("history", []):
            idx = int(entry.get("task_index", len(aliases) + 1))
            task_name = str(entry.get("task", f"task_{idx}"))
            merged = entry.get("merged_checkpoint")
            trained = entry.get("trained_checkpoint")
            if merged:
                aliases[f"t{idx}"] = str(merged)
                aliases[f"task_{idx}"] = str(merged)
                aliases[task_name] = str(merged)
            if trained:
                aliases[f"t{idx}_trained"] = str(trained)
                aliases[f"task_{idx}_trained"] = str(trained)
                aliases[f"{task_name}_trained"] = str(trained)

    aliases.update({str(k): str(v) for k, v in plan.get("checkpoint_aliases", {}).items()})
    return aliases


def resolve_checkpoint(value: str | Path, aliases: dict[str, str]) -> str:
    """把权重别名解析成真实路径；如果不是别名，就原样当成路径返回。"""
    raw = str(value)
    return aliases.get(raw, raw)


def extract_map(metrics: Any, metric_name: str) -> float:
    """从 Ultralytics 的 DetMetrics 结果中提取 map50 或 map50_95。"""
    box = getattr(metrics, "box", metrics)
    if metric_name in {"map50", "mAP50", "mAP@0.5"}:
        return float(box.map50)
    if metric_name in {"map", "map50_95", "mAP50-95", "mAP@0.5:0.95"}:
        return float(box.map)
    raise ValueError(f"不支持指标 '{metric_name}'。请使用 map50 或 map50_95。")


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_yaml: str | Path,
    *,
    metric_name: str,
    device: str | int,
    imgsz: int | None = None,
    split: str = "val",
    cache: dict[tuple[str, str, str, str], float],
) -> float:
    """运行 YOLO 验证流程，并返回指定的 mAP 数值。"""
    checkpoint = str(checkpoint)
    data_yaml = str(data_yaml)
    key = (checkpoint, data_yaml, metric_name, str(imgsz or "default"))
    if key in cache:
        return cache[key]

    from ultralytics import YOLO

    model = YOLO(checkpoint)
    kwargs: dict[str, Any] = {
        "data": data_yaml,
        "device": device,
        "verbose": False,
        "plots": False,
        "split": split,
    }
    if imgsz is not None:
        kwargs["imgsz"] = imgsz

    metrics = model.val(**kwargs)
    value = extract_map(metrics, metric_name)
    cache[key] = value
    return value


def resolve_value(
    spec: dict[str, Any] | float | int,
    aliases: dict[str, str],
    *,
    metric_name: str,
    device: str | int,
    imgsz: int | None,
    split: str,
    cache: dict[tuple[str, str, str, str], float],
) -> dict[str, Any]:
    """
    解析一个指标项。

    一个指标项可以写成两种形式：
      value: 直接填写已经算好的 mAP，例如 {"value": 0.49}
      checkpoint + data: 让脚本自动运行 YOLO val，例如 {"checkpoint": "final", "data": ".../data.yaml"}
    """
    if isinstance(spec, (int, float)):
        return {"value": float(spec), "source": "literal"}

    if "value" in spec and spec["value"] is not None:
        return {"value": float(spec["value"]), "source": "literal"}

    checkpoint = spec.get("checkpoint")
    data_yaml = spec.get("data")
    if not checkpoint or not data_yaml:
        raise ValueError(f"指标项必须包含 value，或者同时包含 checkpoint 和 data：{spec}")

    resolved_checkpoint = resolve_checkpoint(checkpoint, aliases)
    value = evaluate_checkpoint(
        resolved_checkpoint,
        data_yaml,
        metric_name=metric_name,
        device=device,
        imgsz=imgsz,
        split=spec.get("split", split),
        cache=cache,
    )
    return {
        "value": value,
        "checkpoint": resolved_checkpoint,
        "data": str(data_yaml),
        "source": "val",
    }


def compute_ratio(numerator: float, denominator: float, *, clamp_to_one: bool) -> float:
    """计算论文中的比例指标。默认不截断到 1，因为论文公式本身没有裁剪。"""
    if denominator <= 1e-12:
        return 0.0
    ratio = numerator / denominator
    return min(ratio, 1.0) if clamp_to_one else ratio


def evaluate_section(
    items: list[dict[str, Any]],
    aliases: dict[str, str],
    *,
    metric_name: str,
    device: str | int,
    imgsz: int | None,
    split: str,
    clamp_to_one: bool,
    cache: dict[tuple[str, str, str, str], float],
) -> tuple[list[float], list[dict[str, Any]]]:
    """评估一个指标分组，例如 retention 或 generalization。"""
    ratios: list[float] = []
    details: list[dict[str, Any]] = []

    for item in items:
        name = str(item.get("name", f"item_{len(details) + 1}"))
        numerator = resolve_value(
            item["numerator"],
            aliases,
            metric_name=metric_name,
            device=device,
            imgsz=imgsz,
            split=split,
            cache=cache,
        )
        denominator = resolve_value(
            item["denominator"],
            aliases,
            metric_name=metric_name,
            device=device,
            imgsz=imgsz,
            split=split,
            cache=cache,
        )
        ratio = compute_ratio(
            numerator["value"],
            denominator["value"],
            clamp_to_one=bool(item.get("clamp_to_one", clamp_to_one)),
        )
        ratios.append(ratio)
        details.append(
            {
                "name": name,
                "ratio": ratio,
                "ratio_percent": ratio * 100.0,
                "numerator": numerator,
                "denominator": denominator,
                "note": item.get("note", ""),
            }
        )

    return ratios, details


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 DuET 论文指标：Avg RI、Avg GI、RAI。")
    parser.add_argument("--plan", required=True, type=Path, help="指标评估计划文件，支持 YAML 或 JSON。")
    parser.add_argument("--output", type=Path, help="结果 JSON 保存路径；填写后会覆盖 plan.output。")
    parser.add_argument("--device", default=None, help="覆盖评估设备，例如 0 或 cpu。")
    args = parser.parse_args()

    args.plan = resolve_plan_path(args.plan)
    os.chdir(PROJECT_ROOT)

    plan = load_mapping(args.plan)
    manifest = None
    if plan.get("manifest"):
        manifest = load_mapping(plan["manifest"])

    aliases = build_checkpoint_aliases(manifest, plan)

    metric_name = str(plan.get("metric", "map50"))
    device = args.device if args.device is not None else plan.get("device", 0)
    imgsz = plan.get("imgsz")
    split = str(plan.get("split", "val"))
    clamp_to_one = bool(plan.get("clamp_to_one", False))
    cache: dict[tuple[str, str, str, str], float] = {}

    retention, retention_details = evaluate_section(
        plan.get("retention", []),
        aliases,
        metric_name=metric_name,
        device=device,
        imgsz=imgsz,
        split=split,
        clamp_to_one=clamp_to_one,
        cache=cache,
    )
    generalization, generalization_details = evaluate_section(
        plan.get("generalization", []),
        aliases,
        metric_name=metric_name,
        device=device,
        imgsz=imgsz,
        split=split,
        clamp_to_one=clamp_to_one,
        cache=cache,
    )

    avg_ri = sum(retention) / len(retention) if retention else 0.0
    avg_gi = sum(generalization) / len(generalization) if generalization else 0.0
    rai = (avg_ri + avg_gi) / 2.0

    result = {
        "metric": metric_name,
        "clamp_to_one": clamp_to_one,
        "avg_ri": avg_ri,
        "avg_gi": avg_gi,
        "rai": rai,
        "avg_ri_percent": avg_ri * 100.0,
        "avg_gi_percent": avg_gi * 100.0,
        "rai_percent": rai * 100.0,
        "retention": retention,
        "generalization": generalization,
        "retention_details": retention_details,
        "generalization_details": generalization_details,
        "checkpoint_aliases": aliases,
    }

    output = args.output or Path(plan.get("output", "paper_metrics_results.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    rai_payload = {"retention": retention, "generalization": generalization}
    (output.parent / "rai_metrics.json").write_text(
        json.dumps(rai_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n论文指标结果")
    print(f"Avg RI: {avg_ri:.4f} ({avg_ri * 100.0:.2f}%)")
    print(f"Avg GI: {avg_gi:.4f} ({avg_gi * 100.0:.2f}%)")
    print(f"RAI:    {rai:.4f} ({rai * 100.0:.2f}%)")
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()