from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from duet_repro.utils.paths import project_root, resolve_repo_path


SCENARIO_NAMES = ("status1", "status3")
TRAIN_VARIANTS = ("default",)


@dataclass(frozen=True)
class Scenario:
    """一个复现场景的路径注册信息。

    status1/status3 的训练、评估、配置目录不完全相同，统一入口通过
    Scenario 对象把这些路径集中管理，避免每个 tools 脚本重复拼路径。
    """
    name: str
    legacy_root: Path
    config_dir: Path
    legacy_config_dir: Path
    train_script: Path
    eval_script: Path
    ablation_script: Path | None = None


def get_scenario(name: str) -> Scenario:
    """根据场景名返回完整路径配置。"""
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario {name!r}. Expected one of: {', '.join(SCENARIO_NAMES)}")

    root = project_root()
    legacy_root = root / "legacy" / name
    scenario_root = root / name
    return Scenario(
        name=name,
        legacy_root=legacy_root,
        config_dir=root / "experiments" / name,
        legacy_config_dir=scenario_root / "configs",
        train_script=legacy_root / "train_duet.py",
        eval_script=legacy_root / "eval_paper_metrics.py",
        ablation_script=(legacy_root / "run_ablation.py") if name == "status1" else None,
    )


def infer_scenario_from_path(path: str | Path) -> str | None:
    """从路径片段中推断 status1/status3。

    例如 experiments/status1/eval.yaml 可以推断出 status1；如果路径里不含
    场景名，则返回 None，让调用方要求用户显式传 --scenario。
    """
    resolved = resolve_repo_path(path)
    for part in resolved.parts:
        if part in SCENARIO_NAMES:
            return part
    return None


def resolve_scenario(name: str | None = None, *, config_or_plan: str | Path | None = None) -> Scenario:
    """优先使用显式场景名，否则尝试从配置/计划文件路径推断场景。"""
    scenario_name = name or (infer_scenario_from_path(config_or_plan) if config_or_plan else None)
    if scenario_name is None:
        raise ValueError("Please pass --scenario when the config/plan path does not include status1/status3.")
    return get_scenario(scenario_name)


def resolve_config_path(scenario: Scenario, config: str | Path) -> Path:
    """解析训练配置路径。"""
    return _resolve_scenario_file(scenario, config)


def resolve_plan_path(scenario: Scenario, plan: str | Path) -> Path:
    """解析评估计划路径。"""
    return _resolve_scenario_file(scenario, plan)


def resolve_train_script(scenario: Scenario, variant: str = "default") -> Path:
    """解析训练脚本变体；当前只保留 default 变体。"""
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"Unknown train variant {variant!r}. Expected one of: {', '.join(TRAIN_VARIANTS)}")
    return scenario.train_script


def resolve_prepare_plan(scenario_name: str) -> Path:
    """返回数据准备脚本默认使用的 plan 路径。"""
    if scenario_name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario {scenario_name!r}. Expected one of: {', '.join(SCENARIO_NAMES)}")
    return project_root() / "data_process" / "configs" / f"{scenario_name}.yaml"


def _resolve_scenario_file(scenario: Scenario, value: str | Path) -> Path:
    """按统一优先级查找配置/计划文件。

    查找顺序是：
    1. 用户传入的绝对路径；
    2. 仓库根目录下的相对路径；
    3. experiments/status*/ 下的标准配置；
    4. 旧版 status*/configs/ 目录；
    5. 没有后缀时自动补 .yaml。
    """
    raw = Path(value)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        root = project_root()
        candidates.append(root / raw)
        candidates.append(scenario.config_dir / raw)
        candidates.append(scenario.legacy_config_dir / raw)
        if raw.suffix == "":
            candidates.append(scenario.config_dir / f"{raw.name}.yaml")
            candidates.append(scenario.legacy_config_dir / f"{raw.name}.yaml")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve config/plan {value!r}. Tried:\n  - {tried}")
