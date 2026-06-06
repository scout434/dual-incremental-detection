from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from duet_repro.utils.paths import project_root, resolve_repo_path


SCENARIO_NAMES = ("status1", "status3")
TRAIN_VARIANTS = ("default",)


@dataclass(frozen=True)
class Scenario:
    name: str
    legacy_root: Path
    config_dir: Path
    legacy_config_dir: Path
    train_script: Path
    eval_script: Path
    ablation_script: Path | None = None


def get_scenario(name: str) -> Scenario:
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
    resolved = resolve_repo_path(path)
    for part in resolved.parts:
        if part in SCENARIO_NAMES:
            return part
    return None


def resolve_scenario(name: str | None = None, *, config_or_plan: str | Path | None = None) -> Scenario:
    scenario_name = name or (infer_scenario_from_path(config_or_plan) if config_or_plan else None)
    if scenario_name is None:
        raise ValueError("Please pass --scenario when the config/plan path does not include status1/status3.")
    return get_scenario(scenario_name)


def resolve_config_path(scenario: Scenario, config: str | Path) -> Path:
    return _resolve_scenario_file(scenario, config)


def resolve_plan_path(scenario: Scenario, plan: str | Path) -> Path:
    return _resolve_scenario_file(scenario, plan)


def resolve_train_script(scenario: Scenario, variant: str = "default") -> Path:
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"Unknown train variant {variant!r}. Expected one of: {', '.join(TRAIN_VARIANTS)}")
    return scenario.train_script


def resolve_prepare_plan(scenario_name: str) -> Path:
    if scenario_name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario {scenario_name!r}. Expected one of: {', '.join(SCENARIO_NAMES)}")
    return project_root() / "data_process" / "configs" / f"{scenario_name}.yaml"


def _resolve_scenario_file(scenario: Scenario, value: str | Path) -> Path:
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
