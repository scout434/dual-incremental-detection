from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

from duet_repro.experiments.registry import Scenario
from duet_repro.utils.paths import ensure_project_root_on_path


def run_train(
    scenario: Scenario,
    config_path: Path,
    *,
    script_path: Path | None = None,
    start_task: int | None = None,
    previous_checkpoint: str | None = None,
    output_dir: str | None = None,
) -> None:
    ensure_project_root_on_path()
    script_path = script_path or scenario.train_script
    module = _load_module(script_path, f"duet_repro_legacy_{scenario.name}_{script_path.stem}")
    module.main(
        str(config_path),
        start_task=start_task,
        previous_checkpoint=previous_checkpoint,
        output_dir_override=output_dir,
    )


def run_eval(scenario: Scenario, plan_path: Path, *, extra_args: list[str] | None = None) -> None:
    argv = [str(scenario.eval_script), "--plan", str(plan_path), *(extra_args or [])]
    _run_script_as_main(scenario.eval_script, argv)


def run_prepare_data(plan_path: Path, *, extra_args: list[str] | None = None) -> None:
    root = ensure_project_root_on_path()
    script = root / "data_process" / "prepare_data.py"
    argv = [str(script), "--plan", str(plan_path), *(extra_args or [])]
    _run_script_as_main(script, argv)


def run_ablation(scenario: Scenario, *, extra_args: list[str]) -> None:
    if scenario.ablation_script is None:
        raise ValueError("Ablations are currently defined only for status1.")
    _run_script_as_main(scenario.ablation_script, [str(scenario.ablation_script), *extra_args])


def _load_module(script_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_script_as_main(script_path: Path, argv: list[str]) -> None:
    ensure_project_root_on_path()
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv

