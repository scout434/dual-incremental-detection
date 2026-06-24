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
    """加载并执行某个场景的训练脚本。

    legacy 训练脚本不是标准 Python 包模块，因此这里用文件路径动态加载。
    这样可以在不大改旧代码的前提下，给 status1/status3 提供统一入口。
    """
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
    """以命令行脚本方式执行评估。

    评估脚本内部依赖 argparse 读取 sys.argv，因此这里不直接 import main，
    而是临时构造 argv 后用 runpy 模拟 `python eval_script.py ...`。
    """
    argv = [str(scenario.eval_script), "--plan", str(plan_path), *(extra_args or [])]
    _run_script_as_main(scenario.eval_script, argv)


def run_prepare_data(plan_path: Path, *, extra_args: list[str] | None = None) -> None:
    """调用统一数据准备脚本，根据 plan 生成 status1/status3 数据切片。"""
    root = ensure_project_root_on_path()
    script = root / "data_process" / "prepare_data.py"
    argv = [str(script), "--plan", str(plan_path), *(extra_args or [])]
    _run_script_as_main(script, argv)


def run_ablation(scenario: Scenario, *, extra_args: list[str]) -> None:
    """调用场景对应的消融脚本；当前项目只为 status1 定义了消融。"""
    if scenario.ablation_script is None:
        raise ValueError("Ablations are currently defined only for status1.")
    _run_script_as_main(scenario.ablation_script, [str(scenario.ablation_script), *extra_args])


def _load_module(script_path: Path, module_name: str) -> ModuleType:
    """按文件路径加载 legacy 脚本，并返回可调用的模块对象。"""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_script_as_main(script_path: Path, argv: list[str]) -> None:
    """临时替换 sys.argv，把普通脚本当作 `__main__` 执行。

    finally 中恢复 argv 是为了避免一次工具调用结束后污染后续训练/评估命令。
    """
    ensure_project_root_on_path()
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv

