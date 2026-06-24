from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """返回仓库根目录。

    当前文件位于 duet_repro/utils/paths.py，因此向上两级就是项目根目录。
    """
    return Path(__file__).resolve().parents[2]


def ensure_project_root_on_path() -> Path:
    """确保项目根目录在 sys.path 中，便于 legacy 脚本用绝对包名导入。"""
    root = project_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def resolve_repo_path(path: str | Path) -> Path:
    """把相对路径解释为相对于项目根目录的路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate

