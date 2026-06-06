from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_project_root_on_path() -> Path:
    root = project_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate

