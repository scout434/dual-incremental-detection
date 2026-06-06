from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duet_repro.core.task_vectors import load_state_dict
from duet_repro.utils.paths import resolve_repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print lightweight checkpoint state_dict metadata.")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = resolve_repo_path(args.checkpoint)
    state = load_state_dict(checkpoint)
    print(f"checkpoint: {checkpoint}")
    print(f"state_dict_tensors: {len(state)}")
    for key in list(state)[:12]:
        print(f"{key}: {tuple(state[key].shape)}")


if __name__ == "__main__":
    main()
