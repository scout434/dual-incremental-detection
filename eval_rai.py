from __future__ import annotations

import argparse
import json
from pathlib import Path

from duet_repro.core.metrics import rai_from_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = rai_from_indices(
        retention=[float(x) for x in payload.get("retention", [])],
        generalization=[float(x) for x in payload.get("generalization", [])],
    )
    print(f"Avg RI: {result.avg_ri:.4f}")
    print(f"Avg GI: {result.avg_gi:.4f}")
    print(f"RAI:    {result.rai:.4f}")


if __name__ == "__main__":
    main()

