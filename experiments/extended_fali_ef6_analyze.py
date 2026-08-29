#!/usr/bin/env python3
"""EF-6 aggregate analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef6_analysis import run_ef6_aggregate  # noqa: E402


def main() -> int:
    agg = _REPO / "results/extended_fali/ef6_pythia_induction/aggregate"
    res = run_ef6_aggregate(_REPO, agg)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("EF6_GATE") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
