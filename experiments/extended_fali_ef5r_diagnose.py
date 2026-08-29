#!/usr/bin/env python3
"""EF-5R zero-reference diagnosis (analysis only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef5r_diagnosis import run_diagnosis  # noqa: E402


def main() -> int:
    out = _REPO / "results/extended_fali/ef5_reference_diagnosis"
    results = run_diagnosis(_REPO, out)
    print(json.dumps({"diagnosis_category": results["diagnosis"]["diagnosis_category"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
