"""Validate committed inputs before figure rendering."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_data import validate_scientific_inputs  # noqa: E402


def main() -> int:
    summary = validate_scientific_inputs()
    print("VALIDATION PASS")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
