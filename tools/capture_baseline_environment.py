#!/usr/bin/env python3
"""Capture baseline Python environment for reproducibility audits."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def capture_environment() -> dict:
    import torch

    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformer_lens": pkg_version("transformer-lens"),
        "transformers": pkg_version("transformers"),
        "numpy": pkg_version("numpy"),
        "pandas": pkg_version("pandas"),
        "scipy": pkg_version("scipy"),
        "einops": pkg_version("einops"),
        "pytest": pkg_version("pytest"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
    return env


def pip_freeze() -> str:
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def main() -> None:
    env = capture_environment()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "TEST_BASELINE_ENVIRONMENT.json").write_text(
        json.dumps(env, indent=2) + "\n"
    )
    (RESULTS / "TEST_BASELINE_PIP_FREEZE.txt").write_text(pip_freeze())
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
