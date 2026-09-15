#!/usr/bin/env python3
"""EF-6 Pythia confirmation gauge sensitivity (row-mean)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef6_runner import EF6RunConfig, PREREG_SHA, resolve_device, run_ef6_confirmation  # noqa: E402

GAUGE_PREREG_DEFAULT = "e60f81acfff8fc9dd4f924ebb1c087ec89734c05"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, choices=[1, 8, 22])
    p.add_argument("--score-gauge", choices=("native", "row_mean"), default="row_mean")
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--selection-lock", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--gauge-preregistration-sha", default=GAUGE_PREREG_DEFAULT)
    p.add_argument("--save-score-artifacts", action="store_true", default=True)
    p.add_argument("--no-save-score-artifacts", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    if args.require_cuda and device != "cuda":
        raise SystemExit("CUDA required")

    out = args.output_dir or (
        _REPO / f"results/extended_fali/gauge_sensitivity/ef6_rowmean/seed_{args.seed}"
    )
    if not out.is_absolute():
        out = _REPO / out
    rel = str(out.resolve().relative_to(_REPO.resolve())).replace("\\", "/")
    if "results/extended_fali/gauge_sensitivity" not in rel:
        raise ValueError(f"Refusing output path: {rel}")
    if "ef6_pythia_induction" in rel:
        raise ValueError("Refusing to write into native ef6_pythia_induction")

    save_arts = args.save_score_artifacts and not args.no_save_score_artifacts
    val = run_ef6_confirmation(
        EF6RunConfig(
            phase="confirmation-gauge",
            seed=args.seed,
            out_dir=out,
            device=device,
            preregistration_sha=PREREG_SHA,
            repo_root=_REPO,
            selection_lock_path=args.selection_lock,
            score_gauge=args.score_gauge,
            save_score_artifacts=save_arts,
            gauge_preregistration_sha=args.gauge_preregistration_sha,
        )
    )

    hashes = {}
    for name in (
        "attribution_raw.csv",
        "cell_raw.csv.gz",
        "score_gauge_artifacts.pt",
        "dataset_metadata.csv",
        "invariant_report.json",
        "run_config.json",
        "environment.json",
        "preregistration_sha.txt",
        "model_revision.json",
    ):
        fp = out / name
        if fp.exists():
            hashes[name] = {"sha256": file_sha256(fp), "bytes": fp.stat().st_size}
    # Prefer root model revision copy
    root_rev = _REPO / "results/extended_fali/ef6_pythia_induction/model_revision.json"
    if root_rev.exists() and not (out / "model_revision.json").exists():
        (out / "model_revision.json").write_text(root_rev.read_text())
        hashes["model_revision.json"] = {"sha256": file_sha256(out / "model_revision.json"), "bytes": (out / "model_revision.json").stat().st_size}

    manifest = {
        "experiment": "gauge_sensitivity_ef6",
        "score_gauge": args.score_gauge,
        "seed": args.seed,
        "gauge_preregistration_sha": args.gauge_preregistration_sha,
        "ef6_preregistration_sha": PREREG_SHA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": val.get("overall_pass"),
        "artifact_hashes": hashes,
    }
    (out / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if val.get("overall_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
