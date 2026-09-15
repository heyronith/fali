#!/usr/bin/env python3
"""EF-4 GPT-2 gauge sensitivity (row-mean) — confirmatory candidates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_gpt2_runner import RunConfig, resolve_device, run_extended_fali_seed  # noqa: E402

EF4_SEEDS = {42, 1, 8, 22}
N_EXPECTED = 900
NUM_PER_STRATUM = 30
GAUGE_HEADS = ("L10H2", "L10H10")
EF4_PREREG = "2661459f047f65fabc418993d00e3bb06cc0b3b6"
GAUGE_PREREG_DEFAULT = "e60f81acfff8fc9dd4f924ebb1c087ec89734c05"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--score-gauge", choices=("native", "row_mean"), default="row_mean")
    p.add_argument("--num-samples", type=int, default=N_EXPECTED)
    p.add_argument("--num-per-stratum", type=int, default=NUM_PER_STRATUM)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--gauge-preregistration-sha", default=GAUGE_PREREG_DEFAULT)
    p.add_argument("--save-score-artifacts", action="store_true", default=True)
    p.add_argument("--no-save-score-artifacts", action="store_true")
    args = p.parse_args()

    if args.seed not in EF4_SEEDS:
        raise SystemExit(f"EF-4 gauge allows seeds {sorted(EF4_SEEDS)} only")
    if args.num_samples != N_EXPECTED:
        raise SystemExit(f"EF-4 gauge requires N={N_EXPECTED}")

    out = (
        Path(args.output_dir)
        if args.output_dir
        else _REPO / f"results/extended_fali/gauge_sensitivity/ef4_rowmean/seed_{args.seed}"
    )
    if not out.is_absolute():
        out = _REPO / out
    rel = str(out.resolve().relative_to(_REPO.resolve())).replace("\\", "/")
    if "results/extended_fali/gauge_sensitivity" not in rel:
        raise ValueError(f"Refusing output path outside gauge_sensitivity: {rel}")
    # Never overwrite native EF-4
    if "ef4_canonical" in rel:
        raise ValueError("Refusing to write into ef4_canonical")

    device = resolve_device(args.device)
    if args.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"--require-cuda but device={device}")

    ref_meta = _REPO / f"results/revision_phase4/full/seed_{args.seed}/dataset_metadata.csv"
    hist_factorial = _REPO / f"results/revision_phase4/full/seed_{args.seed}/factorial_raw.csv"
    if not ref_meta.exists():
        raise FileNotFoundError(ref_meta)

    save_arts = args.save_score_artifacts and not args.no_save_score_artifacts
    cfg = RunConfig(
        phase="EF-4-gauge",
        num_samples=args.num_samples,
        seed=args.seed,
        out_dir=out,
        reference_meta=ref_meta,
        historical_factorial=hist_factorial if args.score_gauge == "native" else None,
        required_hash_paths=[f"results/revision_phase4/full/seed_{args.seed}/factorial_raw.csv"]
        if hist_factorial.exists()
        else [],
        device=device,
        run_smoke_extras=False,
        compress_cells=True,
        dataset_kind="canonical",
        num_per_stratum=args.num_per_stratum,
        head_names=GAUGE_HEADS,
        preregistration_sha=args.gauge_preregistration_sha,
        score_gauge=args.score_gauge,
        save_score_artifacts=save_arts,
        skip_historical_bridge=True,
    )
    report = run_extended_fali_seed(cfg, _REPO)

    # Copy model revision stub + execution manifest
    gpt2_rev = {
        "repo_id": "gpt2",
        "note": "HookedTransformer.from_pretrained('gpt2') default revision as used by EF-4",
        "transformer_lens_target": "3.2.1",
        "torch_target": "2.6.0+cu124",
    }
    (out / "model_revision.json").write_text(json.dumps(gpt2_rev, indent=2) + "\n")

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
    ):
        fp = out / name
        if fp.exists():
            hashes[name] = {"sha256": file_sha256(fp), "bytes": fp.stat().st_size}

    manifest = {
        "experiment": "gauge_sensitivity_ef4",
        "score_gauge": args.score_gauge,
        "seed": args.seed,
        "heads": list(GAUGE_HEADS),
        "gauge_preregistration_sha": args.gauge_preregistration_sha,
        "ef4_preregistration_sha": EF4_PREREG,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": report.get("overall_pass"),
        "artifact_hashes": hashes,
    }
    (out / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Ensure dataset metadata present (runner writes via alignment path)
    if not (out / "dataset_metadata.csv").exists() and ref_meta.exists():
        shutil.copy2(ref_meta, out / "dataset_metadata.csv")

    return 0 if report.get("overall_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
