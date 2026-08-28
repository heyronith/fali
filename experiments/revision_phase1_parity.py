"""
Compare CPU smoke vs Modal CUDA smoke per-example quantities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IOS_COLS = [
    "A",
    "B_old",
    "B",
    "D_h",
    "C_h",
    "intact_effect",
    "conditional_effect",
    "conditional_gain",
]
ATTN_COLS = [
    "clean_attn",
    "historical_primary_attn",
    "corrected_primary_attn",
    "historical_attention_spike",
    "corrected_attention_spike",
]
KEY = ["example_idx", "head_name"]


def compare(cpu_raw: Path, cuda_raw: Path, out_dir: Path) -> dict:
    cpu = pd.read_csv(cpu_raw)
    cuda = pd.read_csv(cuda_raw)
    merged = cpu.merge(cuda, on=KEY, suffixes=("_cpu", "_cuda"), how="inner")
    assert len(merged) == len(cpu) == len(cuda), (
        f"Row mismatch: cpu={len(cpu)} cuda={len(cuda)} merged={len(merged)}"
    )

    rows = []
    for col in IOS_COLS + ATTN_COLS:
        abs_err = (merged[f"{col}_cuda"] - merged[f"{col}_cpu"]).abs()
        denom = merged[f"{col}_cpu"].abs().clip(lower=1e-8)
        rel_err = abs_err / denom
        rows.append(
            {
                "quantity": col,
                "family": "ios" if col in IOS_COLS else "attention",
                "n": int(len(abs_err)),
                "max_abs_error": float(abs_err.max()),
                "mean_abs_error": float(abs_err.mean()),
                "max_rel_error": float(rel_err.max()),
                "mean_rel_error": float(rel_err.mean()),
            }
        )
    parity = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    parity.to_csv(out_dir / "cuda_cpu_parity.csv", index=False)

    ios = parity[parity["family"] == "ios"]
    attn = parity[parity["family"] == "attention"]
    max_ios = float(ios["max_abs_error"].max())
    max_attn = float(attn["max_abs_error"].max())
    pass_ios = max_ios <= 1e-3
    pass_attn = max_attn <= 1e-4
    passed = pass_ios and pass_attn

    report = [
        "# CUDA vs CPU parity report",
        "",
        f"- CPU raw: `{cpu_raw}`",
        f"- CUDA raw: `{cuda_raw}`",
        f"- Compared rows: {len(merged)} (12 candidates × N examples)",
        "",
        "## Acceptance thresholds",
        "",
        "- IO−S quantities: max abs error ≤ 1e-3",
        "- Attention quantities: max abs error ≤ 1e-4",
        "",
        "## Results",
        "",
        f"- max abs error (IO−S): **{max_ios:.6e}** → {'PASS' if pass_ios else 'FAIL'}",
        f"- max abs error (attention): **{max_attn:.6e}** → {'PASS' if pass_attn else 'FAIL'}",
        f"- overall: **{'PASS' if passed else 'FAIL'}**",
        "",
        "## Per-quantity",
        "",
        "```",
        parity.to_string(index=False),
        "```",
        "",
    ]
    (out_dir / "cuda_cpu_parity_report.md").write_text("\n".join(report) + "\n")
    return {
        "passed": passed,
        "max_ios": max_ios,
        "max_attn": max_attn,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cpu-dir",
        default="results/revision_phase1/smoke",
    )
    p.add_argument(
        "--cuda-dir",
        default="results/revision_phase1/modal_smoke_seed42",
    )
    p.add_argument(
        "--out-dir",
        default="results/revision_phase1/modal_smoke_seed42",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    res = compare(
        root / args.cpu_dir / "candidate_raw.csv",
        root / args.cuda_dir / "candidate_raw.csv",
        root / args.out_dir,
    )
    print(res)
    if not res["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
