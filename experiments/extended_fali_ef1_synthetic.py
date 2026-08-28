#!/usr/bin/env python3
"""EF-1 synthetic validation driver (Level A + Level B)."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from extended_fali_synthetic import (
    LEVEL_A_CASES,
    build_level_a_tc_interaction,
    endpoint_reconstruction_errors,
    single_channel_owen_check,
)
from extended_fali_synthetic_transformer import (
    EF1_ARCH,
    EF1_SEED,
    build_level_b_context,
    endpoint_errors,
    no_live_leak_error,
    quotient_bridge_error,
    reconstruction_errors,
)

OUT = REPO / "results" / "extended_fali" / "ef1_synthetic" / "validation_report.json"
ATOL_A = 1e-12
FP32_TOL = 2e-6
ENDPOINT_TOL = 1e-5
LEAK_TOL = 1e-6
QUOTIENT_B_TOL = 1e-5


def _pass(label: str, ok: bool, detail) -> dict:
    return {"criterion": label, "pass": bool(ok), "detail": detail}


def run_level_a() -> dict:
    single = {}
    ok_single = True
    for name, (builder, intended) in LEVEL_A_CASES.items():
        res = single_channel_owen_check(builder(), intended)
        ok = res["intended_err"] < ATOL_A and all(
            res[k] < ATOL_A for k in res if k.startswith("other_")
        )
        ok_single &= ok
        single[name] = res

    tc = build_level_a_tc_interaction(2.0)
    table = tc.cell_table()
    owen = tc.owen()
    delta = table[(1, 0, 0, 0, 1)] - table[(0, 0, 0, 0, 0)]
    tc_ok = (
        abs(owen["T"] - delta / 2) < ATOL_A
        and abs(owen["C"] - delta / 2) < ATOL_A
        and all(abs(owen[k]) < ATOL_A for k in ("R", "O", "V"))
    )

    endpoint_sys = LEVEL_A_CASES["T"][0]()
    ep = endpoint_reconstruction_errors(endpoint_sys)
    ep_ok = all(v < ATOL_A for v in ep.values())

    bridge = tc.quotient_bridge_error()
    bridge_ok = bridge < ATOL_A

    return {
        "single_channel": single,
        "tc_interaction": {"delta_u": delta, "owen": owen, "pass": tc_ok},
        "endpoint_reconstruction_max": max(ep.values()),
        "quotient_bridge_max": bridge,
        "checks": [
            _pass("level_a_single_channel", ok_single, single),
            _pass("level_a_tc_interaction", tc_ok, owen),
            _pass("level_a_endpoint_reconstruction", ep_ok, ep),
            _pass("level_a_quotient_bridge", bridge_ok, bridge),
        ],
    }


def run_level_b() -> dict:
    ctx = build_level_b_context(seed=EF1_SEED)
    recon = reconstruction_errors(ctx)
    endpoints = endpoint_errors(ctx)
    leak = no_live_leak_error(ctx)
    quotient = quotient_bridge_error(ctx)

    recon_ok = (
        recon["pattern_a"] <= FP32_TOL
        and recon["pattern_b"] <= FP32_TOL
        and recon["z_a"] <= FP32_TOL
        and recon["z_b"] <= FP32_TOL
        and recon["masked_prob_max"] <= 1e-8
    )
    endpoint_ok = all(v <= ENDPOINT_TOL for v in endpoints.values())
    leak_ok = leak <= LEAK_TOL
    quotient_ok = quotient <= QUOTIENT_B_TOL

    return {
        "architecture": EF1_ARCH,
        "seed": EF1_SEED,
        "reconstruction": recon,
        "endpoints": endpoints,
        "no_live_leak": leak,
        "quotient_bridge_max": quotient,
        "checks": [
            _pass("level_b_score_pattern_z", recon_ok, recon),
            _pass("level_b_endpoints", endpoint_ok, endpoints),
            _pass("level_b_no_live_leak", leak_ok, leak),
            _pass("level_b_quotient_bridge", quotient_ok, quotient),
        ],
    }


def main() -> int:
    level_a = run_level_a()
    level_b = run_level_b()
    checks = level_a["checks"] + level_b["checks"]
    overall = all(c["pass"] for c in checks)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": {"ef1_seed": EF1_SEED},
        "environment": {
            "python": sys.version.replace("\n", " "),
            "torch": torch.__version__,
            "transformer_lens": importlib.metadata.version("transformer-lens"),
            "device": "cpu",
            "dtype_level_b": "float32",
        },
        "level_a": level_a,
        "level_b": level_b,
        "overall_pass": overall,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
