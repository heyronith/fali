"""Parse and validate committed figure source data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from figure_style import REPO_ROOT, format_sci

EF1_PATH = REPO_ROOT / "results/extended_fali/ef1_synthetic/validation_report.json"
EF4_CI_PATH = REPO_ROOT / "results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv"
EF4_EXACT_PATH = REPO_ROOT / "results/extended_fali/ef4_canonical/aggregate/exact_means.csv"
EF4_GATE_PATH = REPO_ROOT / "results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json"
EF6_EXACT_PATH = REPO_ROOT / "results/extended_fali/ef6_pythia_induction/aggregate/exact_means.csv"
EF6_BOOT_PATH = REPO_ROOT / "results/extended_fali/ef6_pythia_induction/aggregate/bootstrap_ci.csv"
EF6_GATE_PATH = REPO_ROOT / "results/extended_fali/ef6_pythia_induction/aggregate/ef6_gate.json"
EF6_HYP_PATH = REPO_ROOT / "results/extended_fali/ef6_pythia_induction/aggregate/hypothesis_results.json"

REQUIRED_PATHS = [
    EF1_PATH,
    EF4_CI_PATH,
    EF4_EXACT_PATH,
    EF4_GATE_PATH,
    EF6_EXACT_PATH,
    EF6_BOOT_PATH,
    EF6_GATE_PATH,
    EF6_HYP_PATH,
]

CHANNEL_MAP = {
    "T": ("T", "intended", ["other_R", "other_O", "other_V", "other_C"]),
    "R": ("R", "intended", ["other_T", "other_O", "other_V", "other_C"]),
    "O": ("O", "intended", ["other_T", "other_R", "other_V", "other_C"]),
    "V": ("V", "intended", ["other_T", "other_R", "other_O", "other_C"]),
    "C": ("C", "intended", ["other_T", "other_R", "other_O", "other_V"]),
}
PHI_KEYS = ["phi_T", "phi_R", "phi_O", "phi_V", "phi_C"]
OTHER_TO_PHI = {
    "other_T": "phi_T",
    "other_R": "phi_R",
    "other_O": "phi_O",
    "other_V": "phi_V",
    "other_C": "phi_C",
}


def assert_paths_exist() -> None:
    missing = [p for p in REQUIRED_PATHS if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required figure source files:\n" + "\n".join(str(p) for p in missing))


def load_ef1() -> Dict[str, Any]:
    return json.loads(EF1_PATH.read_text())


def load_ef4_ci() -> pd.DataFrame:
    df = pd.read_csv(EF4_CI_PATH)
    dup = df.duplicated(subset=["head_name", "metric"], keep=False)
    if dup.any():
        raise ValueError("Duplicate rows in EF-4 hierarchical_bootstrap_ci.csv")
    return df


def load_ef4_gate() -> Dict[str, Any]:
    return json.loads(EF4_GATE_PATH.read_text())


def load_ef6_exact() -> pd.DataFrame:
    return pd.read_csv(EF6_EXACT_PATH)


def load_ef6_boot() -> pd.DataFrame:
    df = pd.read_csv(EF6_BOOT_PATH)
    dup = df.duplicated(subset=["metric"], keep=False)
    if dup.any():
        raise ValueError("Duplicate rows in EF-6 bootstrap_ci.csv")
    return df


def load_ef6_gate() -> Dict[str, Any]:
    return json.loads(EF6_GATE_PATH.read_text())


def load_ef6_hypothesis() -> Dict[str, Any]:
    return json.loads(EF6_HYP_PATH.read_text())


def get_ef4_metric(df: pd.DataFrame, head: str, metric: str) -> Dict[str, float]:
    rows = df[(df["head_name"] == head) & (df["metric"] == metric)]
    if len(rows) != 1:
        raise ValueError(f"Expected one EF-4 row for {head}/{metric}, got {len(rows)}")
    row = rows.iloc[0]
    point = float(row["exact_mean"])
    lo = float(row["ci_low"])
    hi = float(row["ci_high"])
    if not (lo <= point <= hi):
        raise ValueError(f"CI ordering failed for {head}/{metric}: {lo} <= {point} <= {hi}")
    return {"point": point, "ci_low": lo, "ci_high": hi}


def get_ef6_metric(exact: pd.DataFrame, boot: pd.DataFrame, metric: str) -> Dict[str, float]:
    erows = exact[exact["metric"] == metric]
    brows = boot[boot["metric"] == metric]
    if len(erows) != 1 or len(brows) != 1:
        raise ValueError(f"Expected one EF-6 row for {metric}")
    point = float(erows.iloc[0]["exact_balanced_mean"])
    lo = float(brows.iloc[0]["ci_low"])
    hi = float(brows.iloc[0]["ci_high"])
    if not (lo <= point <= hi):
        raise ValueError(f"CI ordering failed for EF-6 {metric}: {lo} <= {point} <= {hi}")
    return {"point": point, "ci_low": lo, "ci_high": hi}


def single_channel_matrix(ef1: Dict[str, Any]) -> Tuple[List[str], List[List[float]], List[float]]:
    rows: List[str] = []
    matrix: List[List[float]] = []
    delta_us: List[float] = []
    for ch in ["T", "R", "O", "V", "C"]:
        entry = ef1["level_a"]["single_channel"][ch]
        delta_u = float(entry["delta_u"])
        scale = abs(delta_u)
        if scale < 1e-15:
            raise ValueError(f"Zero delta_u for channel {ch}")
        phi = {k: 0.0 for k in PHI_KEYS}
        phi[f"phi_{ch}"] = float(entry["intended"])
        for other_key, val in entry.items():
            if other_key.startswith("other_"):
                phi[OTHER_TO_PHI[other_key]] = float(val)
        norm = [phi[k] / scale for k in PHI_KEYS]
        rows.append(f"{ch}-only")
        matrix.append(norm)
        delta_us.append(delta_u)
    return rows, matrix, delta_us


def ef1_dashboard_rows(ef1: Dict[str, Any]) -> List[Tuple[str, float]]:
    lvl_a = ef1["level_a"]
    lvl_b = ef1["level_b"]
    recon = lvl_b["reconstruction"]
    ep = lvl_b["independent_endpoints"]
    act = lvl_b["endpoint_activation"]
    inj = lvl_b["injection_scope"]
    leak = lvl_b["live_leak"]
    part = lvl_b["batch_partition"]
    return [
        ("score → pattern A/B", max(float(recon["pattern_a"]), float(recon["pattern_b"]))),
        ("pattern/value → z A/B", max(float(recon["z_a"]), float(recon["z_b"]))),
        ("masked probability", float(recon["masked_prob_max"])),
        ("independent endpoint identity", max(float(ep["u00000_vs_intact"]), float(ep["u11111_vs_conditional"]), float(ep["gain_vs_delta_u"]))),
        ("all-A injected-z identity", float(act["all_a_z_error"])),
        ("all-B injected-z identity", float(act["all_b_z_error"])),
        ("non-candidate modification", float(inj["noncandidate_z_error"])),
        ("live Q leak", float(leak["q_leak_error"])),
        ("live V leak", float(leak["v_leak_error"])),
        ("live Q+V leak", float(leak["qv_leak_error"])),
        ("model quotient bridge", float(lvl_b["quotient_bridge_max"])),
        ("synthetic quotient bridge", float(lvl_a["quotient_bridge_max"])),
        ("partition coverage max", float(max(part["coverage_error_per_example"]))),
        ("partition overlap max", float(max(part["overlap_count_per_example"]))),
    ]


def ef1_grouped_validation_rows(ef1: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Six grouped validation rows for readable Figure 3 panel (c)."""
    lvl_a = ef1["level_a"]
    lvl_b = ef1["level_b"]
    recon = lvl_b["reconstruction"]
    ep = lvl_b["independent_endpoints"]
    act = lvl_b["endpoint_activation"]
    inj = lvl_b["injection_scope"]
    leak = lvl_b["live_leak"]
    part = lvl_b["batch_partition"]

    score_pattern = max(float(recon["pattern_a"]), float(recon["pattern_b"]))
    pattern_z = max(float(recon["z_a"]), float(recon["z_b"]))
    coverage_max = float(max(part["coverage_error_per_example"]))
    overlap_max = float(max(part["overlap_count_per_example"]))
    mask_partition = max(float(recon["masked_prob_max"]), coverage_max, overlap_max)
    mask_status = "PASS" if coverage_max == 0.0 and overlap_max == 0.0 else format_sci(mask_partition)

    endpoints = max(
        float(ep["u00000_vs_intact"]),
        float(ep["u11111_vs_conditional"]),
        float(ep["gain_vs_delta_u"]),
    )
    isolation = max(
        float(inj["noncandidate_z_error"]),
        float(leak["q_leak_error"]),
        float(leak["v_leak_error"]),
        float(leak["qv_leak_error"]),
        float(act["all_a_z_error"]),
        float(act["all_b_z_error"]),
    )
    quotient = max(float(lvl_b["quotient_bridge_max"]), float(lvl_a["quotient_bridge_max"]))

    return [
        ("Score → pattern reconstruction", format_sci(score_pattern)),
        ("Pattern/value → z reconstruction", format_sci(pattern_z)),
        ("Mask + batch partition", mask_status),
        ("Independent endpoints", format_sci(endpoints)),
        ("Candidate isolation / live leakage", format_sci(isolation)),
        ("Coarse–fine quotient bridge", format_sci(quotient)),
    ]


def validate_scientific_inputs() -> Dict[str, Any]:
    assert_paths_exist()
    ef4 = load_ef4_ci()
    ef4_gate = load_ef4_gate()
    ef6_exact = load_ef6_exact()
    ef6_boot = load_ef6_boot()
    ef6_gate = load_ef6_gate()
    ef1 = load_ef1()

    if not ef1.get("overall_pass", False):
        raise AssertionError("EF-1 overall_pass is not true")
    if ef4_gate.get("EF4_FINE_SCORE_GATE") != "PASS":
        raise AssertionError("EF-4 fine-score gate is not PASS")
    if ef6_gate.get("EF6_GATE") != "FAIL":
        raise AssertionError("EF-6 gate is not FAIL")

    l10h2 = {
        "conditional_gain": get_ef4_metric(ef4, "L10H2", "conditional_gain"),
        "phi_T": get_ef4_metric(ef4, "L10H2", "phi_T"),
        "phi_R": get_ef4_metric(ef4, "L10H2", "phi_R"),
        "delta_target_attn": get_ef4_metric(ef4, "L10H2", "delta_target_attn"),
    }
    l10h10 = {
        "conditional_gain": get_ef4_metric(ef4, "L10H10", "conditional_gain"),
        "phi_T": get_ef4_metric(ef4, "L10H10", "phi_T"),
        "phi_R": get_ef4_metric(ef4, "L10H10", "phi_R"),
        "delta_target_attn": get_ef4_metric(ef4, "L10H10", "delta_target_attn"),
    }
    pythia = {
        "conditional_gain": get_ef6_metric(ef6_exact, ef6_boot, "conditional_gain"),
        "conditional_effect": get_ef6_metric(ef6_exact, ef6_boot, "conditional_effect"),
        "phi_T": get_ef6_metric(ef6_exact, ef6_boot, "phi_T"),
        "phi_R": get_ef6_metric(ef6_exact, ef6_boot, "phi_R"),
    }

    checks = [
        (l10h2["conditional_gain"]["point"] > 0, "GPT-2 L10H2 conditional_gain > 0"),
        (l10h2["phi_T"]["point"] > 0, "GPT-2 L10H2 phi_T > 0"),
        (l10h2["phi_R"]["point"] < 0, "GPT-2 L10H2 phi_R < 0"),
        (l10h10["delta_target_attn"]["point"] > 0, "GPT-2 L10H10 delta_target_attn > 0"),
        (l10h10["conditional_gain"]["point"] < 0, "GPT-2 L10H10 conditional_gain < 0"),
        (l10h10["phi_T"]["point"] > 0, "GPT-2 L10H10 phi_T > 0"),
        (l10h10["phi_R"]["point"] < 0, "GPT-2 L10H10 phi_R < 0"),
        (pythia["conditional_effect"]["point"] > 0, "Pythia conditional_effect > 0"),
        (pythia["conditional_gain"]["point"] > 0, "Pythia conditional_gain > 0"),
        (pythia["phi_T"]["point"] < 0, "Pythia phi_T < 0"),
        (pythia["phi_R"]["point"] > 0, "Pythia phi_R > 0"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        raise AssertionError("Scientific validation failed:\n" + "\n".join(failed))

    return {
        "ef1_overall_pass": ef1["overall_pass"],
        "ef4_gate": ef4_gate["EF4_FINE_SCORE_GATE"],
        "ef6_gate": ef6_gate["EF6_GATE"],
        "l10h2": l10h2,
        "l10h10": l10h10,
        "pythia": pythia,
        "ef6_gates": ef6_gate["gates"],
    }
