"""
EF-6 Pythia induction transfer — statistics, pair selection, and confirmatory gates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from extended_fali_induction import CONFIRMATION_SEEDS, DISCOVERY_SEED, QUERY_OFFSETS
from revision_stats import stable_sha256_seed

N_BOOTSTRAP = 10_000
ATTN_THRESHOLD = 0.05
PHI_SHARE_THRESHOLD = 0.10
EF6_SCOPE = "EF6"


def head_name(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def parse_head_name(name: str) -> Tuple[int, int]:
    layer = int(name[1 : name.index("H")])
    h = int(name[name.index("H") + 1 :])
    return layer, h


def observed_offset_balanced_mean(
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    offset_list: Sequence[int] = QUERY_OFFSETS,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    offsets = np.asarray(offsets)
    means = [float(values[offsets == off].mean()) for off in offset_list]
    return float(np.mean(means))


def offset_means(values: np.ndarray, offsets: np.ndarray) -> Dict[int, float]:
    values = np.asarray(values, dtype=np.float64)
    offsets = np.asarray(offsets)
    return {int(off): float(values[offsets == off].mean()) for off in QUERY_OFFSETS}


def seed_means(values: np.ndarray, seed_ids: np.ndarray, offsets: np.ndarray) -> Dict[int, float]:
    values = np.asarray(values, dtype=np.float64)
    seed_ids = np.asarray(seed_ids)
    offsets = np.asarray(offsets)
    out: Dict[int, float] = {}
    for s in sorted(set(int(x) for x in np.unique(seed_ids))):
        mask = seed_ids == s
        out[int(s)] = observed_offset_balanced_mean(values[mask], offsets[mask])
    return out


def observed_confirmation_balanced_mean(
    values: np.ndarray,
    seed_ids: np.ndarray,
    offsets: np.ndarray,
    *,
    seeds: Sequence[int] = CONFIRMATION_SEEDS,
) -> float:
    per_seed = [seed_means(values, seed_ids, offsets)[int(s)] for s in seeds]
    return float(np.mean(per_seed))


def bootstrap_offset_balanced(
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    scope: str,
    metric: str = "metric",
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    offsets = np.asarray(offsets)
    rng = np.random.default_rng(stable_sha256_seed(EF6_SCOPE, scope, metric, "offset_boot"))
    boots: List[float] = []
    for _ in range(n_bootstrap):
        omeans = []
        for off in QUERY_OFFSETS:
            idx = np.where(offsets == off)[0]
            draw = rng.choice(idx, size=len(idx), replace=True)
            omeans.append(float(values[draw].mean()))
        boots.append(float(np.mean(omeans)))
    return np.asarray(boots, dtype=np.float64)


def bootstrap_confirmation_balanced(
    values: np.ndarray,
    seed_ids: np.ndarray,
    offsets: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    scope: str,
    metric: str = "metric",
    seeds: Sequence[int] = CONFIRMATION_SEEDS,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    seed_ids = np.asarray(seed_ids)
    offsets = np.asarray(offsets)
    rng = np.random.default_rng(stable_sha256_seed(EF6_SCOPE, scope, metric, "confirm_boot"))
    boots: List[float] = []
    for _ in range(n_bootstrap):
        seed_avgs: List[float] = []
        for s in seeds:
            smask = seed_ids == s
            omeans = []
            for off in QUERY_OFFSETS:
                idx = np.where(smask & (offsets == off))[0]
                draw = rng.choice(idx, size=len(idx), replace=True)
                omeans.append(float(values[draw].mean()))
            seed_avgs.append(float(np.mean(omeans)))
        boots.append(float(np.mean(seed_avgs)))
    return np.asarray(boots, dtype=np.float64)


def percentile_ci(boot: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


@dataclass
class GateResult:
    name: str
    status: str  # CONFIRMED | NOT_CONFIRMED
    point: float
    ci_low: float
    ci_high: float
    seed_means: Dict[int, float]
    offset_means: Dict[int, float]
    rule_notes: str


def _direction_pass(
    *,
    point: float,
    ci_low: float,
    ci_high: float,
    seed_m: Dict[int, float],
    offset_m: Dict[int, float],
    positive: bool,
    seeds: Sequence[int],
) -> bool:
    if positive:
        ci_ok = ci_low > 0 and ci_high > 0
        seed_ok = all(seed_m[int(s)] > 0 for s in seeds)
        off_ok = all(offset_m[int(o)] > 0 for o in QUERY_OFFSETS)
    else:
        ci_ok = ci_low < 0 and ci_high < 0
        seed_ok = all(seed_m[int(s)] < 0 for s in seeds)
        off_ok = all(offset_m[int(o)] < 0 for o in QUERY_OFFSETS)
    return ci_ok and seed_ok and off_ok


def evaluate_scalar_gate(
    name: str,
    values: np.ndarray,
    seed_ids: np.ndarray,
    offsets: np.ndarray,
    *,
    discovery: bool,
    positive: bool,
    scope: str,
) -> GateResult:
    if discovery:
        point = observed_offset_balanced_mean(values, offsets)
        boot = bootstrap_offset_balanced(values, offsets, scope=scope, metric=name)
        seed_m = {DISCOVERY_SEED: point}
        seeds_for_rule = [DISCOVERY_SEED]
    else:
        point = observed_confirmation_balanced_mean(values, seed_ids, offsets)
        boot = bootstrap_confirmation_balanced(values, seed_ids, offsets, scope=scope, metric=name)
        seed_m = seed_means(values, seed_ids, offsets)
        seeds_for_rule = list(CONFIRMATION_SEEDS)
    ci_low, ci_high = percentile_ci(boot)
    off_m = offset_means(values, offsets) if discovery else _aggregate_offset_means(values, seed_ids, offsets)
    ok = _direction_pass(
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        seed_m=seed_m,
        offset_m=off_m,
        positive=positive,
        seeds=seeds_for_rule,
    )
    return GateResult(
        name=name,
        status="CONFIRMED" if ok else "NOT_CONFIRMED",
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        seed_means=seed_m,
        offset_means=off_m,
        rule_notes="95% CI + seed + offset means",
    )


def _aggregate_offset_means(values, seed_ids, offsets) -> Dict[int, float]:
    values = np.asarray(values)
    seed_ids = np.asarray(seed_ids)
    offsets = np.asarray(offsets)
    out: Dict[int, float] = {}
    for off in QUERY_OFFSETS:
        out[int(off)] = float(values[offsets == off].mean())
    return out


def evaluate_phi_gate(
    name: str,
    values: np.ndarray,
    seed_ids: np.ndarray,
    offsets: np.ndarray,
    phi_score_values: np.ndarray,
    phi_r_values: np.ndarray,
    phi_o_values: np.ndarray,
    *,
    positive: bool,
    scope: str,
) -> GateResult:
    base = evaluate_scalar_gate(
        name,
        values,
        seed_ids,
        offsets,
        discovery=False,
        positive=positive,
        scope=scope,
    )
    mt = abs(float(np.mean(values)))
    mr = abs(float(np.mean(phi_r_values)))
    mo = abs(float(np.mean(phi_o_values)))
    score_abs = mt + mr + mo
    share_num = mt if positive else mr
    share = share_num / score_abs if score_abs >= 1e-6 else 0.0
    mag_ok = score_abs >= 1e-6 and share >= PHI_SHARE_THRESHOLD
    status = "CONFIRMED" if base.status == "CONFIRMED" and mag_ok else "NOT_CONFIRMED"
    note = f"share={share:.4f}; score_abs_total={score_abs:.6f}"
    return GateResult(
        name=name,
        status=status,
        point=base.point,
        ci_low=base.ci_low,
        ci_high=base.ci_high,
        seed_means=base.seed_means,
        offset_means=base.offset_means,
        rule_notes=base.rule_notes + "; " + note,
    )


def ef6_gate(gates: Dict[str, str]) -> str:
    confirmed = {k: v == "CONFIRMED" for k, v in gates.items()}
    if all(confirmed.get(f"G{i}", False) for i in range(5)):
        return "PASS"
    if confirmed.get("G0") and confirmed.get("G1") and confirmed.get("G2"):
        g3 = confirmed.get("G3", False)
        g4 = confirmed.get("G4", False)
        if g3 ^ g4:
            return "PARTIAL"
    return "FAIL"


@dataclass
class PairCandidate:
    primary_layer: int
    primary_head: int
    candidate_layer: int
    candidate_head: int
    conditional_gain_balanced: float
    conditional_effect_balanced: float
    candidate_clean_attention: float
    primary_intact_effect: float
    eligible: bool
    reason: str = ""


def select_head_pair(candidates: Sequence[PairCandidate]) -> Optional[PairCandidate]:
    eligible = [p for p in candidates if p.eligible]
    if not eligible:
        return None
    eligible.sort(
        key=lambda p: (
            -p.conditional_gain_balanced,
            -p.conditional_effect_balanced,
            -p.candidate_clean_attention,
            -p.primary_intact_effect,
            p.primary_layer,
            p.primary_head,
            p.candidate_layer,
            p.candidate_head,
        )
    )
    return eligible[0]


def primary_eligible(
    *,
    clean_attention: float,
    intact_ci_low: float,
    intact_ci_high: float,
    offset_intact_means: Dict[int, float],
    has_later_attention_head: bool,
) -> Tuple[bool, str]:
    if clean_attention < ATTN_THRESHOLD:
        return False, "clean_attention"
    if not (intact_ci_low > 0 and intact_ci_high > 0):
        return False, "intact_ci"
    if not all(offset_intact_means[o] > 0 for o in QUERY_OFFSETS):
        return False, "offset_intact"
    if not has_later_attention_head:
        return False, "no_later_head"
    return True, "ok"


def candidate_pair_eligible(
    *,
    primary_layer: int,
    candidate_layer: int,
    cond_ci_low: float,
    cond_ci_high: float,
    gain_ci_low: float,
    gain_ci_high: float,
    offset_cond_means: Dict[int, float],
    offset_gain_means: Dict[int, float],
) -> Tuple[bool, str]:
    if primary_layer >= candidate_layer:
        return False, "layer_order"
    if not (cond_ci_low > 0 and cond_ci_high > 0):
        return False, "conditional_ci"
    if not all(offset_cond_means[o] > 0 for o in QUERY_OFFSETS):
        return False, "offset_cond"
    if not (gain_ci_low > 0 and gain_ci_high > 0):
        return False, "gain_ci"
    if not all(offset_gain_means[o] > 0 for o in QUERY_OFFSETS):
        return False, "offset_gain"
    return True, "ok"


def write_selection_lock(
    out_path,
    *,
    selection: PairCandidate,
    preregistration_sha: str,
    dataset_sha: str,
    discovery_seed: int = DISCOVERY_SEED,
) -> None:
    payload = {
        "preregistration_sha": preregistration_sha,
        "discovery_seed": discovery_seed,
        "dataset_sha256": dataset_sha,
        "fine_attributions_used_for_selection": False,
        "primary": {
            "layer": selection.primary_layer,
            "head": selection.primary_head,
            "head_name": head_name(selection.primary_layer, selection.primary_head),
        },
        "candidate": {
            "layer": selection.candidate_layer,
            "head": selection.candidate_head,
            "head_name": head_name(selection.candidate_layer, selection.candidate_head),
        },
        "selection_metrics": {
            "conditional_gain_balanced": selection.conditional_gain_balanced,
            "conditional_effect_balanced": selection.conditional_effect_balanced,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_selection_lock(path) -> Dict:
    return json.loads(Path(path).read_text())


def run_ef6_aggregate(repo: Path, aggregate_dir: Path) -> Dict:
    """Aggregate held-out confirmation results and compute EF6 gate."""
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    ef6_root = repo / "results/extended_fali/ef6_pythia_induction"
    frames = []
    for seed in CONFIRMATION_SEEDS:
        attr = pd.read_csv(ef6_root / f"confirmation/seed_{seed}/attribution_raw.csv")
        frames.append(attr)
    attr = pd.concat(frames, ignore_index=True)
    attr.to_csv(aggregate_dir / "attribution_merged_raw.csv", index=False)

    seeds = attr["seed"].values
    offsets = attr["query_offset"].values

    # Recompute behavior from discovery + confirmation behavior files if present
    beh_frames = []
    for seed in CONFIRMATION_SEEDS:
        p = ef6_root / f"confirmation/seed_{seed}/behavior_results.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing confirmation behavior: {p}")
        beh_frames.append(pd.read_csv(p))
    behavior = pd.concat(beh_frames, ignore_index=True)

    g0 = evaluate_scalar_gate(
        "G0",
        behavior["induction_advantage"].values,
        behavior["seed"].values,
        behavior["query_offset"].values,
        discovery=False,
        positive=True,
        scope="g0_confirm",
    )

    g1 = evaluate_scalar_gate(
        "G1",
        behavior["primary_drop"].values,
        behavior["seed"].values,
        behavior["query_offset"].values,
        discovery=False,
        positive=True,
        scope="g1",
    )

    g2_ce = evaluate_scalar_gate(
        "G2_conditional_effect",
        attr["conditional_effect"].values,
        seeds,
        offsets,
        discovery=False,
        positive=True,
        scope="g2_ce",
    )
    g2_cg = evaluate_scalar_gate(
        "G2_conditional_gain",
        attr["conditional_gain"].values,
        seeds,
        offsets,
        discovery=False,
        positive=True,
        scope="g2_cg",
    )
    g2_status = "CONFIRMED" if g2_ce.status == "CONFIRMED" and g2_cg.status == "CONFIRMED" else "NOT_CONFIRMED"

    g3 = evaluate_phi_gate(
        "G3_phi_T",
        attr["phi_T"].values,
        seeds,
        offsets,
        attr["Phi_SCORE"].values,
        attr["phi_R"].values,
        attr["phi_O"].values,
        positive=True,
        scope="g3",
    )
    g4 = evaluate_phi_gate(
        "G4_phi_R",
        attr["phi_R"].values,
        seeds,
        offsets,
        attr["Phi_SCORE"].values,
        attr["phi_R"].values,
        attr["phi_O"].values,
        positive=False,
        scope="g4",
    )

    gates = {
        "G0": g0.status,
        "G1": g1.status,
        "G2": g2_status,
        "G3": g3.status,
        "G4": g4.status,
    }
    ef6_gate_status = ef6_gate(gates)

    hypothesis = {
        "G0": g0.__dict__,
        "G1": g1.__dict__,
        "G2_conditional_effect": g2_ce.__dict__,
        "G2_conditional_gain": g2_cg.__dict__,
        "G2_status": g2_status,
        "G3": g3.__dict__,
        "G4": g4.__dict__,
        "phi_O_descriptive": {
            "mean": float(attr["phi_O"].mean()),
            "note": "exploratory only — not confirmatory",
        },
    }
    (aggregate_dir / "hypothesis_results.json").write_text(json.dumps(hypothesis, indent=2, default=str) + "\n")
    (aggregate_dir / "ef6_gate.json").write_text(
        json.dumps({"EF6_GATE": ef6_gate_status, "gates": gates}, indent=2) + "\n"
    )

    metrics = {
        "induction_advantage": behavior["induction_advantage"].values,
        "primary_drop": behavior["primary_drop"].values,
        "conditional_effect": attr["conditional_effect"].values,
        "conditional_gain": attr["conditional_gain"].values,
        "phi_T": attr["phi_T"].values,
        "phi_R": attr["phi_R"].values,
        "phi_O": attr["phi_O"].values,
        "phi_V": attr["phi_V"].values,
        "phi_C": attr["phi_C"].values,
    }
    exact_rows = []
    seed_rows = []
    offset_rows = []
    boot_rows = []
    for name, vals in metrics.items():
        point = observed_confirmation_balanced_mean(vals, seeds, offsets)
        boot = bootstrap_confirmation_balanced(vals, seeds, offsets, scope=f"agg_{name}", metric=name)
        lo, hi = percentile_ci(boot)
        exact_rows.append({"metric": name, "exact_balanced_mean": point, "ci_low": lo, "ci_high": hi})
        boot_rows.append({"metric": name, "ci_low": lo, "ci_high": hi, "point_observed": point})
        for s, sm in seed_means(vals, seeds, offsets).items():
            seed_rows.append({"metric": name, "seed": s, "mean": sm})
        for off, om in _aggregate_offset_means(vals, seeds, offsets).items():
            offset_rows.append({"metric": name, "query_offset": off, "mean": om})
    pd.DataFrame(exact_rows).to_csv(aggregate_dir / "exact_means.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(aggregate_dir / "seed_means.csv", index=False)
    pd.DataFrame(offset_rows).to_csv(aggregate_dir / "offset_means.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(aggregate_dir / "bootstrap_ci.csv", index=False)
    pd.concat(beh_frames, ignore_index=True).to_csv(aggregate_dir / "behavior_results.csv", index=False)

    summary_lines = [
        "# EF-6 Pythia induction transfer — scientific summary",
        "",
        f"**EF6_GATE:** {ef6_gate_status}",
        "",
        "## Confirmatory gates",
        *[f"- {k}: {v}" for k, v in gates.items()],
        "",
        "## phi_O (exploratory only)",
        f"Mean phi_O = {float(attr['phi_O'].mean()):.6f} (not confirmatory).",
        "",
        "Do not claim universal mechanism or cross-architecture universality.",
    ]
    (aggregate_dir / "scientific_summary.md").write_text("\n".join(summary_lines) + "\n")
    (aggregate_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "phase": "EF-6",
                "model": "EleutherAI/pythia-160m",
                "point_estimate": "observed_offset_balanced_confirmation_mean",
                "bootstrap_draws": N_BOOTSTRAP,
                "confirmation_seeds": list(CONFIRMATION_SEEDS),
            },
            indent=2,
        )
        + "\n"
    )
    val_pass = True
    for s in CONFIRMATION_SEEDS:
        vp = ef6_root / f"confirmation/seed_{s}/validation_report.json"
        if not vp.exists():
            val_pass = False
            break
        if not json.loads(vp.read_text()).get("overall_pass", False):
            val_pass = False
    (aggregate_dir / "validation_summary.json").write_text(
        json.dumps({"overall_pass": val_pass}, indent=2) + "\n"
    )

    integrity = audit_ef5_historical_integrity(repo)
    (aggregate_dir / "historical_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    return {"EF6_GATE": ef6_gate_status, "gates": gates, "integrity": integrity}
