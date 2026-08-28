"""
Phase-1C: Corrected functional classification from existing Phase-1B raw results.

Does NOT run the model. Reads only:
  results/revision_phase1/full/seed_*/candidate_raw.csv

Writes to:
  results/revision_phase1/full/aggregate_corrected/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from revision_config import BACKUP_NAME_MOVERS, head_name_from_pair  # noqa: E402
from revision_stats import pearson_spearman, stratified_bootstrap_ci  # noqa: E402

SEEDS = [42, 1, 8, 22]
METRICS = ["intact_effect", "conditional_effect", "conditional_gain"]
N_BOOT = 10000


def stable_str_hash_int(s: str, mod: int = 10000) -> int:
    """Process-stable integer hash (SHA256), not Python's randomized hash()."""
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % mod


def ci_sign(low: float, high: float) -> str:
    if low > 0:
        return "pos"
    if high < 0:
        return "neg"
    return "includes_zero"


def classify_functional(
    mean_cond: float,
    mean_gain: float,
    n_cond_pos: int,
    n_gain_pos: int,
    n_cond_ci_pos: int,
    n_gain_ci_pos: int,
    n_cond_neg: int,
    n_gain_neg: int,
) -> str:
    """
    Corrected Phase-1C taxonomy.

    Positive conditional gain alone is never enough for recruitment.
    Recruitment requires positive conditional_effect as well.
    """
    # A. Strong positive functional recruitment
    if (
        mean_cond > 0
        and mean_gain > 0
        and n_cond_pos >= 3
        and n_cond_ci_pos >= 3
        and n_gain_ci_pos >= 3
    ):
        return "strong_positive_functional_recruitment"

    # B. Suggestive positive functional recruitment
    if (
        mean_cond > 0
        and mean_gain > 0
        and n_cond_pos >= 3
        and n_gain_pos >= 3
    ):
        return "suggestive_positive_functional_recruitment"

    # C. Supportive but downregulated
    if mean_cond > 0 and mean_gain < 0 and n_cond_pos >= 3 and n_gain_neg >= 3:
        return "supportive_but_downregulated"

    # D. Less suppressive after knockout (NOT recruitment)
    if mean_cond < 0 and mean_gain > 0 and n_cond_neg >= 3 and n_gain_pos >= 3:
        return "less_suppressive_after_knockout"

    # E. More suppressive after knockout
    if mean_cond < 0 and mean_gain < 0 and n_cond_neg >= 3 and n_gain_neg >= 3:
        return "more_suppressive_after_knockout"

    # F. Supportive with no stable modulation
    if mean_cond > 0 and n_cond_pos >= 3 and not (n_gain_pos >= 3 or n_gain_neg >= 3):
        return "supportive_no_stable_modulation"

    # G. Suppressive with no stable modulation
    if mean_cond < 0 and n_cond_neg >= 3 and not (n_gain_pos >= 3 or n_gain_neg >= 3):
        return "suppressive_no_stable_modulation"

    return "unstable_mixed"


def classify_sign_flip(
    mean_intact: float,
    mean_cond: float,
    n_intact_neg: int,
    n_intact_pos: int,
    n_cond_pos: int,
    n_cond_neg: int,
    n_intact_ci_neg: int,
    n_intact_ci_pos: int,
    n_cond_ci_pos: int,
    n_cond_ci_neg: int,
) -> str:
    """
    Strong flip requires mean signs + CI consistency in >=3/4 seeds.
    Suggestive flip requires mean signs + seed-mean consistency without full CI support.
    """
    # Strong / suggestive: suppressive -> supportive
    if mean_intact < 0 and mean_cond > 0:
        if n_intact_ci_neg >= 3 and n_cond_ci_pos >= 3:
            return "strong_suppressive_to_supportive"
        if n_intact_neg >= 3 and n_cond_pos >= 3:
            return "suggestive_suppressive_to_supportive"
        return "unstable"

    # Strong / suggestive: supportive -> suppressive
    if mean_intact > 0 and mean_cond < 0:
        if n_intact_ci_pos >= 3 and n_cond_ci_neg >= 3:
            return "strong_supportive_to_suppressive"
        if n_intact_pos >= 3 and n_cond_neg >= 3:
            return "suggestive_supportive_to_suppressive"
        return "unstable"

    return "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-root", default="results/revision_phase1/full")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=42, help="Bootstrap RNG base seed")
    args = p.parse_args()

    full_root = REPO / args.full_root
    out_dir = full_root / "aggregate_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed bootstrap summaries
    boot_rows = []
    seed_head_stats: Dict[Tuple[int, str], Dict] = {}

    for seed in SEEDS:
        raw = pd.read_csv(full_root / f"seed_{seed}" / "candidate_raw.csv")
        assert raw["example_idx"].nunique() == 200
        assert set(raw["template_id"]) == {"A", "B"}

        for hname, sub in raw.groupby("head_name", sort=False):
            sub = sub.sort_values("example_idx")
            strata = sub["template_id"].to_numpy()
            role = sub["head_role"].iloc[0]
            spike = float(sub["corrected_attention_spike"].mean())

            stats_h = {
                "seed": seed,
                "head_name": hname,
                "head_role": role,
                "mean_attention_spike": spike,
            }
            for metric in METRICS:
                vals = sub[metric].to_numpy(dtype=np.float64)
                # Distinct bootstrap seed per (seed, head, metric); SHA256-stable
                boot_seed = (
                    args.seed
                    + 1009 * seed
                    + 17 * stable_str_hash_int(str(hname), 10000)
                    + 3 * METRICS.index(metric)
                ) % (2**31 - 1)
                mean, lo, hi = stratified_bootstrap_ci(
                    vals,
                    strata,
                    n_bootstrap=args.n_bootstrap,
                    seed=boot_seed,
                )
                sign = ci_sign(lo, hi)
                stats_h[f"mean_{metric}"] = mean
                stats_h[f"{metric}_ci_low"] = lo
                stats_h[f"{metric}_ci_high"] = hi
                stats_h[f"{metric}_ci_sign"] = sign
                stats_h[f"{metric}_seed_positive"] = bool(mean > 0)
                stats_h[f"{metric}_seed_negative"] = bool(mean < 0)

                boot_rows.append(
                    {
                        "seed": seed,
                        "head_name": hname,
                        "head_role": role,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci_sign": sign,
                        "n_bootstrap": args.n_bootstrap,
                        "stratified_by": "template_id",
                    }
                )

            seed_head_stats[(seed, hname)] = stats_h

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "bootstrap_intact_conditional_gain.csv", index=False)

    # Aggregate across seeds + classification
    heads = list(dict.fromkeys(boot_df["head_name"].tolist()))
    class_rows = []

    for hname in heads:
        role = seed_head_stats[(SEEDS[0], hname)]["head_role"]
        spikes = [seed_head_stats[(s, hname)]["mean_attention_spike"] for s in SEEDS]

        metric_aggs = {}
        for metric in METRICS:
            means = [seed_head_stats[(s, hname)][f"mean_{metric}"] for s in SEEDS]
            ci_signs = [seed_head_stats[(s, hname)][f"{metric}_ci_sign"] for s in SEEDS]
            n_pos = sum(1 for m in means if m > 0)
            n_neg = sum(1 for m in means if m < 0)
            n_ci_pos = sum(1 for c in ci_signs if c == "pos")
            n_ci_neg = sum(1 for c in ci_signs if c == "neg")
            metric_aggs[metric] = {
                "mean": float(np.mean(means)),
                "std": float(np.std(means, ddof=1)),
                "n_seeds_positive": n_pos,
                "n_seeds_negative": n_neg,
                "n_ci_pos": n_ci_pos,
                "n_ci_neg": n_ci_neg,
                "per_seed": {s: means[i] for i, s in enumerate(SEEDS)},
                "ci_signs": {s: ci_signs[i] for i, s in enumerate(SEEDS)},
            }

        intact = metric_aggs["intact_effect"]
        cond = metric_aggs["conditional_effect"]
        gain = metric_aggs["conditional_gain"]

        functional_class = classify_functional(
            mean_cond=cond["mean"],
            mean_gain=gain["mean"],
            n_cond_pos=cond["n_seeds_positive"],
            n_gain_pos=gain["n_seeds_positive"],
            n_cond_ci_pos=cond["n_ci_pos"],
            n_gain_ci_pos=gain["n_ci_pos"],
            n_cond_neg=cond["n_seeds_negative"],
            n_gain_neg=gain["n_seeds_negative"],
        )
        sign_flip = classify_sign_flip(
            mean_intact=intact["mean"],
            mean_cond=cond["mean"],
            n_intact_neg=intact["n_seeds_negative"],
            n_intact_pos=intact["n_seeds_positive"],
            n_cond_pos=cond["n_seeds_positive"],
            n_cond_neg=cond["n_seeds_negative"],
            n_intact_ci_neg=intact["n_ci_neg"],
            n_intact_ci_pos=intact["n_ci_pos"],
            n_cond_ci_pos=cond["n_ci_pos"],
            n_cond_ci_neg=cond["n_ci_neg"],
        )

        # Guardrail: never call negative conditional effect a recruit
        assert not (
            cond["mean"] < 0
            and functional_class
            in {
                "strong_positive_functional_recruitment",
                "suggestive_positive_functional_recruitment",
            }
        )

        class_rows.append(
            {
                "head_name": hname,
                "head_role": role,
                "mean_intact_effect": intact["mean"],
                "std_intact_effect": intact["std"],
                "n_seeds_intact_positive": intact["n_seeds_positive"],
                "n_seeds_intact_ci_pos": intact["n_ci_pos"],
                "n_seeds_intact_ci_neg": intact["n_ci_neg"],
                "mean_conditional_effect": cond["mean"],
                "std_conditional_effect": cond["std"],
                "n_seeds_conditional_effect_positive": cond["n_seeds_positive"],
                "n_seeds_conditional_effect_ci_pos": cond["n_ci_pos"],
                "n_seeds_conditional_effect_ci_neg": cond["n_ci_neg"],
                "mean_conditional_gain": gain["mean"],
                "std_conditional_gain": gain["std"],
                "n_seeds_conditional_gain_positive": gain["n_seeds_positive"],
                "n_seeds_conditional_gain_ci_pos": gain["n_ci_pos"],
                "n_seeds_conditional_gain_ci_neg": gain["n_ci_neg"],
                "mean_attention_spike": float(np.mean(spikes)),
                "std_attention_spike": float(np.std(spikes, ddof=1)),
                "seed42_intact": intact["per_seed"][42],
                "seed1_intact": intact["per_seed"][1],
                "seed8_intact": intact["per_seed"][8],
                "seed22_intact": intact["per_seed"][22],
                "seed42_conditional_effect": cond["per_seed"][42],
                "seed1_conditional_effect": cond["per_seed"][1],
                "seed8_conditional_effect": cond["per_seed"][8],
                "seed22_conditional_effect": cond["per_seed"][22],
                "seed42_conditional_gain": gain["per_seed"][42],
                "seed1_conditional_gain": gain["per_seed"][1],
                "seed8_conditional_gain": gain["per_seed"][8],
                "seed22_conditional_gain": gain["per_seed"][22],
                "sign_flip_type": sign_flip,
                "corrected_functional_class": functional_class,
            }
        )

    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(out_dir / "functional_classification.csv", index=False)

    # Canonical backups only; genuine recruits must have positive conditional effect
    backup_names = {head_name_from_pair(h) for h in BACKUP_NAME_MOVERS}
    backups = class_df[class_df["head_name"].isin(backup_names)].copy()
    genuine = backups[
        backups["corrected_functional_class"].isin(
            [
                "strong_positive_functional_recruitment",
                "suggestive_positive_functional_recruitment",
            ]
        )
        & (backups["mean_conditional_effect"] > 0)
    ].sort_values("mean_conditional_gain", ascending=False)
    backups_out = backups.sort_values("mean_conditional_gain", ascending=False)
    backups_out.to_csv(out_dir / "canonical_backup_summary.csv", index=False)
    genuine.to_csv(out_dir / "canonical_backup_genuine_recruits_ranked.csv", index=False)

    # Correlations: attention spike vs genuine recruitment markers
    corr_rows = []
    for name, mask in [
        ("all_12", np.ones(len(class_df), dtype=bool)),
        ("canonical_backups", class_df["head_name"].isin(backup_names).to_numpy()),
        (
            "positive_conditional_effect_only",
            (class_df["mean_conditional_effect"] > 0).to_numpy(),
        ),
    ]:
        sub = class_df.loc[mask]
        c_gain = pearson_spearman(
            sub["mean_attention_spike"], sub["mean_conditional_gain"]
        )
        c_eff = pearson_spearman(
            sub["mean_attention_spike"], sub["mean_conditional_effect"]
        )
        corr_rows.append(
            {
                "subset": name,
                "n": int(mask.sum()),
                "x": "attention_spike",
                "y": "conditional_gain",
                **{f"gain_{k}": v for k, v in c_gain.items()},
            }
        )
        corr_rows.append(
            {
                "subset": name,
                "n": int(mask.sum()),
                "x": "attention_spike",
                "y": "conditional_effect",
                **{f"effect_{k}": v for k, v in c_eff.items()},
            }
        )
    pd.DataFrame(corr_rows).to_csv(out_dir / "attention_vs_function_correlations.csv", index=False)

    # Markdown artifacts
    write_taxonomy(out_dir / "functional_taxonomy.md")
    write_corrected_summary(out_dir, class_df, backups, genuine, corr_rows)
    write_phase2_recommendation(out_dir, class_df)

    print("Wrote corrected outputs to", out_dir)
    print(class_df[["head_name", "mean_conditional_effect", "mean_conditional_gain", "sign_flip_type", "corrected_functional_class"]].to_string(index=False))


def write_taxonomy(path: Path) -> None:
    path.write_text(
        """# Functional taxonomy (Phase-1C)

These labels separate **helpfulness under knockout** from **change relative to intact**.

## Quantities

- **Intact effect** `A − D_h`: how much the head helps IOI in the clean model.
- **Conditional effect** `B − C_h`: how much the head helps IOI after primary Name Movers are ablated.
- **Conditional gain** `(B − C_h) − (A − D_h)`: how much that helpfulness *increases* after the knockout.

## Classes

### Strong positive functional recruitment
The head is helpful under the corrected primary knockout **and** becomes more helpful after that knockout, with stable seed means and bootstrap CIs.

### Suggestive positive functional recruitment
Same directional pattern (helpful + more helpful), but CI consistency is weaker.

### Supportive but downregulated
The head remains helpful under knockout, but less helpful than in the intact model (negative conditional gain). Attention may still rise.

### Less suppressive after knockout
The head still hurts IOI under knockout (negative conditional effect), but hurts *less* than when intact (positive conditional gain). **This is not recruitment.**

### More suppressive after knockout
The head hurts IOI under knockout and hurts even more than when intact.

### Supportive / suppressive with no stable modulation
Direction of conditional effect is stable, but conditional gain is not.

### Unstable / mixed
Seed-level signs disagree; no robust claim.

## Sign flips

- `strong_suppressive_to_supportive`: mean intact < 0, mean conditional > 0, CI excludes 0 in matching direction for >=3/4 seeds.
- `suggestive_suppressive_to_supportive`: mean signs flip and seed means agree (>=3/4), but strong CI requirement fails.
- Analogous `strong_` / `suggestive_supportive_to_suppressive` for the opposite flip.
- `none` / `unstable`: no robust flip.
"""
    )


def write_corrected_summary(
    out_dir: Path,
    class_df: pd.DataFrame,
    backups: pd.DataFrame,
    genuine: pd.DataFrame,
    corr_rows: list,
) -> None:
    focus = ["L10H2", "L10H1", "L9H7", "L10H10", "L11H2", "L10H7"]
    lines = [
        "# Phase-1 corrected summary",
        "",
        "Interpretation rule: **positive conditional gain alone is not functional recruitment**.",
        "Recruitment requires positive **conditional effect** as well.",
        "",
        "## 1. Canonical backups that genuinely become more helpful",
        "",
    ]
    if len(genuine) == 0:
        lines.append("(none under strong/suggestive criteria)")
    else:
        for _, r in genuine.iterrows():
            lines.append(
                f"- **{r['head_name']}** ({r['corrected_functional_class']}): "
                f"cond_effect={r['mean_conditional_effect']:.4f}, "
                f"cond_gain={r['mean_conditional_gain']:.4f}, "
                f"spike={r['mean_attention_spike']:.4f}"
            )
    lines += ["", "## 2. Remain helpful but become less important (downregulated)", ""]
    down = class_df[class_df["corrected_functional_class"] == "supportive_but_downregulated"]
    if len(down) == 0:
        lines.append("(none)")
    else:
        for _, r in down.iterrows():
            lines.append(
                f"- **{r['head_name']}**: cond_effect={r['mean_conditional_effect']:.4f}, "
                f"cond_gain={r['mean_conditional_gain']:.4f}, spike={r['mean_attention_spike']:.4f}"
            )

    lines += ["", "## 3. Remain suppressive despite positive gain (less-suppressive, not recruits)", ""]
    less = class_df[class_df["corrected_functional_class"] == "less_suppressive_after_knockout"]
    if len(less) == 0:
        lines.append("(none)")
    else:
        for _, r in less.iterrows():
            lines.append(
                f"- **{r['head_name']}** ({r['head_role']}): "
                f"cond_effect={r['mean_conditional_effect']:.4f}, "
                f"cond_gain={r['mean_conditional_gain']:.4f}"
            )

    lines += ["", "## 4. Functional sign flips", ""]
    flips = class_df[class_df["sign_flip_type"] != "none"]
    if len(flips) == 0:
        lines.append("(none)")
    else:
        for _, r in flips.iterrows():
            lines.append(
                f"- **{r['head_name']}**: `{r['sign_flip_type']}` "
                f"(intact={r['mean_intact_effect']:.4f}, "
                f"cond_effect={r['mean_conditional_effect']:.4f})"
            )

    lines += ["", "## 5. Does attention spike predict genuine recruitment?", ""]
    # Find correlations of interest
    for row in corr_rows:
        if row["subset"] == "canonical_backups" and row["y"] == "conditional_effect":
            lines.append(
                f"- Among canonical backups, attention spike vs conditional effect: "
                f"Pearson={row.get('effect_pearson_r')}, Spearman={row.get('effect_spearman_r')}."
            )
        if row["subset"] == "canonical_backups" and row["y"] == "conditional_gain":
            lines.append(
                f"- Among canonical backups, attention spike vs conditional gain: "
                f"Pearson={row.get('gain_pearson_r')}, Spearman={row.get('gain_spearman_r')}."
            )
    lines.append(
        "- Conclusion: a large attention spike is **not sufficient** for recruitment "
        "(see L10H10: large spike, positive conditional effect, but negative gain / downregulated). "
        "Absence of a spike is also not decisive for same-layer heads (see L9H7)."
    )

    lines += ["", "## Focused head reassessment", ""]
    lines.append(
        "| head | role | intact | cond_effect | cond_gain | spike | sign_flip | class |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for h in focus:
        r = class_df[class_df["head_name"] == h].iloc[0]
        lines.append(
            f"| {h} | {r['head_role']} | {r['mean_intact_effect']:.4f} | "
            f"{r['mean_conditional_effect']:.4f} | {r['mean_conditional_gain']:.4f} | "
            f"{r['mean_attention_spike']:.4f} | {r['sign_flip_type']} | "
            f"{r['corrected_functional_class']} |"
        )

    lines += ["", "## Full classification table", ""]
    lines.append("```")
    lines.append(
        class_df[
            [
                "head_name",
                "head_role",
                "mean_intact_effect",
                "mean_conditional_effect",
                "mean_conditional_gain",
                "mean_attention_spike",
                "sign_flip_type",
                "corrected_functional_class",
            ]
        ].to_string(index=False)
    )
    lines.append("```")
    lines.append("")
    (out_dir / "phase1_corrected_summary.md").write_text("\n".join(lines) + "\n")


def write_phase2_recommendation(out_dir: Path, class_df: pd.DataFrame) -> None:
    by = class_df.set_index("head_name")

    def cls(h: str) -> str:
        return by.loc[h, "corrected_functional_class"]

    def flip(h: str) -> str:
        return by.loc[h, "sign_flip_type"]

    lines = [
        "# Phase-2 FALI target recommendation (contrastive panel)",
        "",
        "Recommend a **contrastive panel**, not a single favorite head.",
        "These recommendations are derived from corrected Phase-1C labels.",
        "",
        "## Recommended panel",
        "",
    ]

    panel = [
        (
            "L10H2",
            "Putative strong genuine recruited backup",
            "Tests whether FALI target-logit / QK changes accompany true positive conditional effect + gain.",
        ),
        (
            "L10H1",
            f"Possible sign-flip recruit (`{flip('L10H1')}`; class=`{cls('L10H1')}`)",
            "Falsifies ‘spike = recruitment’ only if FALI shows routing while function flips or remains weak; "
            "or supports a clean recruit if both effect and gain are positive.",
        ),
        (
            "L10H10",
            f"Large attention spike but downregulated function (`{cls('L10H10')}`)",
            "Critical negative control for attention-based self-repair claims: spike without recruitment.",
        ),
        (
            "L9H7",
            f"Functional change with zero same-layer attention spike (`{cls('L9H7')}`)",
            "Tests whether FALI/QK can detect recruitment when attention spike is architecturally invisible.",
        ),
        (
            "L10H7",
            f"Suppressor / less-suppressive control (`{cls('L10H7')}`)",
            "Must not be labeled a backup recruit; FALI should reflect suppressor dynamics / reduced suppression.",
        ),
    ]

    for h, role, why in panel:
        r = by.loc[h]
        lines.append(f"### {h}")
        lines.append(f"- Role in panel: {role}")
        lines.append(
            f"- Numbers: intact={r['mean_intact_effect']:.4f}, "
            f"cond_effect={r['mean_conditional_effect']:.4f}, "
            f"cond_gain={r['mean_conditional_gain']:.4f}, "
            f"spike={r['mean_attention_spike']:.4f}"
        )
        lines.append(f"- Falsifiable contrast: {why}")
        lines.append("")

    lines += [
        "## Why not only L10H10",
        "",
        "Under corrected primary knockout, L10H10 remains helpful but becomes *less* important "
        "(supportive but downregulated). Using it alone as the FALI exemplar would overfit the old narrative.",
        "",
        "## Stop",
        "",
        "Do not begin Phase 2 until research-lead review of this panel.",
        "",
    ]
    (out_dir / "phase2_target_recommendation.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
