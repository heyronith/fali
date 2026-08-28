"""
Revision Phase 2/3: Descriptive FALI + causal score / pattern / QK interventions.

Writes ONLY under results/revision_phase2/ (never historical results/ or Phase-1 raw).

Smoke: N=8. Full: N=200 with --allow-full.

Causal interventions run for ALL 12 Phase-1 candidates.
DETAILED_PANEL remains a reporting subset only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from revision_config import (  # noqa: E402
    PRIMARY_NAME_MOVERS,
    all_phase1_candidates,
    head_name_from_pair,
    role_for_head,
)
from revision_dataset import (  # noqa: E402
    build_dataset_metadata,
    make_validated_ioi_dataset,
)
from revision_fali import (  # noqa: E402
    attention_margin,
    competing_logsumexp,
    extract_target_and_competing,
    fali_shapley,
    qk_target_shapley,
    sigmoid_margin_prob,
)
from revision_interventions import build_zero_ablation_hooks  # noqa: E402
from revision_score_interventions import (  # noqa: E402
    hooks_for_k_patch,
    hooks_for_pattern_patch,
    hooks_for_q_patch,
    hooks_for_qk_patch,
    hooks_for_score_hybrid,
)

Head = Tuple[int, int]

# Contrastive reporting subset (causal still runs for all 12 candidates)
DETAILED_PANEL: List[Head] = [
    (10, 2),   # L10H2 strong recruit
    (10, 1),   # L10H1 suggestive recruit / suggestive sign flip
    (10, 10),  # L10H10 large spike, downregulated
    (9, 7),    # L9H7 same-layer scope test
    (10, 7),   # L10H7 suppressor / less-suppressive
    (10, 6),   # L10H6 supportive, no stable modulation
]

REPORTING_PANEL_NAMES = [head_name_from_pair(h) for h in DETAILED_PANEL]

FORBIDDEN_PREFIXES = (
    "results/raw",
    "results/summary",
    "results/revision_phase1/full/seed_",
)

HISTORICAL_HASH_TARGETS = [
    "results/summary/attention_results.json",
    "results/raw/fali_decomposition_raw.csv",
    "results/revision_phase1/full/seed_42/candidate_raw.csv",
]

CACHE_FILTER = None  # set after knowing layers


def resolve_device(requested: Optional[str]) -> str:
    if requested and requested.lower() != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_protected(repo: Path) -> Dict[str, Dict]:
    snap = {}
    for rel in HISTORICAL_HASH_TARGETS:
        p = repo / rel
        st = p.stat() if p.exists() else None
        snap[rel] = {
            "sha256": file_sha256(p),
            "size": st.st_size if st else None,
            "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            if st
            else None,
        }
    return snap


def assert_safe_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    repo = _REPO_ROOT.resolve()
    try:
        rel = str(resolved.relative_to(repo)).replace("\\", "/")
    except ValueError:
        rel = str(resolved)
    for bad in FORBIDDEN_PREFIXES:
        if rel == bad or rel.startswith(bad):
            raise ValueError(f"Refusing to write into protected path: {rel}")
    if "results/revision_phase2" not in rel:
        raise ValueError(
            f"Phase-2 outputs must live under results/revision_phase2/, got {rel}"
        )


def compute_logit_diff(logits: torch.Tensor, dataset) -> torch.Tensor:
    batch = dataset.batch_size
    device = logits.device
    end_idx = dataset.end.to(device)
    batch_idx = torch.arange(batch, device=device)
    logits_at_end = logits[batch_idx, end_idx]
    io_logits = logits_at_end[batch_idx, dataset.io_token_ids.to(device)]
    s_logits = logits_at_end[batch_idx, dataset.s_token_ids.to(device)]
    return io_logits - s_logits


def names_filter_for_layers(layers: Sequence[int]) -> List[str]:
    names = []
    for l in sorted(set(layers)):
        names.extend(
            [
                f"blocks.{l}.attn.hook_attn_scores",
                f"blocks.{l}.attn.hook_pattern",
                f"blocks.{l}.attn.hook_q",
                f"blocks.{l}.attn.hook_k",
            ]
        )
    return names


def align_dataset_to_phase1(
    meta: pd.DataFrame,
    phase1_meta_path: Path,
    alignment_report_path: Path,
    smoke: bool,
) -> None:
    lines = ["# Dataset alignment report", ""]
    if not phase1_meta_path.exists():
        msg = f"Phase-1 metadata missing at {phase1_meta_path}"
        lines += [f"**FAIL**: {msg}", ""]
        alignment_report_path.write_text("\n".join(lines))
        raise FileNotFoundError(msg)

    ref = pd.read_csv(phase1_meta_path)
    cols = ["example_idx", "prompt", "S", "IO", "template_id", "end_position"]
    for c in cols:
        if c not in ref.columns or c not in meta.columns:
            raise ValueError(f"Missing alignment column {c}")

    n = len(meta)
    ref_n = ref.head(n).copy()
    mismatches = []
    for c in cols:
        left = meta[c].astype(str).tolist()
        right = ref_n[c].astype(str).tolist()
        for i, (a, b) in enumerate(zip(left, right)):
            if a != b:
                mismatches.append((i, c, a, b))

    lines.append(f"- Phase-1 reference: `{phase1_meta_path}`")
    lines.append(f"- Compared examples: {n}")
    lines.append(f"- Smoke mode: {smoke}")
    if mismatches:
        lines.append(f"- **FAIL**: {len(mismatches)} mismatches")
        for i, c, a, b in mismatches[:20]:
            lines.append(f"  - example {i} col `{c}`: got `{a}` vs ref `{b}`")
        alignment_report_path.write_text("\n".join(lines) + "\n")
        raise AssertionError(
            f"Dataset does not match Phase-1 metadata ({len(mismatches)} mismatches)"
        )
    lines.append("- **PASS**: prompt/S/IO/template_id/end_position/example_idx match")
    alignment_report_path.write_text("\n".join(lines) + "\n")


def pkg_ver(name: str) -> Optional[str]:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def run_forward(model, tokens, hooks=None, names_filter=None):
    with torch.no_grad():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                if names_filter is not None:
                    return model.run_with_cache(tokens, names_filter=names_filter)
                return model(tokens), None
        if names_filter is not None:
            return model.run_with_cache(tokens, names_filter=names_filter)
        return model(tokens), None


def descriptive_for_head(
    cache_A,
    cache_B,
    layer: int,
    head: int,
    dataset,
    d_head: int,
    seed: int,
) -> List[Dict]:
    rows = []
    scores_A = cache_A[f"blocks.{layer}.attn.hook_attn_scores"]
    scores_B = cache_B[f"blocks.{layer}.attn.hook_attn_scores"]
    pat_A = cache_A[f"blocks.{layer}.attn.hook_pattern"]
    pat_B = cache_B[f"blocks.{layer}.attn.hook_pattern"]
    q_A = cache_A[f"blocks.{layer}.attn.hook_q"]
    q_B = cache_B[f"blocks.{layer}.attn.hook_q"]
    k_A = cache_A[f"blocks.{layer}.attn.hook_k"]
    k_B = cache_B[f"blocks.{layer}.attn.hook_k"]

    hname = head_name_from_pair((layer, head))
    role = role_for_head((layer, head))

    for i in range(dataset.batch_size):
        q = int(dataset.end[i].item())
        io = int(dataset.IO[i].item())

        row_A = scores_A[i, head, q, :]
        row_B = scores_B[i, head, q, :]

        # Mask preservation check
        mask_A = ~torch.isfinite(row_A)
        mask_B = ~torch.isfinite(row_B)
        assert torch.equal(mask_A, mask_B), (
            f"{hname} example {i}: causal mask differs between A and B"
        )

        t_A, b_A, valid = extract_target_and_competing(row_A, io, query_pos=q)
        t_B, b_B, _ = extract_target_and_competing(row_B, io, query_pos=q)
        # Descriptive metrics on CPU
        t_A = t_A.detach().cpu()
        t_B = t_B.detach().cpu()
        b_A = b_A.detach().cpu()
        b_B = b_B.detach().cpu()

        p_A = float(pat_A[i, head, q, io].item())
        p_B = float(pat_B[i, head, q, io].item())

        margin_A = float(attention_margin(t_A, b_A).item())
        margin_B = float(attention_margin(t_B, b_B).item())
        lse_A = float(competing_logsumexp(b_A).item())
        lse_B = float(competing_logsumexp(b_B).item())

        # Softmax-margin identity
        sig_A = float(sigmoid_margin_prob(torch.tensor(margin_A, dtype=torch.float64)).item())
        sig_B = float(sigmoid_margin_prob(torch.tensor(margin_B, dtype=torch.float64)).item())
        margin_err_A = abs(sig_A - p_A)
        margin_err_B = abs(sig_B - p_B)

        fali = fali_shapley(float(t_A), b_A, float(t_B), b_B)

        s_AA, s_BA, s_AB, s_BB, phi_q, phi_k, qk_res = qk_target_shapley(
            q_A[i, q, head, :].detach().cpu(),
            q_B[i, q, head, :].detach().cpu(),
            k_A[i, io, head, :].detach().cpu(),
            k_B[i, io, head, :].detach().cpu(),
            d_head,
        )

        rows.append(
            {
                "seed": seed,
                "example_idx": i,
                "head_name": hname,
                "layer": layer,
                "head": head,
                "head_role": role,
                "target_attn_A": p_A,
                "target_attn_B": p_B,
                "delta_attention": p_B - p_A,
                "target_score_A": float(t_A.item()),
                "target_score_B": float(t_B.item()),
                "delta_target_score": float(t_B.item()) - float(t_A.item()),
                "competing_lse_A": lse_A,
                "competing_lse_B": lse_B,
                "delta_competing_lse": lse_B - lse_A,
                "margin_A": margin_A,
                "margin_B": margin_B,
                "delta_margin": margin_B - margin_A,
                "softmax_margin_err_A": margin_err_A,
                "softmax_margin_err_B": margin_err_B,
                "fali_p00": fali.p00,
                "fali_p10": fali.p10,
                "fali_p01": fali.p01,
                "fali_p11": fali.p11,
                "fali_target": fali.phi_target,
                "fali_competing": fali.phi_competing,
                "fali_residual": fali.residual,
                "fali_total_delta": fali.total_delta_attn,
                "qk_s_AA": s_AA,
                "qk_s_BA": s_BA,
                "qk_s_AB": s_AB,
                "qk_s_BB": s_BB,
                "qk_phi_query": phi_q,
                "qk_phi_key": phi_k,
                "qk_residual": qk_res,
                "score_row_max_abs_diff": float(
                    torch.max(
                        torch.abs(row_B[valid] - row_A[valid])
                    ).item()
                ),
            }
        )
    return rows


def causal_panel_for_head(
    model,
    tokens: torch.Tensor,
    dataset,
    cache_A,
    io_s_B: torch.Tensor,
    layer: int,
    head: int,
    seed: int,
) -> List[Dict]:
    """
    Causal score hybrids + pattern + Q/K patches inside corrected B.

    Sign convention for patch effects:
      pattern_clean_patch_effect = B_IOminusS - patched_clean_pattern_IOminusS
      q/k/qk_clean_effect = B_IOminusS - patched_IOminusS
    Positive => clean patch reduces IO-S relative to B (i.e. ablation-induced
    score/pattern/Q/K change was helping IO-S).
    """
    hname = head_name_from_pair((layer, head))
    role = role_for_head((layer, head))
    end = dataset.end
    io = dataset.IO
    scores_A = cache_A[f"blocks.{layer}.attn.hook_attn_scores"]
    pattern_A = cache_A[f"blocks.{layer}.attn.hook_pattern"]
    q_A = cache_A[f"blocks.{layer}.attn.hook_q"]
    k_A = cache_A[f"blocks.{layer}.attn.hook_k"]

    def ios(hooks):
        logits, _ = run_forward(model, tokens, hooks=hooks, names_filter=None)
        return compute_logit_diff(logits, dataset)

    # F00, F10, F01 (F11 = B)
    f00 = ios(
        hooks_for_score_hybrid(scores_A, layer, head, end, io, "F00")
    )
    f10 = ios(
        hooks_for_score_hybrid(scores_A, layer, head, end, io, "F10")
    )
    f01 = ios(
        hooks_for_score_hybrid(scores_A, layer, head, end, io, "F01")
    )
    f11 = io_s_B

    pattern_patched = ios(
        hooks_for_pattern_patch(pattern_A, layer, head, end)
    )
    q_patched = ios(hooks_for_q_patch(q_A, layer, head, end))
    k_patched = ios(hooks_for_k_patch(k_A, layer, head, end))
    qk_patched = ios(hooks_for_qk_patch(q_A, k_A, layer, head, end))

    rows = []
    for i in range(dataset.batch_size):
        F00 = float(f00[i].item())
        F10 = float(f10[i].item())
        F01 = float(f01[i].item())
        F11 = float(f11[i].item())
        causal_target = 0.5 * ((F10 - F00) + (F11 - F01))
        causal_competing = 0.5 * ((F01 - F00) + (F11 - F10))
        causal_residual = (causal_target + causal_competing) - (F11 - F00)

        rows.append(
            {
                "seed": seed,
                "example_idx": i,
                "head_name": hname,
                "layer": layer,
                "head": head,
                "head_role": role,
                "F00": F00,
                "F10": F10,
                "F01": F01,
                "F11": F11,
                "causal_target": causal_target,
                "causal_competing": causal_competing,
                "causal_residual": causal_residual,
                "causal_total": F11 - F00,
                "pattern_clean_patch_ios": float(pattern_patched[i].item()),
                "pattern_clean_patch_effect": F11 - float(pattern_patched[i].item()),
                "q_clean_ios": float(q_patched[i].item()),
                "q_clean_effect": F11 - float(q_patched[i].item()),
                "k_clean_ios": float(k_patched[i].item()),
                "k_clean_effect": F11 - float(k_patched[i].item()),
                "qk_clean_ios": float(qk_patched[i].item()),
                "qk_clean_effect": F11 - float(qk_patched[i].item()),
            }
        )
    return rows


def summarize_mean(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    aggs = df.groupby(group_cols, sort=False)[value_cols].mean().reset_index()
    return aggs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--output-dir",
        default="results/revision_phase2/smoke",
    )
    p.add_argument(
        "--phase1-meta",
        default="",
        help="Path to Phase-1 dataset_metadata.csv for alignment",
    )
    p.add_argument(
        "--allow-full",
        action="store_true",
        help="Explicit opt-in for N>=200 (default smoke refuses large N)",
    )
    args = p.parse_args()

    if args.num_samples >= 200 and not args.allow_full:
        raise SystemExit(
            "Refusing N>=200 without --allow-full. Phase 2A/3A stops at smoke."
        )

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    assert_safe_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    pre_snap = snapshot_protected(_REPO_ROOT)
    device = resolve_device(args.device)
    if args.require_cuda and not device.startswith("cuda"):
        raise RuntimeError(f"--require-cuda but device={device}")

    run_config = {
        "phase": "2B/3B",
        "num_samples": args.num_samples,
        "seed": args.seed,
        "device": device,
        "dtype": "float32",
        "smoke": bool(args.smoke or args.num_samples <= 8),
        "primary_name_movers": [head_name_from_pair(h) for h in PRIMARY_NAME_MOVERS],
        "descriptive_heads": [head_name_from_pair(h) for h in all_phase1_candidates()],
        "causal_heads": [head_name_from_pair(h) for h in all_phase1_candidates()],
        "reporting_panel_heads": REPORTING_PANEL_NAMES,
        "output_dir": str(out_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    t0 = time.time()
    try:
        log(f"Loading gpt2-small on {device} (FP32)...")
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_writing_weights=True,
            device=device,
            dtype="float32",
        )
        model.eval()
        if device.startswith("cuda"):
            assert next(model.parameters()).is_cuda

        log(f"Building IOI dataset N={args.num_samples} seed={args.seed}...")
        dataset = make_validated_ioi_dataset(
            model, num_samples=args.num_samples, seed=args.seed, template_style="mixed"
        )
        tokens = dataset.clean_tokens.to(device)
        assert tokens.device.type == torch.device(device).type

        meta = build_dataset_metadata(dataset)
        meta.to_csv(out_dir / "dataset_metadata.csv", index=False)

        if args.phase1_meta:
            phase1_meta = Path(args.phase1_meta)
        elif args.num_samples <= 8 and args.seed == 42:
            phase1_meta = _REPO_ROOT / "results/revision_phase1/smoke/dataset_metadata.csv"
            if not phase1_meta.exists():
                phase1_meta = (
                    _REPO_ROOT / "results/revision_phase1/modal_smoke_seed42/dataset_metadata.csv"
                )
        else:
            phase1_meta = (
                _REPO_ROOT
                / f"results/revision_phase1/full/seed_{args.seed}/dataset_metadata.csv"
            )
        align_dataset_to_phase1(
            meta,
            phase1_meta,
            out_dir / "dataset_alignment_report.md",
            smoke=bool(args.num_samples <= 8),
        )
        log(f"Dataset aligned to {phase1_meta}")

        if args.num_samples == 200:
            n_a = int((meta["template_id"] == "A").sum())
            n_b = int((meta["template_id"] == "B").sum())
            assert n_a == 100 and n_b == 100, (
                f"Expected 100 Template A and 100 Template B, got A={n_a} B={n_b}"
            )

        candidates = all_phase1_candidates()
        layers = sorted({l for l, _ in candidates})
        nfilter = names_filter_for_layers(layers)
        d_head = model.cfg.d_head

        # Pass A
        log("Forward A (clean)...")
        logits_A, cache_A = run_forward(model, tokens, names_filter=nfilter)
        io_s_A = compute_logit_diff(logits_A, dataset)

        # Pass B (corrected primary knockout)
        log("Forward B (corrected primary knockout L9H6+L9H9+L10H0)...")
        abl_hooks = build_zero_ablation_hooks(PRIMARY_NAME_MOVERS)
        logits_B, cache_B = run_forward(
            model, tokens, hooks=abl_hooks, names_filter=nfilter
        )
        io_s_B = compute_logit_diff(logits_B, dataset)

        # Descriptive FALI + baselines for all 12 heads
        log("Computing descriptive FALI for all 12 candidates...")
        desc_rows: List[Dict] = []
        for layer, head in candidates:
            desc_rows.extend(
                descriptive_for_head(
                    cache_A, cache_B, layer, head, dataset, d_head, seed=args.seed
                )
            )
        desc_df = pd.DataFrame(desc_rows)
        desc_df.to_csv(out_dir / "fali_raw.csv", index=False)
        desc_df.to_csv(out_dir / "score_baseline_raw.csv", index=False)
        desc_df[
            [
                "seed",
                "example_idx",
                "head_name",
                "head_role",
                "qk_s_AA",
                "qk_s_BA",
                "qk_s_AB",
                "qk_s_BB",
                "qk_phi_query",
                "qk_phi_key",
                "qk_residual",
            ]
        ].to_csv(out_dir / "qk_score_decomposition_raw.csv", index=False)

        sum_cols = [
            "delta_attention",
            "delta_target_score",
            "delta_competing_lse",
            "delta_margin",
            "fali_target",
            "fali_competing",
            "fali_total_delta",
            "fali_residual",
            "target_attn_A",
            "target_attn_B",
            "softmax_margin_err_A",
            "softmax_margin_err_B",
            "score_row_max_abs_diff",
        ]
        fali_sum = summarize_mean(
            desc_df, ["seed", "head_name", "head_role", "layer", "head"], sum_cols
        )
        fali_sum.to_csv(out_dir / "fali_summary.csv", index=False)
        fali_sum.to_csv(out_dir / "score_baseline_summary.csv", index=False)

        qk_sum = summarize_mean(
            desc_df,
            ["seed", "head_name", "head_role", "layer", "head"],
            ["qk_phi_query", "qk_phi_key", "qk_residual", "delta_target_score"],
        )
        qk_sum.to_csv(out_dir / "qk_score_decomposition_summary.csv", index=False)

        # Causal interventions for ALL 12 candidates
        log("Causal interventions for all 12 candidates...")
        causal_rows: List[Dict] = []
        for layer, head in candidates:
            log(f"  causal {head_name_from_pair((layer, head))}...")
            causal_rows.extend(
                causal_panel_for_head(
                    model,
                    tokens,
                    dataset,
                    cache_A,
                    io_s_B,
                    layer,
                    head,
                    seed=args.seed,
                )
            )
        causal_df = pd.DataFrame(causal_rows)
        causal_df.to_csv(out_dir / "causal_score_hybrid_raw.csv", index=False)
        causal_df.to_csv(out_dir / "pattern_patch_raw.csv", index=False)
        causal_df.to_csv(out_dir / "qk_patch_raw.csv", index=False)

        causal_sum_cols = [
            "F00",
            "F10",
            "F01",
            "F11",
            "causal_target",
            "causal_competing",
            "causal_total",
            "causal_residual",
            "pattern_clean_patch_effect",
            "q_clean_effect",
            "k_clean_effect",
            "qk_clean_effect",
        ]
        causal_sum = summarize_mean(
            causal_df, ["seed", "head_name", "head_role", "layer", "head"], causal_sum_cols
        )
        causal_sum.to_csv(out_dir / "causal_score_hybrid_summary.csv", index=False)
        causal_sum.to_csv(out_dir / "pattern_patch_summary.csv", index=False)
        causal_sum.to_csv(out_dir / "qk_patch_summary.csv", index=False)

        # Combined per-seed raw for aggregation convenience
        merged_seed = desc_df.merge(
            causal_df,
            on=["seed", "example_idx", "head_name", "layer", "head", "head_role"],
            how="inner",
            validate="one_to_one",
        )
        assert len(merged_seed) == len(desc_df) == len(causal_df)
        merged_seed.to_csv(out_dir / "phase2_combined_raw.csv", index=False)

        # Acceptance checks
        max_fali_res = float(desc_df["fali_residual"].abs().max())
        max_causal_res = float(causal_df["causal_residual"].abs().max())
        max_margin_err = float(
            max(
                desc_df["softmax_margin_err_A"].max(),
                desc_df["softmax_margin_err_B"].max(),
            )
        )
        # Finite-value NaN/Inf check (exclude columns that may legitimately be ±inf)
        finite_check_cols_desc = [
            c
            for c in desc_df.select_dtypes(include=[np.number]).columns
            if not c.startswith("competing_lse")  # LSE can be -inf if no competitors
        ]
        finite_check_cols_causal = list(
            causal_df.select_dtypes(include=[np.number]).columns
        )
        arr_d = desc_df[finite_check_cols_desc].to_numpy(dtype=np.float64)
        arr_c = causal_df[finite_check_cols_causal].to_numpy(dtype=np.float64)
        has_nan = bool(
            np.isnan(arr_d).any()
            or np.isnan(arr_c).any()
            or np.isinf(arr_d).any()
            or np.isinf(arr_c).any()
        )

        l9h7 = desc_df[desc_df["head_name"] == "L9H7"]
        l9h7_score_diff = float(l9h7["score_row_max_abs_diff"].max()) if len(l9h7) else float("nan")
        l9h7_fali = float(l9h7["fali_target"].abs().max()) if len(l9h7) else float("nan")
        l9h7_dattn = float(l9h7["delta_attention"].abs().max()) if len(l9h7) else float("nan")

        # Tolerances (FP32)
        fali_tol = 1e-6
        causal_tol = 1e-5
        margin_tol = 1e-5
        # Same-layer: expect ~0; allow tiny numerical noise
        same_layer_tol = 1e-5

        checks = {
            "no_nan_inf": not has_nan,
            "fali_residual_ok": max_fali_res <= fali_tol,
            "causal_residual_ok": max_causal_res <= causal_tol,
            "softmax_margin_ok": max_margin_err <= margin_tol,
            "all_12_descriptive": desc_df["head_name"].nunique() == 12,
            "all_12_causal": causal_df["head_name"].nunique() == 12,
            "n_rows_match": len(desc_df) == args.num_samples * 12
            and len(causal_df) == args.num_samples * 12,
            "l9h7_score_invariant": l9h7_score_diff <= same_layer_tol,
            "l9h7_fali_near_zero": l9h7_fali <= same_layer_tol,
            "l9h7_delta_attn_near_zero": l9h7_dattn <= same_layer_tol,
            "cuda_ok": (not args.require_cuda) or device.startswith("cuda"),
        }
        all_pass = all(checks.values())

        post_snap = snapshot_protected(_REPO_ROOT)
        hist_unchanged = pre_snap == post_snap
        checks["historical_untouched"] = hist_unchanged

        report = [
            "# Phase 2/3 validation report",
            "",
            f"- device: `{device}`",
            f"- N={args.num_samples}, seed={args.seed}, dtype=float32",
            f"- max |FALI residual|: {max_fali_res:.3e} (tol {fali_tol})",
            f"- max |causal Shapley residual|: {max_causal_res:.3e} (tol {causal_tol})",
            f"- max softmax-margin error: {max_margin_err:.3e} (tol {margin_tol})",
            f"- L9H7 max |score_row A-B|: {l9h7_score_diff:.3e}",
            f"- L9H7 max |fali_target|: {l9h7_fali:.3e}",
            f"- L9H7 max |delta_attention|: {l9h7_dattn:.3e}",
            f"- historical files unchanged: {hist_unchanged}",
            f"- reporting panel (subset): {REPORTING_PANEL_NAMES}",
            "",
            "## Checks",
        ]
        for k, v in checks.items():
            report.append(f"- {'PASS' if v else 'FAIL'}: `{k}`")
        report.append("")
        report.append("## Causal means (all 12 heads)")
        report.append(causal_sum.to_string(index=False))
        report.append("")
        report.append("## Descriptive summary")
        report.append(
            fali_sum[
                [
                    "head_name",
                    "head_role",
                    "delta_attention",
                    "delta_target_score",
                    "delta_competing_lse",
                    "delta_margin",
                    "fali_target",
                    "fali_competing",
                    "fali_residual",
                ]
            ].to_string(index=False)
        )
        (out_dir / "validation_report.md").write_text("\n".join(report) + "\n")

        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "torch": torch.__version__,
            "transformer_lens": pkg_ver("transformer-lens"),
            "transformers": pkg_ver("transformers"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "protected_snapshot_pre": pre_snap,
            "protected_snapshot_post": post_snap,
            "checks": checks,
            "max_fali_residual": max_fali_res,
            "max_causal_residual": max_causal_res,
            "max_softmax_margin_err": max_margin_err,
            "l9h7_score_row_max_abs_diff": l9h7_score_diff,
            "elapsed_sec": time.time() - t0,
        }
        (out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))

        if not all_pass or not hist_unchanged:
            log("ACCEPTANCE FAILED — see validation_report.md")
            for k, v in checks.items():
                if not v:
                    log(f"  FAIL: {k}")
            raise SystemExit(2)

        log("ACCEPTANCE PASSED")
        log(f"Done in {time.time() - t0:.1f}s -> {out_dir}")

    except Exception:
        log(traceback.format_exc())
        raise
    finally:
        (out_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
