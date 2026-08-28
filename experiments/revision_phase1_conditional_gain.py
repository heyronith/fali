"""
Revision Phase 1: Conditional functional gain of IOI candidate heads
after corrected primary Name Mover knockout.

Smoke / full runs write ONLY under results/revision_phase1/ (never historical results/).
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

# Ensure src/ is importable whether launched from repo root or elsewhere
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from revision_config import (  # noqa: E402
    BACKUP_NAME_MOVERS,
    EXTRA_CONTROLS,
    HISTORICAL_PRIMARY_NAME_MOVERS,
    NEGATIVE_NAME_MOVERS,
    PRIMARY_NAME_MOVERS,
    all_phase1_candidates,
    head_name_from_pair,
    role_for_head,
)
from revision_dataset import (  # noqa: E402
    build_dataset_metadata,
    make_validated_ioi_dataset,
    template_id_for_index,
)
from revision_interventions import (  # noqa: E402
    build_zero_ablation_hooks,
    group_heads_by_layer,
    hook_point_names,
)
from revision_stats import (  # noqa: E402
    pearson_spearman,
    stratified_bootstrap_ci,
    summarize_array,
)

FORBIDDEN_RESULT_PREFIXES = (
    "results/raw",
    "results/summary",
    "results/attention_results.json",
)

HISTORICAL_HASH_TARGETS = [
    "results/summary/attention_results.json",
    "results/summary/ablation_controls_summary.json",
    "results/summary/output_mediation_summary.json",
    "results/raw/fali_decomposition_raw.csv",
]


def resolve_device(requested: Optional[str]) -> str:
    """
    Resolve compute device.

    Prefer CUDA when available. Avoid defaulting to MPS because TransformerLens
    currently warns that MPS may produce silently incorrect results
    (TransformerLens issue #1178). Explicit `--device mps` remains allowed.
    """
    if requested and requested.lower() != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def assert_safe_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    repo = _REPO_ROOT.resolve()
    try:
        rel = resolved.relative_to(repo)
        rel_str = str(rel).replace("\\", "/")
    except ValueError:
        # Outside repo is allowed, but still must not look like historical paths
        rel_str = str(resolved)

    for bad in FORBIDDEN_RESULT_PREFIXES:
        if rel_str == bad or rel_str.startswith(bad + "/"):
            raise ValueError(
                f"Refusing to write into historical results path: {rel_str}"
            )
    if "results/revision_phase1" not in rel_str and not str(resolved).endswith(
        "revision_phase1"
    ):
        # Soft guard: prefer revision_phase1, but allow explicit override only if path
        # does not collide with historical dirs (already checked).
        pass


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_historical_artifacts(repo_root: Path) -> Dict[str, Dict[str, Optional[str]]]:
    snap = {}
    for rel in HISTORICAL_HASH_TARGETS:
        p = repo_root / rel
        st = p.stat() if p.exists() else None
        snap[rel] = {
            "sha256": file_sha256(p),
            "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            if st
            else None,
            "size": st.st_size if st else None,
        }
    return snap


def compute_logit_diff(logits: torch.Tensor, dataset) -> torch.Tensor:
    batch = dataset.batch_size
    device = logits.device
    end_idx = dataset.end.to(device)
    batch_idx = torch.arange(batch, device=device)
    logits_at_end = logits[batch_idx, end_idx]
    io_logits = logits_at_end[batch_idx, dataset.io_token_ids.to(device)]
    s_logits = logits_at_end[batch_idx, dataset.s_token_ids.to(device)]
    return io_logits - s_logits


def extract_head_attn_to_io(cache, layer: int, head: int, dataset) -> torch.Tensor:
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    device = pattern.device
    batch_idx = torch.arange(dataset.batch_size, device=device)
    end_idx = dataset.end.to(device)
    io_idx = dataset.IO.to(device)
    return pattern[batch_idx, head, end_idx, io_idx]


def run_with_optional_hooks(model, tokens, hooks=None, names_filter=None):
    with torch.no_grad():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                if names_filter is None:
                    logits = model(tokens)
                    return logits, None
                return model.run_with_cache(tokens, names_filter=names_filter)
        if names_filter is None:
            return model(tokens), None
        return model.run_with_cache(tokens, names_filter=names_filter)


def environment_dict(
    device: str,
    model: Optional[torch.nn.Module] = None,
    tokens: Optional[torch.Tensor] = None,
) -> Dict:
    def safe_ver(name: str) -> Optional[str]:
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return None

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    cuda_runtime_version = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
        try:
            cuda_runtime_version = torch.version.cuda
        except Exception:
            cuda_runtime_version = None

    model_param_device = None
    model_dtype = None
    if model is not None:
        p = next(model.parameters())
        model_param_device = str(p.device)
        model_dtype = str(p.dtype)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_runtime_version": cuda_runtime_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "model_parameter_device": model_param_device,
        "token_device": str(tokens.device) if tokens is not None else None,
        "torch_dtype": model_dtype,
        "transformer_lens": safe_ver("transformer-lens"),
        "transformers": safe_ver("transformers"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "mixed_precision": False,
        "dtype_policy": "fp32",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Revision Phase-1 conditional gain experiment")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/revision_phase1/smoke",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="Assert CUDA availability and that model/tokens live on CUDA.",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=None,
        help="Bootstrap resamples (default 200 smoke / 10000 full)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (_REPO_ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    assert_safe_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_bootstrap = args.n_bootstrap
    if n_bootstrap is None:
        n_bootstrap = 200 if args.smoke else 10000

    device = resolve_device(args.device)
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    pre_hashes = snapshot_historical_artifacts(_REPO_ROOT)
    start = time.time()
    log("=" * 72)
    log("REVISION PHASE-1 CONDITIONAL GAIN")
    log("=" * 72)
    log(f"output_dir={output_dir}")
    log(f"num_samples={args.num_samples} seed={args.seed} smoke={args.smoke}")
    log(f"device={device} n_bootstrap={n_bootstrap}")

    run_config = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "device": device,
        "smoke": bool(args.smoke),
        "n_bootstrap": n_bootstrap,
        "template_style": "mixed",
        "model": "gpt2-small",
        "center_writing_weights": True,
        "PRIMARY_NAME_MOVERS": [list(h) for h in PRIMARY_NAME_MOVERS],
        "HISTORICAL_PRIMARY_NAME_MOVERS": [list(h) for h in HISTORICAL_PRIMARY_NAME_MOVERS],
        "BACKUP_NAME_MOVERS": [list(h) for h in BACKUP_NAME_MOVERS],
        "NEGATIVE_NAME_MOVERS": [list(h) for h in NEGATIVE_NAME_MOVERS],
        "EXTRA_CONTROLS": [list(h) for h in EXTRA_CONTROLS],
        "candidates": [list(h) for h in all_phase1_candidates()],
    }
    run_config["require_cuda"] = bool(args.require_cuda)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    validation_sections: List[str] = ["# Phase-1 smoke/validation report", ""]

    try:
        if args.require_cuda:
            assert torch.cuda.is_available(), "require-cuda set but CUDA unavailable"
            assert device.startswith("cuda"), f"require-cuda set but device={device}"

        log("Loading gpt2-small (FP32, no mixed precision)...")
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_writing_weights=True,
            device=device,
            dtype="float32",
        )
        model.eval()
        log("Model loaded.")

        dataset = make_validated_ioi_dataset(
            model,
            num_samples=args.num_samples,
            seed=args.seed,
            template_style="mixed",
        )
        # Explicit CUDA-safe placement: always move tokens onto the model device.
        tokens = dataset.clean_tokens.to(device)
        model_param_device = next(model.parameters()).device
        log(f"model_param_device={model_param_device} token_device={tokens.device}")

        if args.require_cuda or device.startswith("cuda"):
            assert torch.cuda.is_available()
            assert next(model.parameters()).device.type == "cuda"
            assert tokens.device.type == "cuda"

        env = environment_dict(device, model=model, tokens=tokens)
        with open(output_dir / "environment.json", "w") as f:
            json.dump(env, f, indent=2)

        meta_df = build_dataset_metadata(dataset)
        meta_df.to_csv(output_dir / "dataset_metadata.csv", index=False)
        log(f"Dataset validated: N={dataset.batch_size}, seq_len={dataset.seq_len}")
        validation_sections.append("## Dataset validation")
        validation_sections.append("- All examples passed S×2 / IO×1 / end==' to' checks.")
        validation_sections.append(
            f"- tokens device={tokens.device}; model device={model_param_device}"
        )
        validation_sections.append("")

        candidates = all_phase1_candidates()
        assert len(candidates) == 12, f"Expected 12 candidates, got {len(candidates)}"

        # Pattern hooks needed for all candidate layers
        candidate_layers = sorted({l for l, _ in candidates})
        pattern_filter = [f"blocks.{l}.attn.hook_pattern" for l in candidate_layers]

        # ---- Baseline passes with attention caches ----
        log("Running Pass A (clean)...")
        logits_A, cache_A = run_with_optional_hooks(
            model, tokens, hooks=None, names_filter=pattern_filter
        )
        A = compute_logit_diff(logits_A, dataset).detach().cpu().numpy()

        hooks_B_old = build_zero_ablation_hooks(HISTORICAL_PRIMARY_NAME_MOVERS)
        assert len(hooks_B_old) == len(group_heads_by_layer(HISTORICAL_PRIMARY_NAME_MOVERS))
        log(f"B_old hooks: {hook_point_names(hooks_B_old)}")
        logits_Bold, cache_Bold = run_with_optional_hooks(
            model, tokens, hooks=hooks_B_old, names_filter=pattern_filter
        )
        B_old = compute_logit_diff(logits_Bold, dataset).detach().cpu().numpy()

        hooks_B = build_zero_ablation_hooks(PRIMARY_NAME_MOVERS)
        assert len(hooks_B) == 2, "Corrected primary set spans layers 9 and 10 only"
        assert hook_point_names(hooks_B) == [
            "blocks.9.attn.hook_z",
            "blocks.10.attn.hook_z",
        ]
        log(f"B hooks: {hook_point_names(hooks_B)}")
        logits_B, cache_B = run_with_optional_hooks(
            model, tokens, hooks=hooks_B, names_filter=pattern_filter
        )
        B = compute_logit_diff(logits_B, dataset).detach().cpu().numpy()

        assert not np.allclose(B, B_old), "Corrected B must differ numerically from B_old"
        validation_sections.append("## Primary knockout")
        validation_sections.append(
            f"- mean A={A.mean():.6f}, B_old={B_old.mean():.6f}, B={B.mean():.6f}"
        )
        validation_sections.append("- Corrected B differs from historical B_old.")
        validation_sections.append(
            f"- Corrected primary hooks (one per layer): {hook_point_names(hooks_B)}"
        )
        validation_sections.append("")

        # Primary-only ablations for knockout-strength table
        primary_rows = []
        primary_conditions = {
            "A_clean": [],
            "L9H6_only": [(9, 6)],
            "L9H9_only": [(9, 9)],
            "L10H0_only": [(10, 0)],
            "B_old_historical": HISTORICAL_PRIMARY_NAME_MOVERS,
            "B_corrected": PRIMARY_NAME_MOVERS,
        }
        # reuse A / B_old / B where possible
        condition_means = {
            "A_clean": A,
            "B_old_historical": B_old,
            "B_corrected": B,
        }
        for name, heads in [
            ("L9H6_only", [(9, 6)]),
            ("L9H9_only", [(9, 9)]),
            ("L10H0_only", [(10, 0)]),
        ]:
            hooks = build_zero_ablation_hooks(heads)
            logits, _ = run_with_optional_hooks(model, tokens, hooks=hooks)
            vals = compute_logit_diff(logits, dataset).detach().cpu().numpy()
            condition_means[name] = vals

        for name, vals in condition_means.items():
            primary_rows.append(
                {
                    "condition": name,
                    "mean_io_s": float(np.mean(vals)),
                    "std_io_s": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "mean_abs_degradation_from_clean": float(np.mean(A - vals)),
                }
            )
        primary_df = pd.DataFrame(primary_rows)
        primary_df.to_csv(output_dir / "primary_ablation_summary.csv", index=False)
        log("Saved primary_ablation_summary.csv")

        # Precompute attention under A / B_old / B for each candidate
        attn_A: Dict[Tuple[int, int], np.ndarray] = {}
        attn_Bold: Dict[Tuple[int, int], np.ndarray] = {}
        attn_B: Dict[Tuple[int, int], np.ndarray] = {}
        for layer, head in candidates:
            attn_A[(layer, head)] = (
                extract_head_attn_to_io(cache_A, layer, head, dataset).detach().cpu().numpy()
            )
            attn_Bold[(layer, head)] = (
                extract_head_attn_to_io(cache_Bold, layer, head, dataset)
                .detach()
                .cpu()
                .numpy()
            )
            attn_B[(layer, head)] = (
                extract_head_attn_to_io(cache_B, layer, head, dataset).detach().cpu().numpy()
            )
            for arr, label in [
                (attn_A[(layer, head)], "A"),
                (attn_Bold[(layer, head)], "B_old"),
                (attn_B[(layer, head)], "B"),
            ]:
                assert np.isfinite(arr).all(), f"Non-finite attention for L{layer}H{head} {label}"
                assert (arr >= -1e-6).all() and (arr <= 1.0 + 1e-6).all(), (
                    f"Attention out of [0,1] for L{layer}H{head} {label}"
                )

        template_ids = np.array(
            [template_id_for_index(i, dataset.template_style) for i in range(dataset.batch_size)]
        )

        raw_rows = []
        summary_rows = []
        bootstrap_rows = []

        log(f"Evaluating {len(candidates)} candidates (D_h and C_h)...")
        for layer, head in candidates:
            hname = head_name_from_pair((layer, head))
            role = role_for_head((layer, head))
            log(f"  Candidate {hname} ({role})")

            hooks_D = build_zero_ablation_hooks([(layer, head)])
            logits_D, _ = run_with_optional_hooks(
                model, tokens, hooks=hooks_D
            )
            D = compute_logit_diff(logits_D, dataset).detach().cpu().numpy()

            hooks_C = build_zero_ablation_hooks(list(PRIMARY_NAME_MOVERS) + [(layer, head)])
            # Exactly one hook per distinct layer
            assert len(hooks_C) == len(
                group_heads_by_layer(list(PRIMARY_NAME_MOVERS) + [(layer, head)])
            )
            logits_C, _ = run_with_optional_hooks(
                model, tokens, hooks=hooks_C
            )
            C = compute_logit_diff(logits_C, dataset).detach().cpu().numpy()

            intact = A - D
            conditional = B - C
            gain = conditional - intact

            hist_spike = attn_Bold[(layer, head)] - attn_A[(layer, head)]
            corr_spike = attn_B[(layer, head)] - attn_A[(layer, head)]

            assert np.isfinite(intact).all()
            assert np.isfinite(conditional).all()
            assert np.isfinite(gain).all()
            # Algebraic identity check
            assert np.allclose(gain, (B - C) - (A - D))

            for i in range(dataset.batch_size):
                raw_rows.append(
                    {
                        "example_idx": i,
                        "template_id": template_ids[i],
                        "head_name": hname,
                        "layer": layer,
                        "head": head,
                        "head_role": role,
                        "A": float(A[i]),
                        "B_old": float(B_old[i]),
                        "B": float(B[i]),
                        "D_h": float(D[i]),
                        "C_h": float(C[i]),
                        "intact_effect": float(intact[i]),
                        "conditional_effect": float(conditional[i]),
                        "conditional_gain": float(gain[i]),
                        "clean_attn": float(attn_A[(layer, head)][i]),
                        "historical_primary_attn": float(attn_Bold[(layer, head)][i]),
                        "corrected_primary_attn": float(attn_B[(layer, head)][i]),
                        "historical_attention_spike": float(hist_spike[i]),
                        "corrected_attention_spike": float(corr_spike[i]),
                    }
                )

            mean_gain, ci_low, ci_high = stratified_bootstrap_ci(
                gain,
                template_ids,
                n_bootstrap=n_bootstrap,
                seed=args.seed,
            )
            sum_intact = summarize_array(intact)
            sum_cond = summarize_array(conditional)
            sum_gain = summarize_array(gain)

            template_A_mask = template_ids == "A"
            template_B_mask = template_ids == "B"
            tmpl_A_gain = (
                float(np.mean(gain[template_A_mask])) if template_A_mask.any() else float("nan")
            )
            tmpl_B_gain = (
                float(np.mean(gain[template_B_mask])) if template_B_mask.any() else float("nan")
            )

            summary_rows.append(
                {
                    "head_name": hname,
                    "head_role": role,
                    "mean_clean_attn": float(np.mean(attn_A[(layer, head)])),
                    "mean_historical_primary_attn": float(np.mean(attn_Bold[(layer, head)])),
                    "mean_corrected_primary_attn": float(np.mean(attn_B[(layer, head)])),
                    "mean_historical_attention_spike": float(np.mean(hist_spike)),
                    "mean_corrected_attention_spike": float(np.mean(corr_spike)),
                    "mean_intact_effect": sum_intact["mean"],
                    "mean_conditional_effect": sum_cond["mean"],
                    "mean_conditional_gain": sum_gain["mean"],
                    "std_conditional_gain": sum_gain["std"],
                    "median_conditional_gain": sum_gain["median"],
                    "conditional_gain_ci_low": ci_low,
                    "conditional_gain_ci_high": ci_high,
                    "conditional_gain_ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                    "fraction_positive_conditional_gain": sum_gain["fraction_positive"],
                    "template_A_conditional_gain": tmpl_A_gain,
                    "template_B_conditional_gain": tmpl_B_gain,
                    "mean_A": float(np.mean(A)),
                    "mean_B_old": float(np.mean(B_old)),
                    "mean_B": float(np.mean(B)),
                    "mean_D_h": float(np.mean(D)),
                    "mean_C_h": float(np.mean(C)),
                }
            )
            bootstrap_rows.append(
                {
                    "head_name": hname,
                    "metric": "conditional_gain",
                    "mean": mean_gain,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_bootstrap": n_bootstrap,
                    "seed": args.seed,
                    "stratified_by": "template_id",
                    "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                }
            )

        raw_df = pd.DataFrame(raw_rows)
        summary_df = pd.DataFrame(summary_rows)
        bootstrap_df = pd.DataFrame(bootstrap_rows)

        # Verify summary means match per-example formulas
        for _, row in summary_df.iterrows():
            sub = raw_df[raw_df["head_name"] == row["head_name"]]
            assert np.isclose(row["mean_conditional_gain"], sub["conditional_gain"].mean())
            assert np.isclose(
                row["mean_conditional_gain"],
                ((sub["B"] - sub["C_h"]) - (sub["A"] - sub["D_h"])).mean(),
            )

        raw_df.to_csv(output_dir / "candidate_raw.csv", index=False)
        summary_df.to_csv(output_dir / "candidate_summary.csv", index=False)
        bootstrap_df.to_csv(output_dir / "bootstrap_summary.csv", index=False)

        # Correlations
        corr_rows = []
        # Across candidates (means)
        across = pearson_spearman(
            summary_df["mean_corrected_attention_spike"].tolist(),
            summary_df["mean_conditional_gain"].tolist(),
        )
        corr_rows.append(
            {
                "level": "across_candidates",
                "head_name": "ALL",
                "x": "mean_corrected_attention_spike",
                "y": "mean_conditional_gain",
                **across,
            }
        )
        # Within candidate
        for hname in summary_df["head_name"].tolist():
            sub = raw_df[raw_df["head_name"] == hname]
            within = pearson_spearman(
                sub["corrected_attention_spike"].tolist(),
                sub["conditional_gain"].tolist(),
            )
            corr_rows.append(
                {
                    "level": "within_candidate",
                    "head_name": hname,
                    "x": "corrected_attention_spike",
                    "y": "conditional_gain",
                    **within,
                }
            )
        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv(output_dir / "correlation_summary.csv", index=False)

        post_hashes = snapshot_historical_artifacts(_REPO_ROOT)
        unchanged = True
        for k in HISTORICAL_HASH_TARGETS:
            if pre_hashes[k]["sha256"] != post_hashes[k]["sha256"]:
                unchanged = False
                log(f"WARNING: historical file changed: {k}")

        validation_sections.append("## Candidate execution")
        validation_sections.append(f"- Evaluated {len(candidates)} candidates (D_h and C_h).")
        validation_sections.append(
            "- Per-example identity `conditional_gain = (B-C)-(A-D)` verified."
        )
        validation_sections.append("- Attention values finite and in [0,1] (tol 1e-6).")
        validation_sections.append("")
        validation_sections.append("## Historical result integrity")
        validation_sections.append(
            f"- Historical headline artifacts unchanged: **{unchanged}**"
        )
        for k in HISTORICAL_HASH_TARGETS:
            validation_sections.append(
                f"  - `{k}` sha256={post_hashes[k]['sha256']}"
            )
        validation_sections.append("")
        validation_sections.append("## Smoke means (not for scientific interpretation)")
        validation_sections.append(f"- A mean IO−S: {A.mean():.6f}")
        validation_sections.append(f"- B_old mean IO−S: {B_old.mean():.6f}")
        validation_sections.append(f"- B mean IO−S: {B.mean():.6f}")
        validation_sections.append("")
        validation_sections.append("## Candidate summary (smoke)")
        validation_sections.append("```")
        validation_sections.append(
            summary_df[
                [
                    "head_name",
                    "head_role",
                    "mean_corrected_attention_spike",
                    "mean_intact_effect",
                    "mean_conditional_effect",
                    "mean_conditional_gain",
                ]
            ].to_string(index=False)
        )
        validation_sections.append("```")
        validation_sections.append("")
        validation_sections.append(
            f"- Runtime seconds: {time.time() - start:.2f}"
        )
        validation_sections.append(
            "- N=8 smoke only; do not interpret scientifically."
        )

        with open(output_dir / "validation_report.md", "w") as f:
            f.write("\n".join(validation_sections) + "\n")

        with open(output_dir / "historical_artifact_hashes.json", "w") as f:
            json.dump({"pre": pre_hashes, "post": post_hashes, "unchanged": unchanged}, f, indent=2)

        elapsed = time.time() - start
        log(f"Done in {elapsed:.2f}s. Outputs in {output_dir}")
        with open(output_dir / "run_log.txt", "w") as f:
            f.write("\n".join(log_lines) + "\n")

        required = [
            "run_config.json",
            "environment.json",
            "dataset_metadata.csv",
            "primary_ablation_summary.csv",
            "candidate_summary.csv",
            "candidate_raw.csv",
            "bootstrap_summary.csv",
            "correlation_summary.csv",
            "run_log.txt",
            "validation_report.md",
        ]
        missing = [r for r in required if not (output_dir / r).exists()]
        if missing:
            raise RuntimeError(f"Missing required outputs: {missing}")

        return 0

    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())
        with open(output_dir / "run_log.txt", "w") as f:
            f.write("\n".join(log_lines) + "\n")
        raise


if __name__ == "__main__":
    sys.exit(main())
