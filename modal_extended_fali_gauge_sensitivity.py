"""
Modal GPU runner for preregistered FALI softmax-gauge sensitivity.

Runs:
  - native smoke (EF-4 N=small subset via dedicated smoke) then rowmean smoke
  - EF-4 rowmean: L10H2/L10H10 × seeds 42,1,8,22
  - EF-6 rowmean confirmation: L8H2 × seeds 1,8,22
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import modal

APP_NAME = "fali-gauge-sensitivity"
REPO_ROOT = Path(__file__).resolve().parent
EF4_SEEDS = [42, 1, 8, 22]
EF6_SEEDS = [1, 8, 22]
GAUGE_PREREG = "e60f81acfff8fc9dd4f924ebb1c087ec89734c05"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .pip_install(
        "transformer-lens==3.2.1",
        "transformers==4.57.6",
        "einops>=0.7.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "accelerate>=0.26.0",
        "huggingface-hub>=0.24.0,<1.0",
        "jaxtyping>=0.2.0",
        "typeguard>=4.0.0",
        "better-abc>=0.0.3",
        "fancy-einsum>=0.0.3",
        "rich>=12.6.0",
        "beartype>=0.14.1",
    )
    .env({"HF_HOME": "/tmp/hf_cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/project/src")
    .add_local_dir(str(REPO_ROOT / "experiments"), remote_path="/root/project/experiments")
    .add_local_file(
        str(REPO_ROOT / "preregistration/gauge_sensitivity_rowmean.md"),
        remote_path="/root/project/preregistration/gauge_sensitivity_rowmean.md",
    )
    .add_local_file(
        str(REPO_ROOT / "results/extended_fali/ef6_pythia_induction/aggregate/selection_lock.json"),
        remote_path="/root/project/results/extended_fali/ef6_pythia_induction/aggregate/selection_lock.json",
    )
    .add_local_file(
        str(REPO_ROOT / "results/extended_fali/ef6_pythia_induction/model_revision.json"),
        remote_path="/root/project/results/extended_fali/ef6_pythia_induction/model_revision.json",
    )
)

for seed in EF4_SEEDS:
    image = image.add_local_file(
        str(REPO_ROOT / f"results/revision_phase4/full/seed_{seed}/dataset_metadata.csv"),
        remote_path=f"/root/project/results/revision_phase4/full/seed_{seed}/dataset_metadata.csv",
    ).add_local_file(
        str(REPO_ROOT / f"results/revision_phase4/full/seed_{seed}/factorial_raw.csv"),
        remote_path=f"/root/project/results/revision_phase4/full/seed_{seed}/factorial_raw.csv",
    )

app = modal.App(APP_NAME, image=image)
GPU_CONFIG = "L4"


def _zip_dir(path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(path.rglob("*")):
            if fp.is_file():
                zf.write(fp, arcname=str(fp.relative_to(path)))
    return buf.getvalue()


def _extract(payload: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        zf.extractall(dest)


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 10, memory=32768)
def run_ef4_gauge_seed(seed: int, score_gauge: str = "row_mean") -> bytes:
    import os
    from pathlib import Path

    project = Path("/root/project")
    out_dir = project / f"results/extended_fali/gauge_sensitivity/ef4_rowmean/seed_{seed}"
    if score_gauge == "native":
        out_dir = project / f"results/extended_fali/gauge_sensitivity/ef4_native_smoke/seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(project / "experiments/extended_fali_gauge_ef4.py"),
        "--seed",
        str(seed),
        "--score-gauge",
        score_gauge,
        "--device",
        "cuda",
        "--require-cuda",
        "--output-dir",
        str(out_dir),
        "--gauge-preregistration-sha",
        GAUGE_PREREG,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(project))
    return _zip_dir(out_dir)


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 8, memory=32768)
def run_ef6_gauge_seed(seed: int, score_gauge: str = "row_mean") -> bytes:
    import os
    from pathlib import Path

    project = Path("/root/project")
    out_dir = project / f"results/extended_fali/gauge_sensitivity/ef6_rowmean/seed_{seed}"
    if score_gauge == "native":
        out_dir = project / f"results/extended_fali/gauge_sensitivity/ef6_native_smoke/seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    lock = project / "results/extended_fali/ef6_pythia_induction/aggregate/selection_lock.json"
    cmd = [
        sys.executable,
        str(project / "experiments/extended_fali_gauge_ef6.py"),
        "--seed",
        str(seed),
        "--score-gauge",
        score_gauge,
        "--device",
        "cuda",
        "--require-cuda",
        "--selection-lock",
        str(lock),
        "--output-dir",
        str(out_dir),
        "--gauge-preregistration-sha",
        GAUGE_PREREG,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(project))
    return _zip_dir(out_dir)


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 2, memory=32768)
def run_gauge_smoke() -> bytes:
    """Smoke: native vs row_mean on EF-6 induction (N=8) for endpoint invariants."""
    import os
    import sys
    from pathlib import Path

    import pandas as pd
    import torch

    project = Path("/root/project")
    os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(project / "src"))

    from extended_fali_ef6_runner import (
        MODEL_ID,
        SMOKE_CANDIDATE,
        SMOKE_PRIMARY,
        _write_env,
        load_pythia,
        resolve_model_revision,
        run_extended_fali_confirmation,
        write_cell_raw,
    )
    from extended_fali_induction import generate_induction_metadata, metadata_to_tensors, save_dataset_artifacts

    # Deterministic CUDA for smoke reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results = {}
    ef6_root = project / "results/extended_fali/ef6_pythia_induction"
    revision = resolve_model_revision(project, ef6_root)
    model = load_pythia("cuda", revision)

    for gauge in ("native", "row_mean"):
        out = project / f"results/extended_fali/gauge_sensitivity/smoke_ef6/{gauge}"
        out.mkdir(parents=True, exist_ok=True)
        meta = generate_induction_metadata(seed=0, smoke_deterministic=True, smoke_n=8)
        dataset = metadata_to_tensors(meta)
        save_dataset_artifacts(dataset, out)
        cell_rows, attr_rows, val = run_extended_fali_confirmation(
            model,
            dataset,
            "cuda",
            SMOKE_PRIMARY,
            SMOKE_CANDIDATE,
            seed=0,
            score_gauge=gauge,
            save_score_artifacts=(gauge == "row_mean"),
            score_artifact_path=(out / "score_gauge_artifacts.pt") if gauge == "row_mean" else None,
        )
        write_cell_raw(out, cell_rows, compress=False)
        pd.DataFrame(attr_rows).to_csv(out / "attribution_raw.csv", index=False)
        (out / "validation_report.json").write_text(json.dumps(val, indent=2) + "\n")
        (out / "invariant_report.json").write_text(json.dumps(val, indent=2) + "\n")
        (out / "preregistration_sha.txt").write_text(GAUGE_PREREG + "\n")
        (out / "run_config.json").write_text(
            json.dumps(
                {
                    "phase": "gauge_smoke",
                    "score_gauge": gauge,
                    "model": MODEL_ID,
                    "model_revision": revision,
                    "preregistration_sha": GAUGE_PREREG,
                },
                indent=2,
            )
            + "\n"
        )
        _write_env(out, "cuda")
        results[gauge] = {
            "overall_pass": val.get("overall_pass"),
            "max_gauge_endpoint_pattern_error": val.get("max_gauge_endpoint_pattern_error"),
            "max_gauge_endpoint_z_error": val.get("max_gauge_endpoint_z_error"),
            "max_endpoint_error": val.get("max_endpoint_error"),
            "max_owen_residual": val.get("max_owen_residual"),
            "checks": val.get("checks"),
        }
        # Do not hard-fail yet; compare gauges below. Endpoint self-check can be noisy for smoke heads.
        print(json.dumps({"gauge": gauge, **results[gauge]}, indent=2), flush=True)

    n = pd.read_csv(project / "results/extended_fali/gauge_sensitivity/smoke_ef6/native/attribution_raw.csv")
    r = pd.read_csv(project / "results/extended_fali/gauge_sensitivity/smoke_ef6/row_mean/attribution_raw.csv")
    m = n.merge(r, on=["example_idx", "query_offset"], suffixes=("_n", "_r"))
    max_u0 = float((m["U00000_n"] - m["U00000_r"]).abs().max())
    max_u1 = float((m["U11111_n"] - m["U11111_r"]).abs().max())
    max_gain = float((m["conditional_gain_n"] - m["conditional_gain_r"]).abs().max())
    max_score = float((m["Phi_SCORE_n"] - m["Phi_SCORE_r"]).abs().max())
    max_intact = float((m["intact_effect_n"] - m["intact_effect_r"]).abs().max())
    # Mixed-cell sensitivity diagnostic
    max_phi_t = float((m["phi_T_n"] - m["phi_T_r"]).abs().max())
    smoke_summary = {
        "results": results,
        "endpoint_max_abs_U00000": max_u0,
        "endpoint_max_abs_U11111": max_u1,
        "endpoint_max_abs_gain": max_gain,
        "endpoint_max_abs_intact": max_intact,
        "coarse_score_max_abs": max_score,
        "max_abs_phi_T_delta": max_phi_t,
        "endpoint_ok": max(max_u0, max_u1, max_gain) <= 1e-5,
        "coarse_score_ok": max_score <= 1e-5,
        "intact_ok": max_intact <= 1e-5,
    }
    out_root = project / "results/extended_fali/gauge_sensitivity/smoke_ef6"
    (out_root / "smoke_summary.json").write_text(json.dumps(smoke_summary, indent=2) + "\n")
    print(json.dumps(smoke_summary, indent=2), flush=True)
    if not smoke_summary["endpoint_ok"]:
        raise RuntimeError(f"Endpoint invariance failed: {smoke_summary}")
    if not smoke_summary["coarse_score_ok"]:
        raise RuntimeError(f"Coarse SCORE invariance failed: {smoke_summary}")
    return _zip_dir(out_root)


@app.function(gpu=GPU_CONFIG, timeout=60 * 30, memory=32768)
def debug_endpoint_gauge() -> str:
    """Single-forward debug: compare endpoint z/U native vs row_mean on same caches."""
    import os
    import sys
    from pathlib import Path

    import torch

    project = Path("/root/project")
    os.environ["PYTHONPATH"] = str(project / "src")
    sys.path.insert(0, str(project / "src"))

    from extended_fali_ef6_runner import (
        SMOKE_CANDIDATE,
        SMOKE_PRIMARY,
        build_ef6_factorial_hooks,
        load_pythia,
        resolve_model_revision,
        target_logprob,
    )
    from extended_fali_gpt2 import cache_names_filter, extract_head_cache, max_z_error_on_prompt
    from extended_fali_induction import generate_induction_metadata, metadata_to_tensors
    from extended_fali_interventions import (
        canonicalize_score_row,
        masked_softmax,
        reconstruct_z_from_scores_values,
        score_pattern_z_pipeline,
    )
    from revision_interventions import build_zero_ablation_hooks

    ef6_root = project / "results/extended_fali/ef6_pythia_induction"
    revision = resolve_model_revision(project, ef6_root)
    model = load_pythia("cuda", revision)
    meta = generate_induction_metadata(seed=0, smoke_deterministic=True, smoke_n=8)
    dataset = metadata_to_tensors(meta)
    partition = dataset.partition("cuda")
    tokens = dataset.repeated_tokens.to("cuda")
    primary, candidate = SMOKE_PRIMARY, SMOKE_CANDIDATE
    c_layer, c_head = candidate
    names = cache_names_filter(sorted({primary[0], c_layer}))
    primary_hooks = build_zero_ablation_hooks([primary])
    with torch.no_grad():
        _, cache_a = model.run_with_cache(tokens, names_filter=names)
    with model.hooks(fwd_hooks=primary_hooks):
        with torch.no_grad():
            _, cache_b = model.run_with_cache(tokens, names_filter=names)
    sa, pa, va, za = extract_head_cache(cache_a, c_layer, c_head)
    sb, pb, vb, zb = extract_head_cache(cache_b, c_layer, c_head)
    sa_c = canonicalize_score_row(sa, partition.valid_mask, "row_mean")
    sb_c = canonicalize_score_row(sb, partition.valid_mask, "row_mean")

    p_n = masked_softmax(sa.masked_fill(~partition.valid_mask, float("-inf")), partition.valid_mask)
    p_c = masked_softmax(sa_c.masked_fill(~partition.valid_mask, float("-inf")), partition.valid_mask)
    lines = [
        f"pattern max |n-c|={float((p_n-p_c).abs().max())}",
        f"z recon max |n-c|={max_z_error_on_prompt(reconstruct_z_from_scores_values(p_n,va), reconstruct_z_from_scores_values(p_c,va), dataset.query_pos)}",
    ]
    for state in [(0, 0, 0, 0, 0), (1, 1, 1, 1, 1)]:
        _, _, zn = score_pattern_z_pipeline(sa, sb, va, vb, partition, state)
        _, _, zc = score_pattern_z_pipeline(
            sa_c,
            sb_c,
            va,
            vb,
            partition,
            state,
            scores_a_native=sa,
            scores_b_native=sb,
        )
        lines.append(f"state {state} pipeline z max={float((zn-zc).abs().max())}")
        # utilities
        for label, z in (("native", zn), ("centered", zc)):
            hooks = build_ef6_factorial_hooks(
                candidate, primary, context_b=bool(state[4]), z_head=z, zero_candidate=False
            )
            hooks0 = build_ef6_factorial_hooks(
                candidate, primary, context_b=bool(state[4]), z_head=None, zero_candidate=True
            )
            m_inj = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks)
            m0 = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks0)
            u = m_inj - m0
            lines.append(f"  {label} U mean={float(u.mean())} first={float(u[0])}")
        hooks_n = build_ef6_factorial_hooks(candidate, primary, context_b=bool(state[4]), z_head=zn, zero_candidate=False)
        hooks_c = build_ef6_factorial_hooks(candidate, primary, context_b=bool(state[4]), z_head=zc, zero_candidate=False)
        un = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks_n)
        uc = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks_c)
        lines.append(f"  |U_n-U_c| max={float((un-uc).abs().max())}")

    # Nondeterminism control: same z injected twice
    _, _, zn0 = score_pattern_z_pipeline(sa, sb, va, vb, partition, (0, 0, 0, 0, 0))
    hooks = build_ef6_factorial_hooks(candidate, primary, context_b=False, z_head=zn0, zero_candidate=False)
    u_a = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks)
    u_b = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks)
    lines.append(f"same-z repeat |Ua-Ub| max={float((u_a-u_b).abs().max())}")
    # Inject za (cached) vs zn0 (pipeline) — should be identical z
    hooks_za = build_ef6_factorial_hooks(candidate, primary, context_b=False, z_head=za, zero_candidate=False)
    hooks_zn = build_ef6_factorial_hooks(candidate, primary, context_b=False, z_head=zn0, zero_candidate=False)
    u_za = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks_za)
    u_zn = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks_zn)
    lines.append(f"|U(za)-U(zn0)| max={float((u_za-u_zn).abs().max())} despite z equal={float((za-zn0).abs().max())}")
    # Force identical centered z by using native pattern for both — only scores path differs
    # Use float64 canonicalize path
    sa64 = sa.double()
    sa_c64 = canonicalize_score_row(sa64, partition.valid_mask, "row_mean").float()
    _, _, z_hi = score_pattern_z_pipeline(sa_c64, sb, va, vb, partition, (0, 0, 0, 0, 0))
    lines.append(f"|zn0 - z_hi|={float((zn0-z_hi).abs().max())}")
    hooks_hi = build_ef6_factorial_hooks(candidate, primary, context_b=False, z_head=z_hi, zero_candidate=False)
    u_hi = target_logprob(model, tokens, dataset.query_pos.to("cuda"), dataset.target_token_id.to("cuda"), hooks_hi)
    lines.append(f"|U(zn0)-U(z_hi)| max={float((u_zn-u_hi).abs().max())}")

    lines.append(f"|za - zn0| max prompt={max_z_error_on_prompt(za, zn0, dataset.query_pos)}")
    _, _, zc0 = score_pattern_z_pipeline(sa_c, sb_c, va, vb, partition, (0, 0, 0, 0, 0))
    lines.append(f"|za - zc0| max prompt={max_z_error_on_prompt(za, zc0, dataset.query_pos)}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(mode: str = "all"):
    """
    mode: smoke | ef4 | ef6 | all | debug
    """
    if mode == "debug":
        print(debug_endpoint_gauge.remote())
        return

    if mode in ("smoke", "all"):
        print("Launching gauge smoke...")
        payload = run_gauge_smoke.remote()
        dest = REPO_ROOT / "results/extended_fali/gauge_sensitivity/smoke_ef6"
        _extract(payload, dest)
        print(json.dumps(json.loads((dest / "smoke_summary.json").read_text()), indent=2))
        summary = json.loads((dest / "smoke_summary.json").read_text())
        if not summary.get("endpoint_ok"):
            raise SystemExit("STOP: endpoint invariance failed in smoke")

    if mode in ("ef4", "all"):
        for seed in EF4_SEEDS:
            print(f"Launching EF-4 rowmean seed={seed}...")
            payload = run_ef4_gauge_seed.remote(seed=seed, score_gauge="row_mean")
            dest = REPO_ROOT / f"results/extended_fali/gauge_sensitivity/ef4_rowmean/seed_{seed}"
            _extract(payload, dest)
            print(f"Downloaded -> {dest}")

    if mode in ("ef6", "all"):
        for seed in EF6_SEEDS:
            print(f"Launching EF-6 rowmean seed={seed}...")
            payload = run_ef6_gauge_seed.remote(seed=seed, score_gauge="row_mean")
            dest = REPO_ROOT / f"results/extended_fali/gauge_sensitivity/ef6_rowmean/seed_{seed}"
            _extract(payload, dest)
            print(f"Downloaded -> {dest}")

    print("Done. Run experiments/extended_fali_gauge_analyze.py locally.")
