"""
Modal GPU runner for Extended FALI EF-5 ablation robustness.

Order: zero damage_drop (forward only) → smoke is run locally →
matched_mean seeds → matched_resample seeds → local aggregate analysis.
"""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import modal

APP_NAME = "fali-extended-ef5-ablation"
REPO_ROOT = Path(__file__).resolve().parent
EF5_SEEDS = [42, 1, 8, 22]
SCHEMES = ["matched_mean", "matched_resample"]

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
)

for seed in EF5_SEEDS:
    image = image.add_local_file(
        str(REPO_ROOT / f"results/revision_phase4/full/seed_{seed}/dataset_metadata.csv"),
        remote_path=f"/root/project/results/revision_phase4/full/seed_{seed}/dataset_metadata.csv",
    ).add_local_file(
        str(REPO_ROOT / f"results/revision_phase4/full/seed_{seed}/factorial_raw.csv"),
        remote_path=f"/root/project/results/revision_phase4/full/seed_{seed}/factorial_raw.csv",
    ).add_local_file(
        str(REPO_ROOT / f"results/extended_fali/ef4_canonical/seed_{seed}/attribution_raw.csv"),
        remote_path=f"/root/project/results/extended_fali/ef4_canonical/seed_{seed}/attribution_raw.csv",
    )

image = image.add_local_file(
    str(REPO_ROOT / "results/extended_fali/ef4_canonical/aggregate/hypothesis_results.json"),
    remote_path="/root/project/results/extended_fali/ef4_canonical/aggregate/hypothesis_results.json",
).add_local_file(
    str(REPO_ROOT / "results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json"),
    remote_path="/root/project/results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json",
)

app = modal.App(APP_NAME, image=image)
GPU_CONFIG = "L4"


def _run_cmd(cmd: list[str], project: Path) -> None:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(project), env=env)


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 2, memory=32768)
def run_zero_damage_remote(seed: int) -> bytes:
    project = Path("/root/project")
    out_dir = project / f"results/extended_fali/ef5_ablation_robustness/zero_damage/seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_cmd(
        [
            sys.executable,
            str(project / "experiments" / "extended_fali_ef5_zero_damage.py"),
            "--seed",
            str(seed),
            "--device",
            "cuda",
            "--require-cuda",
        ],
        project,
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(out_dir)))
    return buf.getvalue()


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 8, memory=32768)
def run_ef5_seed_remote(scheme: str, seed: int) -> bytes:
    project = Path("/root/project")
    out_dir = project / f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_cmd(
        [
            sys.executable,
            str(project / "experiments" / "extended_fali_ef5_ablation.py"),
            "--scheme",
            scheme,
            "--seed",
            str(seed),
            "--num-samples",
            "900",
            "--device",
            "cuda",
            "--require-cuda",
            "--output-dir",
            str(out_dir),
        ],
        project,
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(out_dir)))
    return buf.getvalue()


def _extract(payload: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        zf.extractall(dest)


@app.local_entrypoint()
def main():
    for seed in EF5_SEEDS:
        print(f"Zero damage_drop seed={seed}...")
        payload = run_zero_damage_remote.remote(seed=seed)
        dest = REPO_ROOT / f"results/extended_fali/ef5_ablation_robustness/zero_damage/seed_{seed}"
        _extract(payload, dest)

    for scheme in SCHEMES:
        for seed in EF5_SEEDS:
            print(f"EF-5 {scheme} seed={seed}...")
            payload = run_ef5_seed_remote.remote(scheme=scheme, seed=seed)
            dest = REPO_ROOT / f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}"
            _extract(payload, dest)

    print("Running aggregate analysis locally...")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "extended_fali_ef5_analyze.py")],
        check=True,
        cwd=str(REPO_ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    print(f"Done -> {REPO_ROOT / 'results/extended_fali/ef5_ablation_robustness/aggregate'}")
