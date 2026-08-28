"""
Modal GPU runner for Extended FALI EF-3 two-template full analysis.

Runs seeds 42, 1, 8, 22 sequentially at N=200 each.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import modal

APP_NAME = "fali-extended-ef3-two-template"
REPO_ROOT = Path(__file__).resolve().parent
EF3_SEEDS = [42, 1, 8, 22]

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
    .add_local_file(
        str(REPO_ROOT / "experiments" / "extended_fali_ef3_full.py"),
        remote_path="/root/project/experiments/extended_fali_ef3_full.py",
    )
    .add_local_file(
        str(REPO_ROOT / "experiments" / "extended_fali_ef3_analyze.py"),
        remote_path="/root/project/experiments/extended_fali_ef3_analyze.py",
    )
)

for seed in EF3_SEEDS:
    image = image.add_local_file(
        str(REPO_ROOT / f"results/revision_phase1/full/seed_{seed}/dataset_metadata.csv"),
        remote_path=f"/root/project/results/revision_phase1/full/seed_{seed}/dataset_metadata.csv",
    ).add_local_file(
        str(REPO_ROOT / f"results/revision_phase3c/full/seed_{seed}/factorial_raw.csv"),
        remote_path=f"/root/project/results/revision_phase3c/full/seed_{seed}/factorial_raw.csv",
    )

app = modal.App(APP_NAME, image=image)
GPU_CONFIG = "L4"


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 6, memory=32768)
def run_ef3_seed_remote(seed: int) -> bytes:
    import os
    import subprocess
    import sys
    from pathlib import Path

    project = Path("/root/project")
    out_dir = project / f"results/extended_fali/ef3_two_template/seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(project / "experiments/extended_fali_ef3_full.py"),
        "--seed",
        str(seed),
        "--num-samples",
        "200",
        "--device",
        "cuda",
        "--require-cuda",
        "--output-dir",
        str(out_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(project))

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
    for seed in EF3_SEEDS:
        print(f"Launching EF-3 seed={seed} GPU={GPU_CONFIG}...")
        payload = run_ef3_seed_remote.remote(seed=seed)
        dest = REPO_ROOT / f"results/extended_fali/ef3_two_template/seed_{seed}"
        _extract(payload, dest)
        print(f"Downloaded seed {seed} -> {dest}")

    print("Running aggregate analysis locally...")
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "extended_fali_ef3_analyze.py")],
        check=True,
        cwd=str(REPO_ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    print(f"Analysis complete -> {REPO_ROOT / 'results/extended_fali/ef3_two_template/aggregate'}")
