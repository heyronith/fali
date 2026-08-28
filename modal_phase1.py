"""
Modal GPU runner for FALI Revision Phase-1.

Runs the Phase-1 conditional-gain experiment on a remote CUDA GPU and downloads
results into the local repository under results/revision_phase1/.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import modal

APP_NAME = "fali-revision-phase1"
REPO_ROOT = Path(__file__).resolve().parent

# CUDA PyTorch >=2.6 required by transformer-lens==3.2.1.
# Pin transformers near local Phase-1A smoke (4.57.x) to preserve single-token IOI names.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .pip_install(
        "transformer-lens==3.2.1",
        "transformers==4.57.6",
        "einops>=0.7.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "plotly>=5.15.0",
        "accelerate>=0.26.0",
        "huggingface-hub>=0.24.0,<1.0",
        "jaxtyping>=0.2.0",
        "typeguard>=4.0.0",
        "better-abc>=0.0.3",
        "fancy-einsum>=0.0.3",
        "datasets>=2.7.1",
        "sentencepiece",
        "rich>=12.6.0",
        "beartype>=0.14.1",
        "transformers-stream-generator>=0.0.5,<0.1",
    )
    .env({"HF_HOME": "/tmp/hf_cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/project/src")
    .add_local_file(
        str(REPO_ROOT / "experiments" / "revision_phase1_conditional_gain.py"),
        remote_path="/root/project/experiments/revision_phase1_conditional_gain.py",
    )
)

app = modal.App(APP_NAME, image=image)

# Prefer L4; allow fallbacks if the workspace queues differently.
GPU_CONFIG = "L4"


@app.function(
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 4,
    memory=32768,
)
def run_phase1_remote(
    num_samples: int,
    seed: int,
    smoke: bool,
    n_bootstrap: int,
    remote_output_rel: str,
) -> bytes:
    """
    Execute Phase-1 on CUDA and return a zip of the output directory.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    project = Path("/root/project")
    out_dir = project / remote_output_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure imports resolve
    os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(project / "experiments" / "revision_phase1_conditional_gain.py"),
        "--num-samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--require-cuda",
        "--output-dir",
        str(out_dir),
        "--n-bootstrap",
        str(n_bootstrap),
    ]
    if smoke:
        cmd.append("--smoke")

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(project))

    # Zip results for download
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(out_dir)))
    return buf.getvalue()


def _extract_zip_to(local_dir: Path, payload: bytes) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        zf.extractall(local_dir)


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    seed: int = 42,
    num_samples: int = 8,
    n_bootstrap: int = 200,
    output_rel: str = "results/revision_phase1/modal_smoke_seed42",
):
    """
    Examples:
      modal run modal_phase1.py --mode smoke
      modal run modal_phase1.py --mode full --seed 42 --num-samples 200 --n-bootstrap 10000 \\
          --output-rel results/revision_phase1/full/seed_42
      modal run modal_phase1.py --mode full-all
    """
    if mode == "full-all":
        seeds = [42, 1, 8, 22]
        print(f"Launching Modal GPU={GPU_CONFIG} full-all seeds={seeds}")
        handles = []
        for s in seeds:
            out_rel = f"results/revision_phase1/full/seed_{s}"
            handles.append(
                (
                    s,
                    out_rel,
                    run_phase1_remote.spawn(
                        num_samples=200,
                        seed=s,
                        smoke=False,
                        n_bootstrap=10000,
                        remote_output_rel=out_rel,
                    ),
                )
            )
        for s, out_rel, handle in handles:
            print(f"Waiting for seed={s} ...")
            payload = handle.get()
            local_dir = REPO_ROOT / out_rel
            _extract_zip_to(local_dir, payload)
            print(f"Downloaded seed={s} -> {local_dir} ({len(payload)} bytes)")
        return

    smoke = mode == "smoke"
    if mode == "full":
        smoke = False
        if num_samples < 200:
            num_samples = 200
        if n_bootstrap < 10000:
            n_bootstrap = 10000

    print(
        f"Launching Modal GPU={GPU_CONFIG} mode={mode} seed={seed} "
        f"N={num_samples} bootstrap={n_bootstrap} -> {output_rel}"
    )
    payload = run_phase1_remote.remote(
        num_samples=num_samples,
        seed=seed,
        smoke=smoke,
        n_bootstrap=n_bootstrap,
        remote_output_rel=output_rel,
    )
    local_dir = REPO_ROOT / output_rel
    _extract_zip_to(local_dir, payload)
    print(f"Downloaded results to {local_dir} ({len(payload)} bytes zip)")
    for p in sorted(local_dir.glob("*")):
        print(" -", p.name)
