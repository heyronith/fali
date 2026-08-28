"""
Modal GPU runner for FALI Revision Phase-4 canonical-IOI factorial.

Modes:
  smoke    — N=60 (2×30 strata) seed42
  full     — single seed N=900
  full-all — seeds 42,1,8,22 × N=900 (parallel)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import modal

APP_NAME = "fali-revision-phase4"
REPO_ROOT = Path(__file__).resolve().parent

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
        str(REPO_ROOT / "experiments" / "revision_phase4_factorial.py"),
        remote_path="/root/project/experiments/revision_phase4_factorial.py",
    )
)

app = modal.App(APP_NAME, image=image)
GPU_CONFIG = "L4"


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 5, memory=32768)
def run_phase4_remote(
    num_samples: int,
    seed: int,
    remote_output_rel: str,
    allow_full: bool,
) -> bytes:
    import os
    import subprocess
    import sys
    from pathlib import Path

    project = Path("/root/project")
    out_dir = project / remote_output_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + os.environ.get(
        "PYTHONPATH", ""
    )

    cmd = [
        sys.executable,
        str(project / "experiments" / "revision_phase4_factorial.py"),
        "--num-samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--require-cuda",
        "--output-dir",
        str(out_dir),
    ]
    if num_samples <= 60:
        cmd.append("--smoke")
    if allow_full:
        cmd.append("--allow-full")

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(project))

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
    num_samples: int = 60,
    output_rel: str = "results/revision_phase4/smoke",
):
    """
    Examples:
      modal run modal_phase4.py --mode smoke
      modal run modal_phase4.py --mode full-all
    """
    if mode == "full-all":
        seeds = [42, 1, 8, 22]
        print(
            f"Launching Modal GPU={GPU_CONFIG} Phase4B full-all seeds={seeds} N=900"
        )
        handles = []
        for s in seeds:
            out_rel = f"results/revision_phase4/full/seed_{s}"
            handles.append(
                (
                    s,
                    out_rel,
                    run_phase4_remote.spawn(
                        num_samples=900,
                        seed=s,
                        remote_output_rel=out_rel,
                        allow_full=True,
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

    if mode == "full":
        num_samples = 900
        output_rel = f"results/revision_phase4/full/seed_{seed}"
        allow_full = True
    elif mode == "smoke":
        if num_samples != 60:
            raise SystemExit("Phase-4 smoke requires N=60 (2 × 30 strata).")
        allow_full = False
    else:
        raise SystemExit(f"Unknown mode={mode}; use smoke|full|full-all")

    print(
        f"Launching Modal GPU={GPU_CONFIG} mode={mode} seed={seed} "
        f"N={num_samples} -> {output_rel}"
    )
    payload = run_phase4_remote.remote(
        num_samples=num_samples,
        seed=seed,
        remote_output_rel=output_rel,
        allow_full=allow_full,
    )
    local_dir = REPO_ROOT / output_rel
    _extract_zip_to(local_dir, payload)
    print(f"Downloaded -> {local_dir} ({len(payload)} bytes)")
