"""
Modal GPU runner for Extended FALI EF-2 GPT-2 IOI smoke.

Runs ONLY N=8 seed=42. No --allow-full. No N=200.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import modal

APP_NAME = "fali-extended-ef2-gpt2-smoke"
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
        str(REPO_ROOT / "experiments" / "extended_fali_ef2_gpt2_smoke.py"),
        remote_path="/root/project/experiments/extended_fali_ef2_gpt2_smoke.py",
    )
    .add_local_file(
        str(REPO_ROOT / "results" / "revision_phase3c" / "smoke" / "dataset_metadata.csv"),
        remote_path="/root/project/results/revision_phase3c/smoke/dataset_metadata.csv",
    )
    .add_local_file(
        str(REPO_ROOT / "results" / "revision_phase3c" / "smoke" / "factorial_raw.csv"),
        remote_path="/root/project/results/revision_phase3c/smoke/factorial_raw.csv",
    )
)

app = modal.App(APP_NAME, image=image)
GPU_CONFIG = "L4"


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 2, memory=32768)
def run_ef2_remote(remote_output_rel: str) -> bytes:
    import os
    import subprocess
    import sys
    from pathlib import Path

    project = Path("/root/project")
    out_dir = project / remote_output_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(project / "experiments" / "extended_fali_ef2_gpt2_smoke.py"),
        "--num-samples",
        "8",
        "--seed",
        "42",
        "--device",
        "cuda",
        "--require-cuda",
        "--output-dir",
        str(out_dir),
        "--reference-meta",
        str(project / "results/revision_phase3c/smoke/dataset_metadata.csv"),
        "--historical-factorial",
        str(project / "results/revision_phase3c/smoke/factorial_raw.csv"),
    ]
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
def main(output_rel: str = "results/extended_fali/ef2_gpt2_smoke"):
    print(f"Launching Modal GPU={GPU_CONFIG} EF-2 smoke N=8 seed=42 -> {output_rel}")
    payload = run_ef2_remote.remote(remote_output_rel=output_rel)
    local_dir = REPO_ROOT / output_rel
    _extract_zip_to(local_dir, payload)
    print(f"Downloaded -> {local_dir} ({len(payload)} bytes)")
