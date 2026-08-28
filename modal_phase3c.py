"""
Modal GPU runner for FALI Revision Phase-3C/3D factorial.

Modes:
  smoke     — N=8 seed42
  full      — single seed N=200
  full-all  — seeds 42,1,8,22
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import modal

APP_NAME = "fali-revision-phase3c"
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
        str(REPO_ROOT / "experiments" / "revision_phase3c_factorial.py"),
        remote_path="/root/project/experiments/revision_phase3c_factorial.py",
    )
    .add_local_file(
        str(REPO_ROOT / "results" / "revision_phase1" / "smoke" / "dataset_metadata.csv"),
        remote_path="/root/project/results/revision_phase1/smoke/dataset_metadata.csv",
    )
)

# Mount Phase-1 metadata + Phase-2 combined for each seed
for _seed in (42, 1, 8, 22):
    image = image.add_local_file(
        str(
            REPO_ROOT
            / "results"
            / "revision_phase1"
            / "full"
            / f"seed_{_seed}"
            / "dataset_metadata.csv"
        ),
        remote_path=f"/root/project/results/revision_phase1/full/seed_{_seed}/dataset_metadata.csv",
    ).add_local_file(
        str(
            REPO_ROOT
            / "results"
            / "revision_phase2"
            / "full"
            / f"seed_{_seed}"
            / "phase2_combined_raw.csv"
        ),
        remote_path=f"/root/project/results/revision_phase2/full/seed_{_seed}/phase2_combined_raw.csv",
    )

app = modal.App(APP_NAME, image=image)
GPU_CONFIG = "L4"


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 4, memory=32768)
def run_phase3c_remote(
    num_samples: int,
    seed: int,
    remote_output_rel: str,
    allow_full: bool,
    phase1_meta_rel: str,
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
        str(project / "experiments" / "revision_phase3c_factorial.py"),
        "--num-samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--require-cuda",
        "--output-dir",
        str(out_dir),
        "--phase1-meta",
        str(project / phase1_meta_rel),
    ]
    if num_samples <= 8:
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
    num_samples: int = 8,
    output_rel: str = "results/revision_phase3c/smoke",
):
    """
    Examples:
      modal run modal_phase3c.py --mode smoke
      modal run modal_phase3c.py --mode full-all
    """
    if mode == "full-all":
        seeds = [42, 1, 8, 22]
        print(f"Launching Modal GPU={GPU_CONFIG} Phase3D full-all seeds={seeds} N=200")
        handles = []
        for s in seeds:
            out_rel = f"results/revision_phase3c/full/seed_{s}"
            meta_rel = f"results/revision_phase1/full/seed_{s}/dataset_metadata.csv"
            handles.append(
                (
                    s,
                    out_rel,
                    run_phase3c_remote.spawn(
                        num_samples=200,
                        seed=s,
                        remote_output_rel=out_rel,
                        allow_full=True,
                        phase1_meta_rel=meta_rel,
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

    allow_full = mode == "full"
    if mode == "full":
        num_samples = 200
        output_rel = f"results/revision_phase3c/full/seed_{seed}"
        phase1_meta_rel = f"results/revision_phase1/full/seed_{seed}/dataset_metadata.csv"
    else:
        phase1_meta_rel = "results/revision_phase1/smoke/dataset_metadata.csv"

    print(
        f"Launching Modal GPU={GPU_CONFIG} mode={mode} seed={seed} N={num_samples} -> {output_rel}"
    )
    payload = run_phase3c_remote.remote(
        num_samples=num_samples,
        seed=seed,
        remote_output_rel=output_rel,
        allow_full=allow_full,
        phase1_meta_rel=phase1_meta_rel,
    )
    local_dir = REPO_ROOT / output_rel
    _extract_zip_to(local_dir, payload)
    print(f"Downloaded -> {local_dir} ({len(payload)} bytes)")
