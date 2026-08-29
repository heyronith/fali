"""
Modal GPU runner for Extended FALI EF-6 Pythia induction transfer.
"""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import modal

APP_NAME = "fali-extended-ef6-pythia"
REPO_ROOT = Path(__file__).resolve().parent
CONFIRM_SEEDS = [1, 8, 22]

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
        str(REPO_ROOT / "results/extended_fali/ef6_pythia_induction/aggregate/selection_lock.json"),
        remote_path="/root/project/results/extended_fali/ef6_pythia_induction/aggregate/selection_lock.json",
    )
    .add_local_file(
        str(REPO_ROOT / "results/extended_fali/ef6_pythia_induction/model_revision.json"),
        remote_path="/root/project/results/extended_fali/ef6_pythia_induction/model_revision.json",
    )
)

app = modal.App(APP_NAME, image=image)
GPU_CONFIG = "L4"


def _run_script(script: str, extra: list[str]) -> bytes:
    import os
    from pathlib import Path

    project = Path("/root/project")
    os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    cmd = [sys.executable, str(project / "experiments" / script), *extra, "--device", "cuda", "--require-cuda"]
    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(project), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    if result.returncode != 0:
        for rel in (
            "results/extended_fali/ef6_pythia_induction/smoke/validation_report.json",
            "results/extended_fali/ef6_pythia_induction/discovery/selection_report.json",
        ):
            p = project / rel
            if p.exists():
                print("---", rel, "---", flush=True)
                print(p.read_text(), flush=True)
        result.check_returncode()
    return b""


def _zip_dir(path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(path.rglob("*")):
            if fp.is_file():
                zf.write(fp, arcname=str(fp.relative_to(path.parent)))
    return buf.getvalue()


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 4, memory=32768)
def run_smoke_remote() -> bytes:
    from pathlib import Path

    out = Path("/root/project/results/extended_fali/ef6_pythia_induction/smoke")
    _run_script("extended_fali_ef6_smoke.py", ["--output-dir", str(out)])
    return _zip_dir(out.parent / "smoke")


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 12, memory=32768)
def run_discovery_remote() -> bytes:
    from pathlib import Path

    out = Path("/root/project/results/extended_fali/ef6_pythia_induction/discovery")
    _run_script("extended_fali_ef6_discovery.py", ["--output-dir", str(out)])
    root = Path("/root/project/results/extended_fali/ef6_pythia_induction")
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sub in ("discovery", "aggregate"):
            p = root / sub
            if p.is_dir():
                for fp in sorted(p.rglob("*")):
                    if fp.is_file():
                        zf.write(fp, arcname=str(fp.relative_to(root)))
    return buf.getvalue()


@app.function(gpu=GPU_CONFIG, timeout=60 * 60 * 8, memory=32768)
def run_confirm_remote(seed: int) -> bytes:
    from pathlib import Path

    lock = Path("/root/project/results/extended_fali/ef6_pythia_induction/aggregate/selection_lock.json")
    out = Path(f"/root/project/results/extended_fali/ef6_pythia_induction/confirmation/seed_{seed}")
    _run_script(
        "extended_fali_ef6_confirm.py",
        ["--seed", str(seed), "--selection-lock", str(lock), "--output-dir", str(out)],
    )
    return _zip_dir(out)


def _extract(payload: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        zf.extractall(dest.parent if dest.name.startswith("seed_") else dest)


@app.local_entrypoint()
def smoke():
    payload = run_smoke_remote.remote()
    dest = REPO_ROOT / "results/extended_fali/ef6_pythia_induction/smoke"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        zf.extractall(REPO_ROOT / "results/extended_fali/ef6_pythia_induction")
    print("Smoke complete ->", dest)


@app.local_entrypoint()
def discovery():
    payload = run_discovery_remote.remote()
    dest = REPO_ROOT / "results/extended_fali/ef6_pythia_induction"
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        zf.extractall(dest)
    print("Discovery complete ->", dest)


@app.local_entrypoint()
def confirm():
    for seed in CONFIRM_SEEDS:
        print(f"Confirmation seed={seed}...")
        payload = run_confirm_remote.remote(seed=seed)
        out = REPO_ROOT / f"results/extended_fali/ef6_pythia_induction/confirmation/seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
            zf.extractall(REPO_ROOT / "results/extended_fali/ef6_pythia_induction/confirmation")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments/extended_fali_ef6_analyze.py")],
        check=False,
        cwd=str(REPO_ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
    )


@app.local_entrypoint()
def full():
    smoke()
    discovery()
    confirm()
