#!/usr/bin/env python3
"""
Master Runner: run_all_experiments.py
Runs all mechanistic interpretability pipeline experiments sequentially,
manages imports, validates expected output files, and prints a comprehensive console audit.
"""

import os
import sys
import subprocess
import time
import shutil

# Expected output files to verify
EXPECTED_FILES = {
    "raw_results": [
        "results/raw/fali_raw_results.json",
        "results/raw/fali_raw_results.csv",
        "results/raw/ln_frozen_raw_results.json",
        "results/raw/ln_frozen_raw_results.csv"
    ],
    "summary_results": [
        "results/summary/attention_results.json",
        "results/summary/fali_rigorous_validation.csv",
        "results/summary/scale_evaluation_results.json"
    ],
    "figures": [
        "figures/circuit_disentanglement_plot.png",
        "figures/causal_shift_plot.png",
        "figures/isolation_bar_chart.png",
        "figures/cossim_shift_lollipop.png"
    ]
}

EXPERIMENT_STEPS = [
    {
        "name": "Verify Setup & Dataset",
        "script": "experiments/verify_setup.py",
        "description": "Environment verification and IOI dataset shape constraints validation."
    },
    {
        "name": "Cache Clean Baselines",
        "script": "experiments/cache_baseline.py",
        "description": "Run clean forward pass with caching to capture standard behavior."
    },
    {
        "name": "Ablate Primary Mover Heads",
        "script": "experiments/ablate_primary.py",
        "description": "Evaluate basic effects of primary heads (L9H9, L10H0) ablation."
    },
    {
        "name": "FALI Hybrid Intervention",
        "script": "experiments/hybrid_intervention.py",
        "description": "Conduct target-logit counterfactual interventions at the sub-softmax level."
    },
    {
        "name": "LayerNorm-Frozen Validation",
        "script": "experiments/fali_rigorous_validation.py",
        "description": "Perform physical scale-freezing and Query-Key cosine similarity tracking."
    },
    {
        "name": "Model-Wide Scale Evaluation",
        "script": "experiments/scale_evaluation.py",
        "description": "Map active vs. passive composition across all 144 attention heads."
    },
    {
        "name": "Causal Double Ablation",
        "script": "experiments/causal_validation.py",
        "description": "Execute four-pass protocol verifying causal indispensability of L10H10."
    },
    {
        "name": "Generate Analytical Figures",
        "script": "experiments/generate_new_figures.py",
        "description": "Produce publication-ready lollipop, bar, and trajectory charts."
    }
]

def main():
    print("=" * 80)
    print("  CANONICAL MECHANISTIC INTERPRETABILITY PIPELINE RUNNER")
    print("=" * 80)
    print("Initializing environment and clearing/setting directory structure...")
    
    # Ensure correct directories exist
    os.makedirs("src", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # Configure environment
    env = os.environ.copy()
    # Prepend 'src' to the PYTHONPATH to ensure smooth, robust imports
    env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + env.get("PYTHONPATH", "")
    
    step_results = []
    
    # Execute each step in sequence
    for i, step in enumerate(EXPERIMENT_STEPS, 1):
        name = step["name"]
        script = step["script"]
        desc = step["description"]
        
        print(f"\n[{i}/{len(EXPERIMENT_STEPS)}] Running Step: {name}")
        print(f"  Script: {script}")
        print(f"  Description: {desc}")
        print("-" * 60)
        
        # Verify script exists
        if not os.path.exists(script):
            print(f"❌ Error: Script file not found: {script}")
            step_results.append({
                "step": i,
                "name": name,
                "status": "MISSING",
                "duration": 0.0,
                "error": "Script file not found"
            })
            continue
            
        start_time = time.time()
        try:
            # We let stdout/stderr stream directly to print logs live
            result = subprocess.run(
                [sys.executable, script],
                env=env,
                check=True
            )
            duration = time.time() - start_time
            print(f"✓ Step {i} completed successfully in {duration:.2f}s.")
            step_results.append({
                "step": i,
                "name": name,
                "status": "PASSED",
                "duration": duration,
                "error": ""
            })
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"❌ Error running script: {script}")
            print(f"  Exit Code: {e.returncode}")
            step_results.append({
                "step": i,
                "name": name,
                "status": "FAILED",
                "duration": duration,
                "error": f"Exit Code {e.returncode}"
            })
            # Stop execution on failure to preserve pipeline sanity
            print("\n❌ Pipeline execution halted due to failure in critical step.")
            break
        except Exception as ex:
            duration = time.time() - start_time
            print(f"❌ Unexpected error: {ex}")
            step_results.append({
                "step": i,
                "name": name,
                "status": "ERROR",
                "duration": duration,
                "error": str(ex)
            })
            break
            
    print("\n" + "=" * 80)
    print("  VERIFYING PIPELINE OUTPUT ARTIFACTS")
    print("=" * 80)
    
    all_files_exist = True
    verified_files = []
    
    for category, files in EXPECTED_FILES.items():
        print(f"\nCategory: {category.upper()}")
        print("-" * 40)
        for f in files:
            exists = os.path.exists(f)
            size_str = ""
            if exists:
                size_str = f"({os.path.getsize(f)} bytes)"
                print(f"  ✓ [FOUND] {f:<50} {size_str}")
                verified_files.append((f, True, size_str))
            else:
                print(f"  ❌ [MISSING] {f:<50}")
                verified_files.append((f, False, ""))
                all_files_exist = False

    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION SUMMARY AUDIT")
    print("=" * 80)
    print(f"| {'Step':<4} | {'Experiment Step Name':<30} | {'Status':<10} | {'Duration (s)':<12} | {'Notes':<12} |")
    print(f"|{'-' * 6}|{'-' * 32}|{'-' * 12}|{'-' * 14}|{'-' * 14}|")
    
    total_time = 0.0
    all_passed = True
    for res in step_results:
        total_time += res["duration"]
        status_str = res["status"]
        if status_str != "PASSED":
            all_passed = False
        print(f"| {res['step']:<4} | {res['name']:<30} | {status_str:<10} | {res['duration']:<12.2f} | {res['error']:<12} |")
    print(f"|{'-' * 6}|{'-' * 32}|{'-' * 12}|{'-' * 14}|{'-' * 14}|")
    print(f"Total pipeline execution duration: {total_time:.2f}s")
    
    if all_passed and all_files_exist:
        print("\n🎉 SUCCESS: All experiments ran successfully, and all output files exist.")
        sys.exit(0)
    else:
        print("\n⚠ WARNING: Some experiments failed or some expected files are missing. Inspect audit logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
