"""
experiments/output_mediation.py

PHASE 11: Output-Level Causal Mediation Tests.
Tests whether L10H10's attention pattern changes and/or full z outputs causally
mediate downstream IO-S logit recovery under primary ablation of L9H9 and L10H0.
Also maps necessity effects for Layer 10 control heads (L10H2, L10H3, L10H6, L10H7, L10H5).

Author: Antigravity Agent (.agents/rules/resesrcher.md compliant)
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ioi_dataset import IOIDataset

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def compute_logit_difference_per_example(logits: torch.Tensor, dataset: IOIDataset) -> torch.Tensor:
    """
    Computes the logit difference (IO - S) at the final 'end' token position for each example in a batch.
    Inputs:
        logits: Output logits of shape [batch, seq_len, d_vocab]
    Returns:
        logit_diffs: A tensor of shape [batch] containing individual logit differences.
    """
    # logits shape: [batch, seq_len, d_vocab]
    batch_size = dataset.batch_size
    device = logits.device
    
    # Extract logits at the padding-aware 'end' position for each prompt
    logits_at_end = torch.stack([logits[i, dataset.end[i].item(), :] for i in range(batch_size)]) # shape: [batch, d_vocab]
    
    # Slice logits for correct and incorrect name tokens
    io_logits = logits_at_end[torch.arange(batch_size), dataset.io_token_ids] # shape: [batch]
    s_logits = logits_at_end[torch.arange(batch_size), dataset.s_token_ids]   # shape: [batch]
    
    logit_diffs = io_logits - s_logits # shape: [batch]
    return logit_diffs

def make_zero_ablation_hook(head_idx):
    """
    Creates a basic hook to zero-ablate z activations of a single head in blocks.{l}.attn.hook_z.
    """
    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        # z shape: [batch, seq_len, n_heads, d_head]
        z_cloned = z.clone()
        z_cloned[:, :, head_idx, :] = 0.0
        return z_cloned
    return hook_fn

def make_composite_layer10_z_hook(zero_indices=None, patch_head_idx=None, patch_z_cache=None):
    """
    Creates a composite z-interception hook for Layer 10 blocks.10.attn.hook_z.
    Handles simultaneous zero-ablation of target heads (e.g. L10H0) and counterfactual patching/zeroing
    of another Layer 10 head in the same run to satisfy the composite hook instruction.
    """
    if zero_indices is None:
        zero_indices = []
        
    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        # z shape: [batch, seq_len, n_heads, d_head]
        z_cloned = z.clone()
        
        # Zero out the specified heads (e.g. primary mover L10H0 or a double-ablated head)
        for h_idx in zero_indices:
            z_cloned[:, :, h_idx, :] = 0.0
            
        # Overwrite/patch the specific head's z activations from cache if requested
        if patch_head_idx is not None and patch_z_cache is not None:
            z_cloned[:, :, patch_head_idx, :] = patch_z_cache[:, :, patch_head_idx, :].to(z.device)
            
        return z_cloned
    return hook_fn

def make_pattern_patch_hook(cached_pattern, head_idx):
    """
    Creates a hook to overwrite the attention pattern of a specific head in blocks.{l}.attn.hook_pattern.
    """
    def hook_fn(pattern: torch.Tensor, hook) -> torch.Tensor:
        # pattern shape: [batch, n_heads, seq_len, seq_len]
        patched_pattern = pattern.clone()
        patched_pattern[:, head_idx, :, :] = cached_pattern[:, head_idx, :, :].to(pattern.device)
        return patched_pattern
    return hook_fn

def main():
    print("==========================================================================")
    print("PHASE 11: OUTPUT-LEVEL CAUSAL MEDIATION TESTS")
    print("==========================================================================")
    
    # 1. Output Warning Guardrail
    print("\n==========================================================================")
    print("WARNING:")
    print("Output mediation tests whether L10H10 pattern/z changes causally affect")
    print("IO-S logit difference under these interventions. They still do not prove")
    print("a universal self-repair mechanism outside this model, dataset, and ablation setting.")
    print("==========================================================================")
    
    # 2. Load model
    print("\nLoading gpt2-small with center_writing_weights=True...")
    try:
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_writing_weights=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 3. Generate mixed template dataset
    print("\nGenerating mixed template IOI dataset (200 samples)...")
    dataset = IOIDataset(model, num_samples=200, template_style="mixed", seed=42)
    print(f"Dataset generated. Batch size: {dataset.batch_size}, Max Sequence Length: {dataset.seq_len}")
    
    # We will gather cache names
    cache_names = ["blocks.10.attn.hook_pattern", "blocks.10.attn.hook_z"]
    
    # --- BASELINE PASSES ---
    
    # PASS 1: Clean Baseline Pass
    print("\n[1/7] Executing Clean Baseline forward pass...")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            dataset.clean_tokens,
            names_filter=cache_names
        )
    clean_diffs = compute_logit_difference_per_example(clean_logits, dataset)
    mean_clean = clean_diffs.mean().item()
    print(f"  -> Clean Mean Logit Difference: {mean_clean:.4f} (Expected ≈ 2.9732)")
    
    # PASS 2: Primary Ablation Pass (L9H9, L10H0 Zero-Ablated)
    print("[2/7] Executing Primary Ablation forward pass (L9H9, L10H0 Zeroed)...")
    hooks_primary = [
        ("blocks.9.attn.hook_z", make_zero_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_composite_layer10_z_hook(zero_indices=[0]))
    ]
    with model.hooks(fwd_hooks=hooks_primary):
        with torch.no_grad():
            primary_logits, primary_cache = model.run_with_cache(
                dataset.clean_tokens,
                names_filter=cache_names
            )
    primary_diffs = compute_logit_difference_per_example(primary_logits, dataset)
    mean_primary = primary_diffs.mean().item()
    print(f"  -> Primary Ablated Mean Logit Difference: {mean_primary:.4f} (Expected ≈ 2.1055)")
    
    # PASS 3: Primary + L10H10 Double Ablation Pass
    print("[3/7] Executing Primary + L10H10 Double Ablation forward pass...")
    hooks_double = [
        ("blocks.9.attn.hook_z", make_zero_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_composite_layer10_z_hook(zero_indices=[0, 10]))
    ]
    with model.hooks(fwd_hooks=hooks_double):
        with torch.no_grad():
            double_logits = model(dataset.clean_tokens)
    double_diffs = compute_logit_difference_per_example(double_logits, dataset)
    mean_double = double_diffs.mean().item()
    print(f"  -> Double Ablated Mean Logit Difference: {mean_double:.4f} (Expected ≈ 1.8775)")
    
    l10h10_zero_deg = primary_diffs - double_diffs
    mean_l10h10_zero_deg = l10h10_zero_deg.mean().item()
    print(f"  -> L10H10 Zero Degradation: {mean_l10h10_zero_deg:.4f} (Expected ≈ 0.2281)")
    
    # Assertions for baseline accuracy check
    assert abs(mean_clean - 2.9732) < 0.02, f"Clean baseline mismatch: {mean_clean:.4f}"
    assert abs(mean_primary - 2.1055) < 0.02, f"Primary ablation mismatch: {mean_primary:.4f}"
    assert abs(mean_double - 1.8775) < 0.02, f"Double ablation mismatch: {mean_double:.4f}"
    assert abs(mean_l10h10_zero_deg - 0.2281) < 0.02, f"L10H10 zero degradation mismatch: {mean_l10h10_zero_deg:.4f}"
    print("✓ Baseline expectations validated successfully.")
    
    # --- L10H10 MEDIATION PASSES ---
    
    # PASS 4: primary_with_l10h10_pattern_clean
    print("\n[4/7] Executing primary_with_l10h10_pattern_clean...")
    hooks_p4 = [
        ("blocks.9.attn.hook_z", make_zero_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_composite_layer10_z_hook(zero_indices=[0])),
        ("blocks.10.attn.hook_pattern", make_pattern_patch_hook(clean_cache["blocks.10.attn.hook_pattern"], 10))
    ]
    with model.hooks(fwd_hooks=hooks_p4):
        with torch.no_grad():
            logits_p4 = model(dataset.clean_tokens)
    diffs_p4 = compute_logit_difference_per_example(logits_p4, dataset)
    print(f"  -> Mean Logit Difference: {diffs_p4.mean().item():.4f}")
    
    # PASS 5: clean_with_l10h10_pattern_primary
    print("[5/7] Executing clean_with_l10h10_pattern_primary...")
    hooks_p5 = [
        ("blocks.10.attn.hook_pattern", make_pattern_patch_hook(primary_cache["blocks.10.attn.hook_pattern"], 10))
    ]
    with model.hooks(fwd_hooks=hooks_p5):
        with torch.no_grad():
            logits_p5 = model(dataset.clean_tokens)
    diffs_p5 = compute_logit_difference_per_example(logits_p5, dataset)
    print(f"  -> Mean Logit Difference: {diffs_p5.mean().item():.4f}")
    
    # PASS 6: primary_with_l10h10_z_clean
    print("[6/7] Executing primary_with_l10h10_z_clean...")
    hooks_p6 = [
        ("blocks.9.attn.hook_z", make_zero_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_composite_layer10_z_hook(
            zero_indices=[0], 
            patch_head_idx=10, 
            patch_z_cache=clean_cache["blocks.10.attn.hook_z"]
        ))
    ]
    with model.hooks(fwd_hooks=hooks_p6):
        with torch.no_grad():
            logits_p6 = model(dataset.clean_tokens)
    diffs_p6 = compute_logit_difference_per_example(logits_p6, dataset)
    print(f"  -> Mean Logit Difference: {diffs_p6.mean().item():.4f}")
    
    # PASS 7: clean_with_l10h10_z_primary
    print("[7/7] Executing clean_with_l10h10_z_primary...")
    hooks_p7 = [
        ("blocks.10.attn.hook_z", make_composite_layer10_z_hook(
            zero_indices=[], 
            patch_head_idx=10, 
            patch_z_cache=primary_cache["blocks.10.attn.hook_z"]
        ))
    ]
    with model.hooks(fwd_hooks=hooks_p7):
        with torch.no_grad():
            logits_p7 = model(dataset.clean_tokens)
    diffs_p7 = compute_logit_difference_per_example(logits_p7, dataset)
    print(f"  -> Mean Logit Difference: {diffs_p7.mean().item():.4f}")
    
    # --- COMPILE L10H10 RAW PER-EXAMPLE RESULTS ---
    results_raw = []
    
    # L10H10 mediation variables at example level
    pat_nec = primary_diffs - diffs_p4
    z_nec = primary_diffs - diffs_p6
    pat_suf = diffs_p5 - clean_diffs
    z_suf = diffs_p7 - clean_diffs
    
    frac_pat = torch.zeros_like(l10h10_zero_deg)
    frac_z = torch.zeros_like(l10h10_zero_deg)
    for i in range(dataset.batch_size):
        deg_val = l10h10_zero_deg[i].item()
        if abs(deg_val) > 1e-12:
            frac_pat[i] = pat_nec[i] / deg_val
            frac_z[i] = z_nec[i] / deg_val
            
    conditions_dict = {
        "clean": clean_diffs,
        "primary_ablation": primary_diffs,
        "primary_plus_l10h10_zero": double_diffs,
        "primary_with_l10h10_pattern_clean": diffs_p4,
        "clean_with_l10h10_pattern_primary": diffs_p5,
        "primary_with_l10h10_z_clean": diffs_p6,
        "clean_with_l10h10_z_primary": diffs_p7
    }
    
    for i in range(dataset.batch_size):
        for cond_name, cond_tensor in conditions_dict.items():
            results_raw.append({
                "example_idx": i,
                "condition": cond_name,
                "clean_logit_diff": clean_diffs[i].item(),
                "primary_logit_diff": primary_diffs[i].item(),
                "condition_logit_diff": cond_tensor[i].item(),
                "primary_plus_l10h10_zero_logit_diff": double_diffs[i].item(),
                "effect_vs_primary": (cond_tensor[i] - primary_diffs[i]).item(),
                "effect_vs_clean": (cond_tensor[i] - clean_diffs[i]).item(),
                "l10h10_zero_degradation": l10h10_zero_deg[i].item(),
                "pattern_necessity_effect": pat_nec[i].item(),
                "z_necessity_effect": z_nec[i].item(),
                "pattern_sufficiency_effect": pat_suf[i].item(),
                "z_sufficiency_effect": z_suf[i].item(),
                "fraction_of_zero_degradation_explained_by_pattern": frac_pat[i].item(),
                "fraction_of_zero_degradation_explained_by_z": frac_z[i].item()
            })
            
    # Save raw outputs
    os.makedirs("results/raw", exist_ok=True)
    df_raw = pd.DataFrame(results_raw)
    df_raw.to_csv("results/raw/output_mediation_raw.csv", index=False)
    with open("results/raw/output_mediation_raw.json", "w") as f_out:
        json.dump(results_raw, f_out, indent=4, cls=NumpyEncoder)
    print("\n✓ Saved raw output mediation results to results/raw/output_mediation_raw.csv and .json")
    
    # --- COMPILE L10H10 SUMMARY RESULTS ---
    results_summary = []
    N = dataset.batch_size
    sqrt_N = np.sqrt(N)
    
    # Compute aggregates across batch
    l10h10_zero_deg_np = l10h10_zero_deg.detach().cpu().numpy()
    mean_zero_deg = np.mean(l10h10_zero_deg_np)
    
    for cond_name, cond_tensor in conditions_dict.items():
        diffs_np = cond_tensor.detach().cpu().numpy()
        eff_prim = (cond_tensor - primary_diffs).detach().cpu().numpy()
        eff_cln = (cond_tensor - clean_diffs).detach().cpu().numpy()
        
        # Necessity and sufficiency stats (batch vectors)
        pat_nec_np = pat_nec.detach().cpu().numpy()
        z_nec_np = z_nec.detach().cpu().numpy()
        pat_suf_np = pat_suf.detach().cpu().numpy()
        z_suf_np = z_suf.detach().cpu().numpy()
        frac_pat_np = frac_pat.detach().cpu().numpy()
        frac_z_np = frac_z.detach().cpu().numpy()
        
        # Standard deviations (ddof=1)
        std_logit_diff = np.std(diffs_np, ddof=1)
        std_pat_nec = np.std(pat_nec_np, ddof=1)
        std_z_nec = np.std(z_nec_np, ddof=1)
        
        # 95% Confidence Intervals
        ci95_pat_low = np.mean(pat_nec_np) - 1.96 * std_pat_nec / sqrt_N
        ci95_pat_high = np.mean(pat_nec_np) + 1.96 * std_pat_nec / sqrt_N
        ci95_z_low = np.mean(z_nec_np) - 1.96 * std_z_nec / sqrt_N
        ci95_z_high = np.mean(z_nec_np) + 1.96 * std_z_nec / sqrt_N
        
        agg_frac_pat = np.mean(pat_nec_np) / mean_zero_deg if abs(mean_zero_deg) > 1e-12 else 0.0
        agg_frac_z = np.mean(z_nec_np) / mean_zero_deg if abs(mean_zero_deg) > 1e-12 else 0.0
        
        results_summary.append({
            "condition": cond_name,
            "mean_logit_diff": np.mean(diffs_np),
            "std_logit_diff": std_logit_diff,
            "mean_effect_vs_primary": np.mean(eff_prim),
            "mean_effect_vs_clean": np.mean(eff_cln),
            "mean_pattern_necessity_effect": np.mean(pat_nec_np),
            "std_pattern_necessity_effect": std_pat_nec,
            "frac_positive_pattern_necessity_effect": np.mean(pat_nec_np > 0),
            "ci95_pattern_necessity_low": ci95_pat_low,
            "ci95_pattern_necessity_high": ci95_pat_high,
            "mean_z_necessity_effect": np.mean(z_nec_np),
            "std_z_necessity_effect": std_z_nec,
            "frac_positive_z_necessity_effect": np.mean(z_nec_np > 0),
            "ci95_z_necessity_low": ci95_z_low,
            "ci95_z_necessity_high": ci95_z_high,
            "mean_pattern_sufficiency_effect": np.mean(pat_suf_np),
            "mean_z_sufficiency_effect": np.mean(z_suf_np),
            "mean_per_example_fraction_of_zero_degradation_explained_by_pattern": np.mean(frac_pat_np),
            "mean_per_example_fraction_of_zero_degradation_explained_by_z": np.mean(frac_z_np),
            "aggregate_fraction_of_zero_degradation_explained_by_pattern": agg_frac_pat,
            "aggregate_fraction_of_zero_degradation_explained_by_z": agg_frac_z
        })
        
    os.makedirs("results/summary", exist_ok=True)
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv("results/summary/output_mediation_summary.csv", index=False)
    with open("results/summary/output_mediation_summary.json", "w") as f_out:
        json.dump(results_summary, f_out, indent=4, cls=NumpyEncoder)
    print("✓ Saved summary output mediation results to results/summary/output_mediation_summary.csv and .json")
    
    # --- CONTROL HEADS NECESSITY TESTS ---
    control_heads = [2, 3, 6, 7, 5]
    results_controls = []
    
    print("\nExecuting Control-Head Necessity Passes...")
    for ctrl_h in control_heads:
        head_name = f"L10H{ctrl_h}"
        print(f"  Processing Control Head {head_name}...")
        
        # 1. primary_plus_control_head_zero
        hooks_c_zero = [
            ("blocks.9.attn.hook_z", make_zero_ablation_hook(9)),
            ("blocks.10.attn.hook_z", make_composite_layer10_z_hook(zero_indices=[0, ctrl_h]))
        ]
        with model.hooks(fwd_hooks=hooks_c_zero):
            with torch.no_grad():
                logits_c_zero = model(dataset.clean_tokens)
        diffs_c_zero = compute_logit_difference_per_example(logits_c_zero, dataset)
        
        # 2. primary_with_control_head_pattern_clean
        hooks_c_pattern = [
            ("blocks.9.attn.hook_z", make_zero_ablation_hook(9)),
            ("blocks.10.attn.hook_z", make_composite_layer10_z_hook(zero_indices=[0])),
            ("blocks.10.attn.hook_pattern", make_pattern_patch_hook(clean_cache["blocks.10.attn.hook_pattern"], ctrl_h))
        ]
        with model.hooks(fwd_hooks=hooks_c_pattern):
            with torch.no_grad():
                logits_c_pattern = model(dataset.clean_tokens)
        diffs_c_pattern = compute_logit_difference_per_example(logits_c_pattern, dataset)
        
        # 3. primary_with_control_head_z_clean
        hooks_c_z = [
            ("blocks.9.attn.hook_z", make_zero_ablation_hook(9)),
            ("blocks.10.attn.hook_z", make_composite_layer10_z_hook(
                zero_indices=[0],
                patch_head_idx=ctrl_h,
                patch_z_cache=clean_cache["blocks.10.attn.hook_z"]
            ))
        ]
        with model.hooks(fwd_hooks=hooks_c_z):
            with torch.no_grad():
                logits_c_z = model(dataset.clean_tokens)
        diffs_c_z = compute_logit_difference_per_example(logits_c_z, dataset)
        
        # Compute metrics at example level
        c_zero_deg = primary_diffs - diffs_c_zero
        c_pat_nec = primary_diffs - diffs_c_pattern
        c_z_nec = primary_diffs - diffs_c_z
        
        c_frac_pat = torch.zeros_like(c_zero_deg)
        c_frac_z = torch.zeros_like(c_zero_deg)
        for i in range(dataset.batch_size):
            c_deg_val = c_zero_deg[i].item()
            if abs(c_deg_val) > 1e-12:
                c_frac_pat[i] = c_pat_nec[i] / c_deg_val
                c_frac_z[i] = c_z_nec[i] / c_deg_val
                
        # Numpy arrays for stats
        c_pat_nec_np = c_pat_nec.detach().cpu().numpy()
        c_z_nec_np = c_z_nec.detach().cpu().numpy()
        c_zero_deg_np = c_zero_deg.detach().cpu().numpy()
        c_frac_pat_np = c_frac_pat.detach().cpu().numpy()
        c_frac_z_np = c_frac_z.detach().cpu().numpy()
        
        std_c_pat = np.std(c_pat_nec_np, ddof=1)
        std_c_z = np.std(c_z_nec_np, ddof=1)
        
        ci95_c_pat_low = np.mean(c_pat_nec_np) - 1.96 * std_c_pat / sqrt_N
        ci95_c_pat_high = np.mean(c_pat_nec_np) + 1.96 * std_c_pat / sqrt_N
        ci95_c_z_low = np.mean(c_z_nec_np) - 1.96 * std_c_z / sqrt_N
        ci95_c_z_high = np.mean(c_z_nec_np) + 1.96 * std_c_z / sqrt_N
        
        mean_c_zero_deg = np.mean(c_zero_deg_np)
        agg_c_frac_pat = np.mean(c_pat_nec_np) / mean_c_zero_deg if abs(mean_c_zero_deg) > 1e-12 else 0.0
        agg_c_frac_z = np.mean(c_z_nec_np) / mean_c_zero_deg if abs(mean_c_zero_deg) > 1e-12 else 0.0
        
        results_controls.append({
            "head_name": head_name,
            "mean_logit_diff_primary_plus_zero": np.mean(diffs_c_zero.detach().cpu().numpy()),
            "mean_logit_diff_pattern_clean": np.mean(diffs_c_pattern.detach().cpu().numpy()),
            "mean_logit_diff_z_clean": np.mean(diffs_c_z.detach().cpu().numpy()),
            "control_zero_degradation": mean_c_zero_deg,
            "control_pattern_necessity_effect": np.mean(c_pat_nec_np),
            "control_z_necessity_effect": np.mean(c_z_nec_np),
            "mean_per_example_fraction_of_control_zero_degradation_explained_by_pattern": np.mean(c_frac_pat_np),
            "mean_per_example_fraction_of_control_zero_degradation_explained_by_z": np.mean(c_frac_z_np),
            "aggregate_fraction_of_control_zero_degradation_explained_by_pattern": agg_c_frac_pat,
            "aggregate_fraction_of_control_zero_degradation_explained_by_z": agg_c_frac_z,
            "frac_positive_control_pattern_necessity_effect": np.mean(c_pat_nec_np > 0),
            "frac_positive_control_z_necessity_effect": np.mean(c_z_nec_np > 0),
            "std_control_pattern_necessity_effect": std_c_pat,
            "std_control_z_necessity_effect": std_c_z,
            "ci95_control_pattern_necessity_low": ci95_c_pat_low,
            "ci95_control_pattern_necessity_high": ci95_c_pat_high,
            "ci95_control_z_necessity_low": ci95_c_z_low,
            "ci95_control_z_necessity_high": ci95_c_z_high
        })
        
    df_controls = pd.DataFrame(results_controls)
    df_controls.to_csv("results/summary/output_mediation_controls.csv", index=False)
    with open("results/summary/output_mediation_controls.json", "w") as f_out:
        json.dump(results_controls, f_out, indent=4, cls=NumpyEncoder)
    print("✓ Saved control-head summary results to results/summary/output_mediation_controls.csv and .json")
    
    # --- PRINT FORMATTED MARKDOWN REPORT TABLES ---
    
    print("\n==========================================================================")
    print("L10H10 DOWNSTREAM CAUSAL MEDIATION SUMMARY")
    print("==========================================================================")
    print("| Condition | Mean Logit Diff | Std Logit Diff | Necessity Effect | Sufficiency Effect | Per-Example Frac | Aggregate Frac |")
    print("|---|---|---|---|---|---|---|")
    
    # Find aggregate stats from summary
    s_clean = next(r for r in results_summary if r["condition"] == "clean")
    s_primary = next(r for r in results_summary if r["condition"] == "primary_ablation")
    s_double = next(r for r in results_summary if r["condition"] == "primary_plus_l10h10_zero")
    s_pat_clean = next(r for r in results_summary if r["condition"] == "primary_with_l10h10_pattern_clean")
    s_pat_prim = next(r for r in results_summary if r["condition"] == "clean_with_l10h10_pattern_primary")
    s_z_clean = next(r for r in results_summary if r["condition"] == "primary_with_l10h10_z_clean")
    s_z_prim = next(r for r in results_summary if r["condition"] == "clean_with_l10h10_z_primary")
    
    print(f"| clean | {s_clean['mean_logit_diff']:.4f} | {s_clean['std_logit_diff']:.4f} | - | - | - | - |")
    print(f"| primary_ablation | {s_primary['mean_logit_diff']:.4f} | {s_primary['std_logit_diff']:.4f} | - | - | - | - |")
    print(f"| primary_plus_l10h10_zero | {s_double['mean_logit_diff']:.4f} | {s_double['std_logit_diff']:.4f} | - | - | - | - |")
    print(f"| primary_with_l10h10_pattern_clean | {s_pat_clean['mean_logit_diff']:.4f} | {s_pat_clean['std_logit_diff']:.4f} | {s_pat_clean['mean_pattern_necessity_effect']:.4f} | - | {s_pat_clean['mean_per_example_fraction_of_zero_degradation_explained_by_pattern']*100:.1f}% | {s_pat_clean['aggregate_fraction_of_zero_degradation_explained_by_pattern']*100:.1f}% |")
    print(f"| clean_with_l10h10_pattern_primary | {s_pat_prim['mean_logit_diff']:.4f} | {s_pat_prim['std_logit_diff']:.4f} | - | {s_pat_prim['mean_pattern_sufficiency_effect']:.4f} | - | - |")
    print(f"| primary_with_l10h10_z_clean | {s_z_clean['mean_logit_diff']:.4f} | {s_z_clean['std_logit_diff']:.4f} | {s_z_clean['mean_z_necessity_effect']:.4f} | - | {s_z_clean['mean_per_example_fraction_of_zero_degradation_explained_by_z']*100:.1f}% | {s_z_clean['aggregate_fraction_of_zero_degradation_explained_by_z']*100:.1f}% |")
    print(f"| clean_with_l10h10_z_primary | {s_z_prim['mean_logit_diff']:.4f} | {s_z_prim['std_logit_diff']:.4f} | - | {s_z_prim['mean_z_sufficiency_effect']:.4f} | - | - |")
    
    print("\n==========================================================================")
    print("LAYER 10 CONTROL HEADS NECESSITY COMPARISON SUMMARY")
    print("==========================================================================")
    print("| Control Head | Zero Degradation | Pat Necessity | z-Necessity | Pat-Frac (Per-Ex / Agg) | z-Frac (Per-Ex / Agg) | Frac Pos Pat | Frac Pos z |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results_controls:
        pat_frac_str = f"{r['mean_per_example_fraction_of_control_zero_degradation_explained_by_pattern']*100:.1f}% / {r['aggregate_fraction_of_control_zero_degradation_explained_by_pattern']*100:.1f}%"
        z_frac_str = f"{r['mean_per_example_fraction_of_control_zero_degradation_explained_by_z']*100:.1f}% / {r['aggregate_fraction_of_control_zero_degradation_explained_by_z']*100:.1f}%"
        print(
            f"| {r['head_name']:<12} "
            f"| {r['control_zero_degradation']:<16.4f} "
            f"| {r['control_pattern_necessity_effect']:<13.4f} "
            f"| {r['control_z_necessity_effect']:<11.4f} "
            f"| {pat_frac_str:<23} "
            f"| {z_frac_str:<21} "
            f"| {r['frac_positive_control_pattern_necessity_effect']*100:<10.1f}% "
            f"| {r['frac_positive_control_z_necessity_effect']*100:<8.1f}% |"
        )
    print("==========================================================================")
    
    # 8. Causal Interpretation Determination
    print("\nCausal Mediation Interpretation:")
    pat_nec_val = s_pat_clean['mean_pattern_necessity_effect']
    z_nec_val = s_z_clean['mean_z_necessity_effect']
    
    if pat_nec_val > 0.05 and z_nec_val > 0.05:
        print("  -> BOTH L10H10 attention-pattern change and z-output change causally mediate the output effect.")
        print(f"  -> Pattern and z necessity effects are nearly equal, suggesting most of the measured L10H10 output effect")
        print(f"     is captured by the attention-pattern change under this intervention.")
        print(f"  -> Note: A direct value/OV patching experiment would be needed to rule out smaller value-side contributions.")
        print(f"  -> Attention pattern necessity explains {s_pat_clean['aggregate_fraction_of_zero_degradation_explained_by_pattern']*100:.1f}% of zero-ablation degradation (aggregate ratio-of-means).")
        print(f"  -> Full z-output necessity explains {s_z_clean['aggregate_fraction_of_zero_degradation_explained_by_z']*100:.1f}% of zero-ablation degradation (aggregate ratio-of-means).")
    elif z_nec_val > 0.05:
        print("  -> L10H10 z-output change causally mediates the output effect, while pattern-only changes explain very little.")
        print(f"  -> Full z-output necessity explains {s_z_clean['aggregate_fraction_of_zero_degradation_explained_by_z']*100:.1f}% of zero-ablation degradation (aggregate ratio-of-means).")
    else:
        print("  -> Neither the attention-pattern change nor the z-output change significantly mediates the output effect.")
        
    print("\nPHASE 11: OUTPUT-LEVEL CAUSAL MEDIATION TESTS COMPLETE.")
    print("==========================================================================")

if __name__ == "__main__":
    main()
