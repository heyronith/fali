"""
experiments/fali_decomposition.py

PHASE 9: Path-Stable FALI Decomposition.
Decomposes total post-softmax attention spikes under primary ablation (L9H9, L10H0)
into target-logit and background-logit contributions via Shapley values.

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

def make_zero_ablation_hook(head_indices):
    """
    Surgically zero-ablates a list of heads in blocks.{l}.attn.hook_z.
    """
    def hook(z: torch.Tensor, hook) -> torch.Tensor:
        z_cloned = z.clone()
        for head_idx in head_indices:
            z_cloned[:, :, head_idx, :] = 0.0
        return z_cloned
    return hook

def f(x: torch.Tensor, io_pos: int) -> float:
    """
    Softmax function evaluated at the IO token position.
    Args:
        x: Pre-softmax attention logit scores of shape [q+1]
        io_pos: The key position of the IO token
    Returns:
        prob: Attention probability to IO token
    """
    probs = torch.softmax(x, dim=-1)
    return probs[io_pos].item()

def main():
    print("==========================================================================")
    print("PHASE 9: PATH-STABLE FALI SHAPLEY DECOMPOSITION")
    print("==========================================================================")
    
    # 1. Load gpt2-small with center_writing_weights=True
    print("Loading gpt2-small with center_writing_weights=True...")
    try:
        model = HookedTransformer.from_pretrained(
            "gpt2-small", 
            center_writing_weights=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    device = model.cfg.device
    
    # 2. Generate 200 mixed template IOI dataset
    print("\nGenerating mixed template IOI dataset (200 samples)...")
    dataset = IOIDataset(model, num_samples=200, template_style="mixed", seed=42)
    print(f"Dataset generated. Batch size: {dataset.batch_size}, Sequence Length: {dataset.seq_len}")
    
    # 3. Setup hooks filter to save memory
    attn_scores_hooks = [f"blocks.{l}.attn.hook_attn_scores" for l in range(12)]
    attn_patterns_hooks = [f"blocks.{l}.attn.hook_pattern" for l in range(12)]
    names_filter = attn_scores_hooks + attn_patterns_hooks
    
    # 4. PASS A: Clean cached run
    print("\nExecuting Clean baseline cached forward pass...")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            dataset.clean_tokens,
            names_filter=names_filter
        )
        
    # 5. PASS B: Ablated cached run (primary ablation L9H9 and L10H0)
    print("Executing Ablated cached forward pass (L9H9, L10H0 ablated)...")
    hooks_primary = [
        ("blocks.9.attn.hook_z", make_zero_ablation_hook([9])),
        ("blocks.10.attn.hook_z", make_zero_ablation_hook([0]))
    ]
    with model.hooks(fwd_hooks=hooks_primary):
        with torch.no_grad():
            ablated_logits, ablated_cache = model.run_with_cache(
                dataset.clean_tokens,
                names_filter=names_filter
            )
            
    print("\nProcessing path-stable FALI Shapley decomposition...")
    
    results_raw = []
    results_summary = []
    
    for l in range(12):
        for h in range(12):
            head_name = f"L{l}H{h}"
            
            clean_attns = []
            ablated_attns = []
            totals = []
            target_effects_tf = []
            background_effects_tf = []
            target_effects_bf = []
            background_effects_bf = []
            target_shapleys = []
            background_shapleys = []
            max_residual = 0.0
            
            for i in range(dataset.batch_size):
                q = dataset.end[i].item()
                io_pos = dataset.IO[i].item()
                
                # Fetch pre-softmax scores
                c_score = clean_cache[f"blocks.{l}.attn.hook_attn_scores"][i, h, q, :q+1]
                a_score = ablated_cache[f"blocks.{l}.attn.hook_attn_scores"][i, h, q, :q+1]
                
                # Sliced softmax probabilities
                fc = f(c_score, io_pos)
                fa = f(a_score, io_pos)
                
                total = fa - fc
                
                # Target-first path: target_first_vector = c with IO score replaced by a_IO
                target_first = c_score.clone()
                target_first[io_pos] = a_score[io_pos]
                f_tf = f(target_first, io_pos)
                target_effect_tf = f_tf - fc
                background_effect_tf = fa - f_tf
                
                # Background-first path: background_first_vector = a with IO score replaced by c_IO
                background_first = a_score.clone()
                background_first[io_pos] = c_score[io_pos]
                f_bf = f(background_first, io_pos)
                background_effect_bf = f_bf - fc
                target_effect_bf = fa - f_bf
                
                # Shapley decomposition
                target_shapley = 0.5 * (target_effect_tf + target_effect_bf)
                background_shapley = 0.5 * (background_effect_tf + background_effect_bf)
                
                # Validation residual check
                residual = abs((target_shapley + background_shapley) - total)
                if residual > max_residual:
                    max_residual = residual
                
                results_raw.append({
                    "example_idx": i,
                    "layer": l,
                    "head": h,
                    "head_name": head_name,
                    "clean_attn": fc,
                    "ablated_attn": fa,
                    "total_spike": total,
                    "target_effect_tf": target_effect_tf,
                    "background_effect_tf": background_effect_tf,
                    "target_effect_bf": target_effect_bf,
                    "background_effect_bf": background_effect_bf,
                    "target_shapley": target_shapley,
                    "background_shapley": background_shapley,
                    "target_shapley_fraction_of_total": target_shapley / total if total != 0 else 0.0,
                    "background_shapley_fraction_of_total": background_shapley / total if total != 0 else 0.0,
                    "decomposition_residual": residual
                })
                
                clean_attns.append(fc)
                ablated_attns.append(fa)
                totals.append(total)
                target_effects_tf.append(target_effect_tf)
                background_effects_tf.append(background_effect_tf)
                target_effects_bf.append(target_effect_bf)
                background_effects_bf.append(background_effect_bf)
                target_shapleys.append(target_shapley)
                background_shapleys.append(background_shapley)
                
            # Perform Shapley aggregates safely
            mean_total = np.mean(totals)
            mean_t_shap = np.mean(target_shapleys)
            mean_b_shap = np.mean(background_shapleys)
            
            mean_target_tf = np.mean(target_effects_tf)
            mean_target_bf = np.mean(target_effects_bf)
            
            target_background_disagreement_abs = abs(mean_target_tf - mean_target_bf)
            if abs(mean_t_shap) > 1e-12:
                target_background_disagreement_relative = target_background_disagreement_abs / abs(mean_t_shap)
            else:
                target_background_disagreement_relative = 0.0
                
            frac_examples_target_shapley_positive = np.mean([1.0 if ts > 0 else 0.0 for ts in target_shapleys])
            frac_examples_background_shapley_positive = np.mean([1.0 if bs > 0 else 0.0 for bs in background_shapleys])
            frac_examples_background_opposes_total = np.mean([1.0 if ((bs > 0 and tot < 0) or (bs < 0 and tot > 0)) else 0.0 for bs, tot in zip(background_shapleys, totals)])
            
            results_summary.append({
                "layer": l,
                "head": h,
                "head_name": head_name,
                "clean_attn": np.mean(clean_attns),
                "ablated_attn": np.mean(ablated_attns),
                "total_spike": mean_total,
                "target_effect_tf": mean_target_tf,
                "background_effect_tf": np.mean(background_effects_tf),
                "target_effect_bf": mean_target_bf,
                "background_effect_bf": np.mean(background_effects_bf),
                "target_shapley": mean_t_shap,
                "background_shapley": mean_b_shap,
                "target_shapley_fraction_of_total": mean_t_shap / mean_total if mean_total != 0 else 0.0,
                "background_shapley_fraction_of_total": mean_b_shap / mean_total if mean_total != 0 else 0.0,
                "target_background_disagreement_abs": target_background_disagreement_abs,
                "target_background_disagreement_relative": target_background_disagreement_relative,
                "frac_examples_target_shapley_positive": frac_examples_target_shapley_positive,
                "frac_examples_background_shapley_positive": frac_examples_background_shapley_positive,
                "frac_examples_background_opposes_total": frac_examples_background_opposes_total,
                "decomposition_residual_max_abs": max_residual
            })
            
    # Save raw outputs
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    
    df_raw = pd.DataFrame(results_raw)
    df_raw.to_csv("results/raw/fali_decomposition_raw.csv", index=False)
    with open("results/raw/fali_decomposition_raw.json", "w") as f_out:
        json.dump(results_raw, f_out, indent=4)
        
    # Save summary outputs
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv("results/summary/fali_decomposition_summary.csv", index=False)
    with open("results/summary/fali_decomposition_summary.json", "w") as f_out:
        json.dump(results_summary, f_out, indent=4)
        
    print("✓ Saved raw FALI decomposition results to results/raw/fali_decomposition_raw.csv and results/raw/fali_decomposition_raw.json")
    print("✓ Saved summary FALI decomposition results to results/summary/fali_decomposition_summary.csv and results/summary/fali_decomposition_summary.json")
    
    # Validation checks
    print("\n==========================================================================")
    print("WARNING:")
    print("This decomposition explains attention-score contributions, not full output-level causal mediation.")
    print("==========================================================================")
    
    print("\n==========================================================================")
    print("VALIDATION CHECKS:")
    print("==========================================================================")
    
    # 1. Look up L10H10 summary row
    l10h10_row = next(r for r in results_summary if r["head_name"] == "L10H10")
    print(f"L10H10 Clean Attn: {l10h10_row['clean_attn']:.4f} (Expected ≈ 0.3779)")
    print(f"L10H10 Ablated Attn: {l10h10_row['ablated_attn']:.4f} (Expected ≈ 0.5672)")
    
    assert abs(l10h10_row['clean_attn'] - 0.3779) < 0.01, "L10H10 clean attention mismatch"
    assert abs(l10h10_row['ablated_attn'] - 0.5672) < 0.01, "L10H10 ablated attention mismatch"
    
    # 2. Look up FALI Target-First Hybrid check
    # f(target_first_vector) = target_effect_tf + clean_attn
    l10h10_hybrid = l10h10_row["target_effect_tf"] + l10h10_row["clean_attn"]
    print(f"L10H10 Target-First Hybrid Value: {l10h10_hybrid:.4f} (Expected ≈ 0.6008)")
    assert abs(l10h10_hybrid - 0.6008) < 0.01, "L10H10 Target-First Hybrid value mismatch"
    
    # Print new stats
    print(f"L10H10 target_background_disagreement_abs: {l10h10_row['target_background_disagreement_abs']:.6f}")
    print(f"L10H10 target_background_disagreement_relative: {l10h10_row['target_background_disagreement_relative']:.6f}")
    print(f"L10H10 frac_examples_target_shapley_positive: {l10h10_row['frac_examples_target_shapley_positive']:.4f}")
    print(f"L10H10 frac_examples_background_shapley_positive: {l10h10_row['frac_examples_background_shapley_positive']:.4f}")
    print(f"L10H10 frac_examples_background_opposes_total: {l10h10_row['frac_examples_background_opposes_total']:.4f}")
    
    # 3. Sum check: max residual across all heads
    max_res_across_all = max(r["decomposition_residual_max_abs"] for r in results_summary)
    print(f"Max decomposition sum residual across all 144 heads: {max_res_across_all:.2e} (Expected < 1e-6)")
    assert max_res_across_all < 1e-6, f"Decomposition residual check failed: {max_res_across_all}"
    print("✓ All validation assertions successfully verified.")
    
    # 4. Print beautiful, structured markdown comparative table for Layer 10 heads
    print("\n==========================================================================")
    print("LAYER 10 HEADS PATH-STABLE SHAPLEY DECOMPOSITION")
    print("==========================================================================")
    headers = [
        "Head", "Clean Attn", "Ablated Attn", "Total Spike", 
        "T-Effect TF", "B-Effect TF", "T-Effect BF", "B-Effect BF",
        "Target Shap", "Bg Shap", "Target Shap %"
    ]
    print("| " + " | ".join(f"{h:<11}" for h in headers) + " |")
    print("|" + "|".join("-" * 13 for _ in headers) + "|")
    
    for r in results_summary:
        if r["layer"] == 10:
            target_pct = (r["target_shapley"] / r["total_spike"] * 100) if r["total_spike"] != 0 else 0.0
            highlight = " ★" if r["head_name"] in ["L10H10", "L10H2", "L10H3", "L10H6"] else ""
            print(
                f"| {r['head_name'] + highlight:<11} "
                f"| {r['clean_attn']:<11.4f} "
                f"| {r['ablated_attn']:<11.4f} "
                f"| {r['total_spike']:<11.4f} "
                f"| {r['target_effect_tf']:<11.4f} "
                f"| {r['background_effect_tf']:<11.4f} "
                f"| {r['target_effect_bf']:<11.4f} "
                f"| {r['background_effect_bf']:<11.4f} "
                f"| {r['target_shapley']:<11.4f} "
                f"| {r['background_shapley']:<11.4f} "
                f"| {target_pct:<10.1f}% |"
            )
    print("==========================================================================")
    print("★ Highlights canonical backup or secondary heads of interest.")
    print("==========================================================================")
    print("PHASE 9: PATH-STABLE FALI SHAPLEY DECOMPOSITION COMPLETE.")
    print("==========================================================================")

if __name__ == "__main__":
    main()
