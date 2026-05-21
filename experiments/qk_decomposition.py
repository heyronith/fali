"""
experiments/qk_decomposition.py

PHASE 10: QK Norm/Direction Decomposition.
Decomposes pre-softmax attention score changes on the IOI mixed dataset
into query-vector, key-vector, norm-product, and cosine/directional alignment
components using exact Shapley value decomposition and counterfactual evaluations.

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
        # z shape: [batch_size, seq_len, n_heads, d_head]
        z_cloned = z.clone()
        for head_idx in head_indices:
            # Setting hook_z to zero for the targeted head indices across all positions
            z_cloned[:, :, head_idx, :] = 0.0
        return z_cloned
    return hook

def compute_shapley_vector(q_c, q_a, k_c, k_a, d_h):
    """
    Computes vector-level Shapley value decomposition (Query vs Key).
    Inputs:
        q_c, q_a: Clean and ablated query vectors of shape [d_head]
        k_c, k_a: Clean and ablated key vectors of shape [d_head]
    """
    # q_c, q_a, k_c, k_a shape: [d_head]
    sqrt_dh = np.sqrt(d_h)
    
    s_cc = torch.dot(q_c, k_c).item() / sqrt_dh
    s_aa = torch.dot(q_a, k_a).item() / sqrt_dh
    s_ca = torch.dot(q_c, k_a).item() / sqrt_dh
    s_ac = torch.dot(q_a, k_c).item() / sqrt_dh
    
    qk_score_delta = s_aa - s_cc
    
    # query_shapley: Query changing while holding Key constant
    query_shapley = 0.5 * ((s_ac - s_cc) + (s_aa - s_ca))
    # key_shapley: Key changing while holding Query constant
    key_shapley = 0.5 * ((s_ca - s_cc) + (s_aa - s_ac))
    
    residual = abs(query_shapley + key_shapley - qk_score_delta)
    return query_shapley, key_shapley, qk_score_delta, s_cc, s_aa, residual

def compute_shapley_geometry(q_c, q_a, k_c, k_a, d_h):
    """
    Computes geometry-level Shapley value decomposition (Norm Product vs Cosine).
    Inputs:
        q_c, q_a: Clean and ablated query vectors of shape [d_head]
        k_c, k_a: Clean and ablated key vectors of shape [d_head]
    """
    # q_c, q_a, k_c, k_a shape: [d_head]
    sqrt_dh = np.sqrt(d_h)
    
    norm_q_c = torch.linalg.norm(q_c).item()
    norm_k_c = torch.linalg.norm(k_c).item()
    norm_q_a = torch.linalg.norm(q_a).item()
    norm_k_a = torch.linalg.norm(k_a).item()
    
    n_c = norm_q_c * norm_k_c
    n_a = norm_q_a * norm_k_a
    
    dot_cc = torch.dot(q_c, k_c).item()
    dot_aa = torch.dot(q_a, k_a).item()
    
    cos_c = dot_cc / n_c if n_c > 1e-12 else 0.0
    cos_a = dot_aa / n_a if n_a > 1e-12 else 0.0
    
    s_cc = (n_c * cos_c) / sqrt_dh
    s_aa = (n_a * cos_a) / sqrt_dh
    
    s_nc_ca = (n_c * cos_a) / sqrt_dh
    s_na_cc = (n_a * cos_c) / sqrt_dh
    
    qk_score_delta = s_aa - s_cc
    
    # norm_shapley: Norm-product changing while Cosine held constant
    norm_shapley = 0.5 * ((s_na_cc - s_cc) + (s_aa - s_nc_ca))
    # cosine_shapley: Cosine changing while Norm held constant
    cosine_shapley = 0.5 * ((s_nc_ca - s_cc) + (s_aa - s_na_cc))
    
    residual = abs(norm_shapley + cosine_shapley - qk_score_delta)
    return norm_shapley, cosine_shapley, norm_q_c, norm_q_a, norm_k_c, norm_k_a, cos_c, cos_a, residual

def main():
    print("==========================================================================")
    print("PHASE 10: QK NORM/DIRECTION DECOMPOSITION")
    print("==========================================================================")
    
    # Warning guardrail printed explicitly
    print("\n==========================================================================")
    print("WARNING:")
    print("QK decomposition tests whether the IO-target score increase is associated with")
    print("q/k vector changes, norm-product changes, q-k directional alignment, or both.")
    print("It does not by itself prove full output-level causal mediation.")
    print("==========================================================================")
    
    # 1. Load gpt2-small with center_writing_weights=True
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
        
    device = model.cfg.device
    d_head = model.cfg.d_head
    
    # 2. Generate mixed template dataset
    print("\nGenerating mixed template IOI dataset (200 samples)...")
    dataset = IOIDataset(model, num_samples=200, template_style="mixed", seed=42)
    print(f"Dataset generated. Batch size: {dataset.batch_size}, Sequence Length: {dataset.seq_len}")
    
    # Cache filters
    q_hooks = [f"blocks.{l}.attn.hook_q" for l in range(12)]
    k_hooks = [f"blocks.{l}.attn.hook_k" for l in range(12)]
    score_hooks = [f"blocks.{l}.attn.hook_attn_scores" for l in range(12)]
    pattern_hooks = [f"blocks.{l}.attn.hook_pattern" for l in range(12)]
    names_filter = q_hooks + k_hooks + score_hooks + pattern_hooks
    
    # PASS A: Clean cached run
    print("\nExecuting Clean baseline cached forward pass...")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            dataset.clean_tokens,
            names_filter=names_filter
        )
        
    # PASS B: Primary ablated cached run (L9H9 and L10H0 zero-ablated)
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
            
    # PASS C: L9H9-only ablation diagnostic pass
    print("Executing Diagnostic cached forward pass (L9H9-only ablated)...")
    hooks_l9_only = [
        ("blocks.9.attn.hook_z", make_zero_ablation_hook([9]))
    ]
    with model.hooks(fwd_hooks=hooks_l9_only):
        with torch.no_grad():
            _, l9_only_cache = model.run_with_cache(
                dataset.clean_tokens,
                names_filter=names_filter
            )
            
    print("\nProcessing QK decomposition...")
    
    results_raw = []
    results_summary = []
    
    # We will verify Layer 10 hook_z same-layer independence diagnostic
    l10_q_diff_max = 0.0
    l10_k_diff_max = 0.0
    
    for l in range(12):
        for h in range(12):
            head_name = f"L{l}H{h}"
            
            clean_attns = []
            ablated_attns = []
            
            q_norms_clean = []
            q_norms_ablated = []
            q_norms_delta = []
            k_norms_clean = []
            k_norms_ablated = []
            k_norms_delta = []
            
            cosines_clean = []
            cosines_ablated = []
            cosines_delta = []
            
            scores_clean = []
            scores_ablated = []
            scores_delta = []
            
            query_shapleys = []
            key_shapleys = []
            norm_shapleys = []
            cosine_shapleys = []
            
            vector_residuals = []
            geometry_residuals = []
            
            for i in range(dataset.batch_size):
                q_pos = dataset.end[i].item()
                io_pos = dataset.IO[i].item()
                
                # Fetch patterns and scores for validation
                attn_c = clean_cache[f"blocks.{l}.attn.hook_pattern"][i, h, q_pos, io_pos].item()
                attn_a = ablated_cache[f"blocks.{l}.attn.hook_pattern"][i, h, q_pos, io_pos].item()
                
                # Pre-softmax attention scores from cache
                cached_score_clean = clean_cache[f"blocks.{l}.attn.hook_attn_scores"][i, h, q_pos, io_pos].item()
                cached_score_ablated = ablated_cache[f"blocks.{l}.attn.hook_attn_scores"][i, h, q_pos, io_pos].item()
                
                # Extract Q and K vectors
                # Cached hook_q/hook_k shape: [batch_size, seq_len, n_heads, d_head]
                qc = clean_cache[f"blocks.{l}.attn.hook_q"][i, q_pos, h, :]  # shape: [d_head]
                ka = ablated_cache[f"blocks.{l}.attn.hook_k"][i, io_pos, h, :]  # shape: [d_head]
                
                qa = ablated_cache[f"blocks.{l}.attn.hook_q"][i, q_pos, h, :]  # shape: [d_head]
                kc = clean_cache[f"blocks.{l}.attn.hook_k"][i, io_pos, h, :]  # shape: [d_head]
                
                # Vector-level Shapley
                q_shap, k_shap, qk_s_delta, s_cc, s_aa, v_res = compute_shapley_vector(qc, qa, kc, ka, d_head)
                
                # Validation checks: Dot score must match hook_attn_scores within 1e-4
                assert abs(s_cc - cached_score_clean) < 1e-4, f"Clean QK score mismatch at example {i}, {head_name}: {s_cc} vs {cached_score_clean}"
                assert abs(s_aa - cached_score_ablated) < 1e-4, f"Ablated QK score mismatch at example {i}, {head_name}: {s_aa} vs {cached_score_ablated}"
                
                # Geometry-level Shapley
                n_shap, c_shap, q_nc, q_na, k_nc, k_na, cos_c, cos_a, g_res = compute_shapley_geometry(qc, qa, kc, ka, d_head)
                
                q_n_delta = q_na - q_nc
                k_n_delta = k_na - k_nc
                cos_delta = cos_a - cos_c
                
                # Diagnostics: Same-layer hook_z L10H0 independence check
                if l == 10:
                    # Diagnostic check: Extract Q/K vectors from L9H9-only cache to verify independence
                    # hook_q/hook_k shape: [batch_size, seq_len, n_heads, d_head]
                    q_l9 = l9_only_cache[f"blocks.10.attn.hook_q"][i, q_pos, h, :]  # shape: [d_head]
                    k_l9 = l9_only_cache[f"blocks.10.attn.hook_k"][i, io_pos, h, :]  # shape: [d_head]
                    
                    q_diff = torch.max(torch.abs(qa - q_l9)).item()
                    k_diff = torch.max(torch.abs(ka - k_l9)).item()
                    
                    if q_diff > l10_q_diff_max:
                        l10_q_diff_max = q_diff
                    if k_diff > l10_k_diff_max:
                        l10_k_diff_max = k_diff
                
                # Append raw per-example outputs
                results_raw.append({
                    "example_idx": i,
                    "layer": l,
                    "head": h,
                    "head_name": head_name,
                    "clean_attn": attn_c,
                    "ablated_attn": attn_a,
                    "q_norm_clean": q_nc,
                    "q_norm_ablated": q_na,
                    "q_norm_delta": q_n_delta,
                    "k_norm_clean": k_nc,
                    "k_norm_ablated": k_na,
                    "k_norm_delta": k_n_delta,
                    "cosine_clean": cos_c,
                    "cosine_ablated": cos_a,
                    "cosine_delta": cos_delta,
                    "qk_score_clean": s_cc,
                    "qk_score_ablated": s_aa,
                    "qk_score_delta": qk_s_delta,
                    "query_shapley": q_shap,
                    "key_shapley": k_shap,
                    "norm_shapley": n_shap,
                    "cosine_shapley": c_shap,
                    "vector_decomposition_residual": v_res,
                    "geometry_decomposition_residual": g_res
                })
                
                clean_attns.append(attn_c)
                ablated_attns.append(attn_a)
                
                q_norms_clean.append(q_nc)
                q_norms_ablated.append(q_na)
                q_norms_delta.append(q_n_delta)
                k_norms_clean.append(k_nc)
                k_norms_ablated.append(k_na)
                k_norms_delta.append(k_n_delta)
                
                cosines_clean.append(cos_c)
                cosines_ablated.append(cos_a)
                cosines_delta.append(cos_delta)
                
                scores_clean.append(s_cc)
                scores_ablated.append(s_aa)
                scores_delta.append(qk_s_delta)
                
                query_shapleys.append(q_shap)
                key_shapleys.append(k_shap)
                norm_shapleys.append(n_shap)
                cosine_shapleys.append(c_shap)
                
                vector_residuals.append(v_res)
                geometry_residuals.append(g_res)
                
            # Aggregate cohort means safely
            mean_score_delta = np.mean(scores_delta)
            mean_query_shap = np.mean(query_shapleys)
            mean_key_shap = np.mean(key_shapleys)
            mean_norm_shap = np.mean(norm_shapleys)
            mean_cosine_shap = np.mean(cosine_shapleys)
            
            # Safe fraction computation
            if abs(mean_score_delta) > 1e-12:
                q_frac = mean_query_shap / mean_score_delta
                k_frac = mean_key_shap / mean_score_delta
                n_frac = mean_norm_shap / mean_score_delta
                c_frac = mean_cosine_shap / mean_score_delta
            else:
                q_frac, k_frac, n_frac, c_frac = 0.0, 0.0, 0.0, 0.0
                
            results_summary.append({
                "layer": l,
                "head": h,
                "head_name": head_name,
                "clean_attn": np.mean(clean_attns),
                "ablated_attn": np.mean(ablated_attns),
                "q_norm_clean": np.mean(q_norms_clean),
                "q_norm_ablated": np.mean(q_norms_ablated),
                "q_norm_delta": np.mean(q_norms_delta),
                "k_norm_clean": np.mean(k_norms_clean),
                "k_norm_ablated": np.mean(k_norms_ablated),
                "k_norm_delta": np.mean(k_norms_delta),
                "cosine_clean": np.mean(cosines_clean),
                "cosine_ablated": np.mean(cosines_ablated),
                "cosine_delta": np.mean(cosines_delta),
                "qk_score_clean": np.mean(scores_clean),
                "qk_score_ablated": np.mean(scores_ablated),
                "qk_score_delta": mean_score_delta,
                "query_shapley": mean_query_shap,
                "key_shapley": mean_key_shap,
                "norm_shapley": mean_norm_shap,
                "cosine_shapley": mean_cosine_shap,
                "query_shapley_fraction_of_delta": q_frac,
                "key_shapley_fraction_of_delta": k_frac,
                "norm_shapley_fraction_of_delta": n_frac,
                "cosine_shapley_fraction_of_delta": c_frac,
                "std_qk_score_delta": np.std(scores_delta),
                "frac_positive_qk_score_delta": np.mean([1.0 if d > 0 else 0.0 for d in scores_delta]),
                "std_cosine_delta": np.std(cosines_delta),
                "frac_positive_cosine_delta": np.mean([1.0 if d > 0 else 0.0 for d in cosines_delta]),
                "frac_positive_norm_shapley": np.mean([1.0 if d > 0 else 0.0 for d in norm_shapleys]),
                "frac_positive_cosine_shapley": np.mean([1.0 if d > 0 else 0.0 for d in cosine_shapleys]),
                "vector_decomposition_residual_max_abs": np.max(vector_residuals),
                "geometry_decomposition_residual_max_abs": np.max(geometry_residuals)
            })
            
    # Save outputs
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    
    df_raw = pd.DataFrame(results_raw)
    df_raw.to_csv("results/raw/qk_decomposition_raw.csv", index=False)
    with open("results/raw/qk_decomposition_raw.json", "w") as f_out:
        json.dump(results_raw, f_out, indent=4)
        
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv("results/summary/qk_decomposition_summary.csv", index=False)
    with open("results/summary/qk_decomposition_summary.json", "w") as f_out:
        json.dump(results_summary, f_out, indent=4)
        
    print("✓ Saved raw QK decomposition results to results/raw/qk_decomposition_raw.csv and results/raw/qk_decomposition_raw.json")
    print("✓ Saved summary QK decomposition results to results/summary/qk_decomposition_summary.csv and results/summary/qk_decomposition_summary.json")
    
    # Validation checks
    print("\n==========================================================================")
    print("VALIDATION CHECKS:")
    print("==========================================================================")
    
    # 1. Look up L10H10 summary row
    l10h10_row = next(r for r in results_summary if r["head_name"] == "L10H10")
    print(f"L10H10 Clean Attn: {l10h10_row['clean_attn']:.4f} (Expected ≈ 0.3779)")
    print(f"L10H10 Ablated Attn: {l10h10_row['ablated_attn']:.4f} (Expected ≈ 0.5672)")
    
    assert abs(l10h10_row['clean_attn'] - 0.3779) < 0.01, "L10H10 clean attention mismatch"
    assert abs(l10h10_row['ablated_attn'] - 0.5672) < 0.01, "L10H10 ablated attention mismatch"
    
    # 2. Verify score delta and cosine delta are reported positive for L10H10
    print(f"L10H10 qk_score_delta: {l10h10_row['qk_score_delta']:.4f} (Expected > 0)")
    print(f"L10H10 cosine_delta: {l10h10_row['cosine_delta']:.4f}")
    assert l10h10_row['qk_score_delta'] > 0.0, "L10H10 QK score delta is not positive!"
    
    # 3. Sum check: max residuals across all heads
    max_vector_res = max(r["vector_decomposition_residual_max_abs"] for r in results_summary)
    max_geom_res = max(r["geometry_decomposition_residual_max_abs"] for r in results_summary)
    print(f"Max vector Shapley residual: {max_vector_res:.2e} (Expected < 1e-6)")
    print(f"Max geometry Shapley residual: {max_geom_res:.2e} (Expected < 1e-6)")
    
    assert max_vector_res < 1e-6, f"Vector Shapley residual check failed: {max_vector_res}"
    assert max_geom_res < 1e-6, f"Geometry Shapley residual check failed: {max_geom_res}"
    
    # 4. Diagnostic same-layer independence check
    print(f"Max same-layer Q vector difference (Co-Ablation vs L9H9-only): {l10_q_diff_max:.2e}")
    print(f"Max same-layer K vector difference (Co-Ablation vs L9H9-only): {l10_k_diff_max:.2e}")
    assert l10_q_diff_max < 1e-6, "Query same-layer independence check failed!"
    assert l10_k_diff_max < 1e-6, "Key same-layer independence check failed!"
    print("✓ Diagnostic confirms same-layer hook_z L10H0 ablation does not affect Layer 10 QK inputs.")
    print("✓ All validation assertions successfully verified.")
    
    # 5. Print formatted markdown table for Layer 10 heads
    print("\n==========================================================================")
    print("LAYER 10 HEADS PATH-STABLE QK SHAPLEY DECOMPOSITION SUMMARY")
    print("==========================================================================")
    
    headers = [
        "Head", "q-Norm Cln", "q-Norm Abl", "k-Norm Cln", "k-Norm Abl",
        "Cos Clean", "Cos Abl", "Cos Delta", "Score Cln", "Score Abl",
        "Score Del", "Cos-Shap", "Cos-Shap%"
    ]
    print("| " + " | ".join(f"{h:<10}" for h in headers) + " |")
    print("|" + "|".join("-" * 12 for _ in headers) + "|")
    
    for r in results_summary:
        if r["layer"] == 10:
            cos_pct = (r["cosine_shapley"] / r["qk_score_delta"] * 100) if r["qk_score_delta"] != 0 else 0.0
            highlight = " ★" if r["head_name"] in ["L10H10", "L10H2", "L10H3", "L10H6"] else ""
            print(
                f"| {r['head_name'] + highlight:<10} "
                f"| {r['q_norm_clean']:<10.4f} "
                f"| {r['q_norm_ablated']:<10.4f} "
                f"| {r['k_norm_clean']:<10.4f} "
                f"| {r['k_norm_ablated']:<10.4f} "
                f"| {r['cosine_clean']:<10.4f} "
                f"| {r['cosine_ablated']:<10.4f} "
                f"| {r['cosine_delta']:<10.4f} "
                f"| {r['qk_score_clean']:<10.4f} "
                f"| {r['qk_score_ablated']:<10.4f} "
                f"| {r['qk_score_delta']:<10.4f} "
                f"| {r['cosine_shapley']:<10.4f} "
                f"| {cos_pct:<9.1f}% |"
            )
            
    print("==========================================================================")
    print("★ Highlights canonical backup or secondary heads of interest.")
    print("==========================================================================")
    print("PHASE 10: QK NORM/DIRECTION DECOMPOSITION COMPLETE.")
    print("==========================================================================")

if __name__ == "__main__":
    main()
