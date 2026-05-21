"""
fali_rigorous_validation.py

Rigorous mechanistic interpretability script validating active self-repair in GPT-2 Small.
Implements:
1. A LayerNorm-Frozen Counterfactual Interception.
2. Direct Query-Key Cosine Similarity and Dot-Product geometric tracking.
3. Isolated sequential ablation runs (L9H9 vs. L10H0).
4. Paired two-tailed t-tests with 95% confidence intervals across our 200-sample cohort.

Author: Antigravity Agent (.agents/rules/resesrcher.md compliant)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy import stats
from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset

def compute_logit_difference(logits: torch.Tensor, dataset: IOIDataset) -> torch.Tensor:
    """
    Computes the logit difference (IO - S) at the final 'end' token position for a batch.
    Shape input logits: [batch, seq_len, d_vocab]
    Shape output: [batch]
    """
    batch_size = dataset.batch_size
    logits_at_end = torch.stack([logits[i, dataset.end[i].item(), :] for i in range(batch_size)])
    io_logits = logits_at_end[torch.arange(batch_size), dataset.io_token_ids]
    s_logits = logits_at_end[torch.arange(batch_size), dataset.s_token_ids]
    return io_logits - s_logits

def make_safe_ablation_hook(head_idx: int):
    """
    Creates a clone-safe activation interception hook that zero-ablates a specific attention head
    at blocks.{l}.attn.hook_z.
    """
    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        # z shape: [batch, seq_len, n_heads, d_head]
        z_cloned = z.clone()
        z_cloned[:, :, head_idx, :] = 0.0
        return z_cloned
    return hook_fn

def main():
    print("==========================================================================")
    print("FALI RIGOROUS CAUSAL VALIDATION pipeline")
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

    # 2. Generate a cohort of 200 samples with mixed template styles
    print("\nGenerating mixed template IOI dataset (200 samples)...")
    dataset = IOIDataset(model, num_samples=200, template_style="mixed", seed=42)
    print(f"Dataset generated. Batch size: {dataset.batch_size}, Sequence Length: {dataset.seq_len}")

    # Extract coordinates
    device = model.cfg.device
    batch_idx = torch.arange(dataset.batch_size, device=device) # [batch]
    end_idx = dataset.end.to(device) # [batch]
    io_idx = dataset.IO.to(device) # [batch]

    # Let's perform forward runs and cache activations

    # PASS 1: Clean Baseline Pass
    print("\nExecuting Pass 1: Clean Baseline cached forward pass...")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(dataset.clean_tokens)
    clean_logit_diffs = compute_logit_difference(clean_logits, dataset)
    print(f"✓ Clean Baseline Mean Logit Difference: {clean_logit_diffs.mean().item():.4f}")

    # Cache clean LayerNorm scale for layer 10
    # Shape of blocks.10.ln1.hook_scale: [batch, seq_len, 1]
    clean_scale = clean_cache["blocks.10.ln1.hook_scale"].clone()
    print(f"Cached blocks.10.ln1.hook_scale with shape {clean_scale.shape}")

    # PASS 2: Standard Ablation (L9H9 and L10H0 zero-ablated)
    print("\nExecuting Pass 2: Standard Ablation cached forward pass (L9H9, L10H0)...")
    primary_ablation_hooks = [
        ("blocks.9.attn.hook_z", make_safe_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_safe_ablation_hook(0))
    ]
    with model.hooks(fwd_hooks=primary_ablation_hooks):
        with torch.no_grad():
            ablated_logits, ablated_cache = model.run_with_cache(dataset.clean_tokens)
    ablated_logit_diffs = compute_logit_difference(ablated_logits, dataset)
    print(f"✓ Standard Ablated Mean Logit Difference: {ablated_logit_diffs.mean().item():.4f}")

    # PASS 3: LayerNorm-Frozen Counterfactual Interception
    # We hook blocks.10.ln1.hook_scale and force it to be the cached clean scale tensor.
    # While still zero-ablating L9H9 and L10H0.
    def make_scale_interception_hook(cached_scale_tensor: torch.Tensor):
        def hook_fn(scale: torch.Tensor, hook) -> torch.Tensor:
            # scale shape: [batch, seq_len, 1]
            return cached_scale_tensor.clone()
        return hook_fn

    ln_frozen_hooks = [
        ("blocks.9.attn.hook_z", make_safe_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_safe_ablation_hook(0)),
        ("blocks.10.ln1.hook_scale", make_scale_interception_hook(clean_scale))
    ]
    print("\nExecuting Pass 3: LayerNorm-Frozen Counterfactual Interception...")
    with model.hooks(fwd_hooks=ln_frozen_hooks):
        with torch.no_grad():
            ln_frozen_logits, ln_frozen_cache = model.run_with_cache(dataset.clean_tokens)
    ln_frozen_logit_diffs = compute_logit_difference(ln_frozen_logits, dataset)
    print(f"✓ LN-Frozen Mean Logit Difference: {ln_frozen_logit_diffs.mean().item():.4f}")


    # STEP 2: Geometric Tracking for Head L10H10
    print("\nRunning STEP 2: Geometric Tracking for Head L10H10...")

    # We want Query vector at the final token position for head 10: Shape: [batch, d_head]
    # And Key vector at target IO token position for head 10: Shape: [batch, d_head]
    # In TransformerLens, "q" or "k" in cache has shape: [batch, seq_len, n_heads, d_head]
    
    # Extract clean vectors
    q_clean = clean_cache["q", 10][batch_idx, end_idx, 10, :]   # [batch, d_head]
    k_clean = clean_cache["k", 10][batch_idx, io_idx, 10, :]    # [batch, d_head]

    # Extract ablated vectors
    q_abl = ablated_cache["q", 10][batch_idx, end_idx, 10, :]   # [batch, d_head]
    k_abl = ablated_cache["k", 10][batch_idx, io_idx, 10, :]    # [batch, d_head]

    # Extract LN-frozen vectors
    q_ln = ln_frozen_cache["q", 10][batch_idx, end_idx, 10, :]  # [batch, d_head]
    k_ln = ln_frozen_cache["k", 10][batch_idx, io_idx, 10, :]   # [batch, d_head]

    # Assert shape is correct: [200, 64]
    assert q_clean.shape == (200, 64), f"Incorrect Query shape: {q_clean.shape}"
    assert k_clean.shape == (200, 64), f"Incorrect Key shape: {k_clean.shape}"

    def compute_metrics(q: torch.Tensor, k: torch.Tensor):
        # Dot product across the batch: [batch]
        dot_product = (q * k).sum(dim=-1)
        # Norms
        q_norm = torch.linalg.norm(q, dim=-1)
        k_norm = torch.linalg.norm(k, dim=-1)
        # Cosine similarity
        cos_sim = dot_product / (q_norm * k_norm)
        return dot_product, cos_sim

    clean_dot, clean_cossim = compute_metrics(q_clean, k_clean)
    abl_dot, abl_cossim = compute_metrics(q_abl, k_abl)
    ln_dot, ln_cossim = compute_metrics(q_ln, k_ln)

    print(f"Geometric Metrics (L10H10):")
    print(f"  Clean:     Dot Product = {clean_dot.mean().item():.4f}, CosSim = {clean_cossim.mean().item():.4f}")
    print(f"  Ablated:   Dot Product = {abl_dot.mean().item():.4f}, CosSim = {abl_cossim.mean().item():.4f}")
    print(f"  LN-Frozen: Dot Product = {ln_dot.mean().item():.4f}, CosSim = {ln_cossim.mean().item():.4f}")


    # STEP 3: Sequential Layer Isolation
    print("\nRunning STEP 3: Sequential Layer Isolation (Condition Alpha vs Beta vs Gamma)...")
    
    # Condition Alpha: Zero-ablate L9H9 only.
    alpha_hooks = [("blocks.9.attn.hook_z", make_safe_ablation_hook(9))]
    with model.hooks(fwd_hooks=alpha_hooks):
        with torch.no_grad():
            _, alpha_cache = model.run_with_cache(dataset.clean_tokens)
    
    # Condition Beta: Zero-ablate L10H0 only.
    beta_hooks = [("blocks.10.attn.hook_z", make_safe_ablation_hook(0))]
    with model.hooks(fwd_hooks=beta_hooks):
        with torch.no_grad():
            _, beta_cache = model.run_with_cache(dataset.clean_tokens)

    # Condition Gamma: Co-ablation of both (which is already Pass 2 'ablated_cache')
    gamma_cache = ablated_cache

    # Retrieve L10H10 attention probability to IO
    # Shape of blocks.10.attn.hook_pattern: [batch, n_heads, q_seq_len, k_seq_len]
    # We want token position `end`'s allocation to token position `IO` for head 10.
    clean_attn = clean_cache["blocks.10.attn.hook_pattern"][batch_idx, 10, end_idx, io_idx]
    abl_attn = ablated_cache["blocks.10.attn.hook_pattern"][batch_idx, 10, end_idx, io_idx]
    ln_frozen_attn = ln_frozen_cache["blocks.10.attn.hook_pattern"][batch_idx, 10, end_idx, io_idx]
    
    alpha_attn = alpha_cache["blocks.10.attn.hook_pattern"][batch_idx, 10, end_idx, io_idx]
    beta_attn = beta_cache["blocks.10.attn.hook_pattern"][batch_idx, 10, end_idx, io_idx]

    print(f"L10H10 Attention to IO under Isolation Conditions:")
    print(f"  Clean:             {clean_attn.mean().item():.4f}")
    print(f"  Condition Alpha (L9H9 Abl): {alpha_attn.mean().item():.4f} (Delta = {(alpha_attn - clean_attn).mean().item():.4f})")
    print(f"  Condition Beta (L10H0 Abl):  {beta_attn.mean().item():.4f} (Delta = {(beta_attn - clean_attn).mean().item():.4f})")
    print(f"  Condition Gamma (Co-Ablation): {abl_attn.mean().item():.4f} (Delta = {(abl_attn - clean_attn).mean().item():.4f})")
    print(f"  LayerNorm-Frozen Interception: {ln_frozen_attn.mean().item():.4f} (Delta = {(ln_frozen_attn - clean_attn).mean().item():.4f})")


    # STEP 4: Statistical Consolidation
    print("\nRunning STEP 4: Statistical Consolidation (t-tests & 95% Confidence Intervals)...")

    # Shift comparisons
    delta_raw_hydra = (abl_attn - clean_attn).cpu().numpy()
    delta_ln_frozen_hydra = (ln_frozen_attn - clean_attn).cpu().numpy()
    delta_cossim = (abl_cossim - clean_cossim).cpu().numpy()

    def perform_t_test_and_ci(x1, x2, label):
        # Paired two-tailed t-test
        res = stats.ttest_rel(x1, x2)
        diff = x1 - x2
        mean_diff = np.mean(diff)
        sem = stats.sem(diff)
        # 95% Confidence Interval for mean difference
        ci = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=sem)
        print(f"  Paired t-test for {label}:")
        print(f"    Mean Difference = {mean_diff:.5f}")
        print(f"    t-statistic     = {res.statistic:.4f}")
        print(f"    p-value         = {res.pvalue:.4e}")
        print(f"    95% CI          = ({ci[0]:.5f}, {ci[1]:.5f})")
        return res.statistic, res.pvalue, ci

    print("\n1. Standard Hydra Spike vs. LN-Frozen Hydra Spike:")
    t_stat_hydra, p_val_hydra, ci_hydra = perform_t_test_and_ci(delta_raw_hydra, delta_ln_frozen_hydra, "Raw vs. LN-Frozen Attention Delta")

    print("\n2. Cosine Similarity Change (Ablated vs. Clean):")
    # Clean vs ablated cosine similarity across the batch
    res_cossim = stats.ttest_rel(abl_cossim.cpu().numpy(), clean_cossim.cpu().numpy())
    diff_cossim = (abl_cossim - clean_cossim).cpu().numpy()
    ci_cossim = stats.t.interval(0.95, len(diff_cossim)-1, loc=np.mean(diff_cossim), scale=stats.sem(diff_cossim))
    print(f"  Paired t-test for L10H10 Cosine Similarity (Ablated vs. Clean):")
    print(f"    Mean Difference = {np.mean(diff_cossim):.5f}")
    print(f"    t-statistic     = {res_cossim.statistic:.4f}")
    print(f"    p-value         = {res_cossim.pvalue:.4e}")
    print(f"    95% CI          = ({ci_cossim[0]:.5f}, {ci_cossim[1]:.5f})")

    # Let's save a cleanly formatted CSV summary tracking
    # [Head, Clean_Attn, Abl_Attn, LN_Frozen_Attn, Clean_CosSim, Abl_CosSim, P_Value]
    # We will generate this CSV row for the primary head studied L10H10, as well as general heads of interest in Layer 10.
    
    # We will compute these values for all Layer 10 heads to provide a complete validation suite!
    layer_10_rows = []
    
    # Calculate for all Layer 10 heads
    print("\nGenerating metrics for all Layer 10 heads to output CSV...")
    for h in range(12):
        # Extract attention probs
        h_clean_attn = clean_cache["blocks.10.attn.hook_pattern"][batch_idx, h, end_idx, io_idx]
        h_abl_attn = ablated_cache["blocks.10.attn.hook_pattern"][batch_idx, h, end_idx, io_idx]
        h_ln_frozen_attn = ln_frozen_cache["blocks.10.attn.hook_pattern"][batch_idx, h, end_idx, io_idx]
        
        # Query & Key vectors
        h_q_clean = clean_cache["q", 10][batch_idx, end_idx, h, :]
        h_k_clean = clean_cache["k", 10][batch_idx, io_idx, h, :]
        
        h_q_abl = ablated_cache["q", 10][batch_idx, end_idx, h, :]
        h_k_abl = ablated_cache["k", 10][batch_idx, io_idx, h, :]
        
        _, h_clean_cossim = compute_metrics(h_q_clean, h_k_clean)
        _, h_abl_cossim = compute_metrics(h_q_abl, h_k_abl)
        
        # Perform paired t-test for attention difference under standard vs LN-frozen
        h_delta_raw = (h_abl_attn - h_clean_attn).cpu().numpy()
        h_delta_ln = (h_ln_frozen_attn - h_clean_attn).cpu().numpy()
        h_res = stats.ttest_rel(h_delta_raw, h_delta_ln)
        
        layer_10_rows.append({
            "Head": f"L10H{h}",
            "Clean_Attn": h_clean_attn.mean().item(),
            "Abl_Attn": h_abl_attn.mean().item(),
            "LN_Frozen_Attn": h_ln_frozen_attn.mean().item(),
            "Clean_CosSim": h_clean_cossim.mean().item(),
            "Abl_CosSim": h_abl_cossim.mean().item(),
            "P_Value": h_res.pvalue
        })
        
    df = pd.DataFrame(layer_10_rows)
    csv_path = "fali_rigorous_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Cleanly formatted CSV summary study saved to: {csv_path}")
    print("\nCSV Data:\n", df.to_string(index=False))

    print("\n==========================================================================")
    print("FALI RIGOROUS CAUSAL VALIDATION COMPLETE.")
    print("==========================================================================")

if __name__ == "__main__":
    main()
