import torch
import sys
import os
from transformer_lens import HookedTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ioi_dataset import IOIDataset

def compute_logit_difference(logits: torch.Tensor, dataset: IOIDataset) -> float:
    """
    Computes the mean logit difference (IO - S) at the final 'end' token position for a batch.
    """
    batch_size = dataset.batch_size
    # Extract logits at final token position (index 'end' which is padding-aware)
    logits_at_end = torch.stack([logits[i, dataset.end[i].item(), :] for i in range(batch_size)])
    
    # Extract logits for correct IO and incorrect S tokens
    io_logits = logits_at_end[torch.arange(batch_size), dataset.io_token_ids]
    s_logits = logits_at_end[torch.arange(batch_size), dataset.s_token_ids]
    
    logit_diffs = io_logits - s_logits
    return logit_diffs.mean().item()

def make_safe_ablation_hook(head_idx):
    """
    Creates a clone-safe activation interception hook that surgically zero-ablates 
    a specific attention head at blocks.{l}.attn.hook_z.
    """
    def hook_fn(z, hook):
        # z shape: [batch, seq_len, n_heads, d_head]
        z_cloned = z.clone()
        z_cloned[:, :, head_idx, :] = 0.0
        return z_cloned
    return hook_fn

def main():
    print("==========================================================================")
    print("PHASE 6: CAUSAL LOGIT RECUPERATION TEST (DOUBLE ABLATION)")
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
        
    # 2. Generate a large cohort of 200 samples with mixed Template Style A and Template Style B
    print("\nGenerating mixed template IOI dataset (200 samples, 50/50 alternating style A and B)...")
    dataset = IOIDataset(model, num_samples=200, template_style="mixed", seed=42)
    print(f"Dataset generated. Batch size: {dataset.batch_size}, Max Sequence Length: {dataset.seq_len}")
    
    # PASS A: Clean Baseline
    print("\nExecuting PASS A: Clean Baseline pass...")
    with torch.no_grad():
        clean_logits = model(dataset.clean_tokens)
    clean_logit_diff = compute_logit_difference(clean_logits, dataset)
    print(f"✓ Clean Baseline Mean Logit Difference: {clean_logit_diff:.4f}")
    
    # PASS B: Primary Ablation (L9H9, L10H0)
    print("\nExecuting PASS B: Primary Ablation pass (L9H9, L10H0)...")
    primary_ablation_hooks = [
        ("blocks.9.attn.hook_z", make_safe_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_safe_ablation_hook(0))
    ]
    with model.hooks(fwd_hooks=primary_ablation_hooks):
        with torch.no_grad():
            primary_ablated_logits = model(dataset.clean_tokens)
    primary_ablated_logit_diff = compute_logit_difference(primary_ablated_logits, dataset)
    print(f"✓ Primary Ablated Mean Logit Difference: {primary_ablated_logit_diff:.4f}")
    
    # PASS C: Double Ablation (L9H9, L10H0, L10H10)
    print("\nExecuting PASS C: Double Ablation pass (L9H9, L10H0, L10H10)...")
    double_ablation_hooks = [
        ("blocks.9.attn.hook_z", make_safe_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_safe_ablation_hook(0)),
        ("blocks.10.attn.hook_z", make_safe_ablation_hook(10))
    ]
    with model.hooks(fwd_hooks=double_ablation_hooks):
        with torch.no_grad():
            double_ablated_logits = model(dataset.clean_tokens)
    double_ablated_logit_diff = compute_logit_difference(double_ablated_logits, dataset)
    print(f"✓ Double Ablated Mean Logit Difference: {double_ablated_logit_diff:.4f}")
    
    # PASS D: L10H10-Only Ablation
    print("\nExecuting PASS D: L10H10-Only Ablation pass...")
    l10h10_only_hooks = [
        ("blocks.10.attn.hook_z", make_safe_ablation_hook(10))
    ]
    with model.hooks(fwd_hooks=l10h10_only_hooks):
        with torch.no_grad():
            pass_d_logits = model(dataset.clean_tokens)
    pass_d_logit_diff = compute_logit_difference(pass_d_logits, dataset)
    print(f"✓ L10H10-Only Ablated Mean Logit Difference: {pass_d_logit_diff:.4f}")
    
    print("\n" + "=" * 80)
    print("LOGIT RECUPERATION RESULTS:")
    print("=" * 80)
    print(f"PASS A (Clean Baseline):         {clean_logit_diff:.4f}")
    print(f"PASS B (Primary Ablation):       {primary_ablated_logit_diff:.4f}")
    print(f"PASS C (Double Ablation):        {double_ablated_logit_diff:.4f}")
    print(f"PASS D (L10H10-Only Ablation):   {pass_d_logit_diff:.4f}")
    
    # Assertions
    assert primary_ablated_logit_diff < clean_logit_diff, "Primary ablation did not degrade logit performance."
    assert double_ablated_logit_diff < primary_ablated_logit_diff, "Double ablation did not further degrade logit performance."
    print("\n✓ Causal assertions passed: Double ablation of the backup head successfully disrupted the self-repair mechanism.")
    print("==========================================================================")

if __name__ == "__main__":
    main()
