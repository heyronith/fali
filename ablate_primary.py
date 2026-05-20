import torch
import sys
from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset

def get_head_attention_to_io(cache, layer: int, head: int, dataset: IOIDataset) -> torch.Tensor:
    """
    Safely extracts attention probability from 'end' token to 'IO' token for a given head.
    Ensures all coordinate tensors are explicitly cast to the pattern device.
    
    Args:
        cache: The activation cache from model.run_with_cache().
        layer: The attention layer index (0-indexed).
        head: The attention head index (0-indexed).
        dataset: The IOIDataset instance holding prompts and token indices.
        
    Returns:
        attn_values: A PyTorch tensor of shape [batch_size] containing attention probabilities.
    """
    # Extract post-softmax attention pattern for the layer
    # Shape: [batch, n_heads, seq_len, seq_len]
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    device = pattern.device
    
    # Cast all indexing coordinates to pattern.device to prevent device mismatches
    batch_idx = torch.arange(dataset.batch_size, device=device)
    end_idx = dataset.end.to(device)
    io_idx = dataset.IO.to(device)
    
    # Extract attention values from 'end' to 'IO' for the specified head using advanced diagonal indexing
    # Shape of output: [batch]
    attn_values = pattern[batch_idx, head, end_idx, io_idx]
    
    return attn_values

def make_ablation_hook(head_idx: int):
    """
    Creates an activation interception hook that surgically zero-ablates a specific attention head
    at blocks.{l}.attn.hook_z.
    
    Args:
        head_idx: The 0-indexed attention head to ablate.
        
    Returns:
        ablation_hook: The HookFunction compatible with TransformerLens.
    """
    def ablation_hook(z: torch.Tensor, hook) -> torch.Tensor:
        # z shape: [batch, seq_len, n_heads, d_head]
        # Zero-out the target head in-place along the head dimension
        z[:, :, head_idx, :] = 0.0
        return z
    return ablation_hook

def compute_logit_difference(logits: torch.Tensor, dataset: IOIDataset) -> torch.Tensor:
    """
    Computes the logit difference (IO - S) at the final 'end' token position for a batch.
    
    Args:
        logits: Output logits of shape [batch, seq_len, d_vocab]
        dataset: The IOIDataset instance.
        
    Returns:
        logit_diffs: A tensor of shape [batch] containing logit differences.
    """
    batch_size = dataset.batch_size
    # Extract logits at final token position (index 'end')
    # Shape: [batch, d_vocab]
    logits_at_end = torch.stack([logits[i, dataset.end[i].item(), :] for i in range(batch_size)])
    
    # Extract logits for correct IO and incorrect S tokens
    # Shape: [batch]
    io_logits = logits_at_end[torch.arange(batch_size), dataset.io_token_ids]
    s_logits = logits_at_end[torch.arange(batch_size), dataset.s_token_ids]
    
    # Shape: [batch]
    logit_diffs = io_logits - s_logits
    return logit_diffs

def main():
    print("==================================================")
    print("PHASE 3: STANDARD UPSTREAM HEAD ABLATION & HYDRA EFFECT")
    print("==================================================")
    
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
        
    # 2. Generate a batch of clean prompts
    print("\nGenerating IOI dataset (10 samples, template style A)...")
    dataset = IOIDataset(model, num_samples=10, template_style="A", seed=42)
    print(f"Dataset generated. Batch: {dataset.batch_size}, Sequence length: {dataset.seq_len}")
    
    # 3. RUN 1: Sequential clean baseline pass
    print("\nExecuting sequential clean baseline pass...")
    with torch.no_grad():
        # clean_logits shape: [batch, seq_len, d_vocab]
        clean_logits, clean_cache = model.run_with_cache(dataset.clean_tokens)
        
    clean_logit_diffs = compute_logit_difference(clean_logits, dataset)
    mean_clean_logit_diff = clean_logit_diffs.mean().item()
    print(f"✓ Clean Baseline Mean Logit Difference: {mean_clean_logit_diff:.4f}")
    
    # 4. RUN 2: Upstream ablated pass
    print("\nRegistering ablation hooks for L9H9 and L10H0...")
    # Register hooks simultaneously using list of (name, hook_fn) tuples in model.hooks context manager
    ablation_hooks = [
        ("blocks.9.attn.hook_z", make_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_ablation_hook(0))
    ]
    
    print("Executing forward pass under Primary Name Mover ablation...")
    with model.hooks(fwd_hooks=ablation_hooks):
        with torch.no_grad():
            # ablated_logits shape: [batch, seq_len, d_vocab]
            ablated_logits, ablated_cache = model.run_with_cache(dataset.clean_tokens)
            
    print("✓ Ablated run cached and hooks detached successfully.")
    
    # 5. Compute ablated run Performance Metric (Logit Delta at 'end' token)
    ablated_logit_diffs = compute_logit_difference(ablated_logits, dataset)
    mean_ablated_logit_diff = ablated_logit_diffs.mean().item()
    print(f"✓ Ablated Run Mean Logit Difference: {mean_ablated_logit_diff:.4f}")
    
    # Assertion 1: Verify that ablated performance drops below clean baseline
    assert mean_ablated_logit_diff < mean_clean_logit_diff, (
        f"Assertion Failed: Ablated logit difference ({mean_ablated_logit_diff:.4f}) "
        f"did not drop below clean baseline ({mean_clean_logit_diff:.4f})."
    )
    print("✓ Assertion Passed: Upstream ablation degraded model performance as expected.")
    
    # 6. Extract baseline and ablated attention values for Backup Name Movers
    backup_heads = [
        {"layer": 10, "head": 10, "role": "Backup Name Mover"},
        {"layer": 11, "head": 10, "role": "Backup Name Mover"}
    ]
    
    telemetry_results = []
    
    for head_info in backup_heads:
        layer = head_info["layer"]
        head = head_info["head"]
        role = head_info["role"]
        head_name = f"L{layer}H{head}"
        
        # Extract attention probabilities under clean baseline
        # Shape: [batch]
        clean_attn = get_head_attention_to_io(clean_cache, layer, head, dataset)
        
        # Extract attention probabilities under ablated conditions
        # Shape: [batch]
        ablated_attn = get_head_attention_to_io(ablated_cache, layer, head, dataset)
        
        mean_clean_attn = clean_attn.mean().item()
        mean_ablated_attn = ablated_attn.mean().item()
        
        # Log difference to measure the Hydra spike!
        attn_spike = mean_ablated_attn - mean_clean_attn
        
        telemetry_results.append({
            "name": head_name,
            "role": role,
            "mean_clean": mean_clean_attn,
            "mean_ablated": mean_ablated_attn,
            "spike": attn_spike
        })
        
    # 7. Output a side-by-side comparative Markdown table
    print("\n" + "=" * 90)
    print("COMPARATIVE METRICS JUXTAPOSITION: CLEAN VS. ABLATED (THE HYDRA EFFECT)")
    print("=" * 90)
    
    print("\nMODEL PERFORMANCE (LOGIT DIFFERENCE):")
    print(f"| {'Metric':<25} | {'Clean Baseline':<16} | {'Ablated Run':<16} | {'Absolute Change':<16} |")
    print(f"|{'-' * 27}|{'-' * 18}|{'-' * 18}|{'-' * 18}|")
    logit_change = mean_ablated_logit_diff - mean_clean_logit_diff
    print(f"| {'Mean Logit Diff (IO - S)':<25} | {mean_clean_logit_diff:<16.6f} | {mean_ablated_logit_diff:<16.6f} | {logit_change:<16.6f} |")
    
    print("\nBACKUP HEADS ATTENTION TO IO:")
    print(f"| {'Backup Head':<12} | {'Clean Baseline':<16} | {'Ablated Run':<16} | {'Hydra Spike (Δ)':<16} |")
    print(f"|{'-' * 14}|{'-' * 18}|{'-' * 18}|{'-' * 18}|")
    for r in telemetry_results:
        print(f"| {r['name']:<12} | {r['mean_clean']:<16.6f} | {r['mean_ablated']:<16.6f} | {r['spike']:<16.6f} |")
    print("=" * 90)
    
    # Explain baseline Hydra Effect
    print("\nCausal Hydra Effect Analysis:")
    for r in telemetry_results:
        if r["spike"] > 0:
            pct_increase = (r["spike"] / r["mean_clean"]) * 100
            print(f"  ✓ {r['name']} exhibits a positive Hydra spike: attention to IO increased by +{r['spike']:.4f} ({pct_increase:.1f}% relative change).")
        else:
            print(f"  ⚠ {r['name']} did not show a positive spike under ablation: change is {r['spike']:.4f}.")
            
    print("\nAll causal ablation tests completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
