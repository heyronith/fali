"""
hybrid_intervention.py

PHASE 4: Counterfactual "Frozen Attention Logit" Intervention
This script implements a surgical causal intervention targeting the Backup Name Mover heads
(L10H10 and L11H10) to disentangle active algorithmic routing from passive softmax 
denominator shifts (the Hydra Effect normalisation artifact).

Scientific Context:
- Standard Ablation: Zero-ablating Primary Name Mover heads (L9H9, L10H0) results in a dramatic 
  increase in attention allocation from the 'end' token to the 'IO' token in Backup Name Mover heads.
- The Question: Is this spike driven by active compensation (routing of logits)? Or is it a 
  mathematical artifact of softmax (since ablating primary heads decreases background token logits, 
  making the IO token's relative share larger when normalized via softmax)?
- The Intervention: By pinning all background token pre-softmax attention logits to their clean 
  baseline values while allowing only the dynamic ablated IO logit to flow through, we can compute 
  a counterfactual attention probability. If the counterfactual attention remains baseline-like, 
  the Hydra spike is an active routing change. If it is ablated-like, it is a softmax artifact.
"""

import torch
import sys
import json
from typing import List, Dict
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
    # In gpt2-small: [10, 12, 15, 15]
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    device = pattern.device
    
    # Cast all indexing coordinates to pattern.device to prevent device mismatches
    batch_idx = torch.arange(dataset.batch_size, device=device) # Shape: [batch] (length 10)
    end_idx = dataset.end.to(device) # Shape: [batch] (length 10)
    io_idx = dataset.IO.to(device) # Shape: [batch] (length 10)
    
    # Extract attention values from 'end' to 'IO' for the specified head using advanced diagonal indexing
    # Shape of output: [batch] (length 10)
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
        # In gpt2-small hook_z shape: [10, 15, 12, 64]
        # Zero-out the target head in-place along the head dimension
        z[:, :, head_idx, :] = 0.0
        return z
    return ablation_hook

def make_hybrid_logit_hook(clean_scores: torch.Tensor, head_idx: int, dataset: IOIDataset):
    """
    Surgically pins background token logits to clean baseline values while 
    allowing the dynamic, ablated logit for the target IO token to pass through.
    
    Args:
        clean_scores: Pre-softmax attention logit scores from clean baseline pass
                      Shape: [batch, num_heads, q_seq_len, k_seq_len] (e.g. [10, 12, 15, 15])
        head_idx: The target attention head index (0-indexed) to intervene on.
        dataset: The IOIDataset instance holding prompts and token indices.
        
    Returns:
        hybrid_hook: The HookFunction compatible with TransformerLens.
    """
    def hybrid_hook(ablated_scores: torch.Tensor, hook) -> torch.Tensor:
        # ablated_scores shape: [batch, num_heads, q_seq_len, k_seq_len]
        # In gpt2-small hook_attn_scores shape: [10, 12, 15, 15]
        device = ablated_scores.device
        
        # Prepare coordinate tensors with explicit casting to avoid device mismatches
        batch_idx = torch.arange(dataset.batch_size, device=device) # Shape: [batch] (length 10)
        end_pos = dataset.end.to(device) # Shape: [batch] (length 10)
        io_pos = dataset.IO.to(device) # Shape: [batch] (length 10)
        
        # 1. Extract and save the dynamically computed ablated logit for the IO token
        # Dynamic logit shape: [batch] (length 10)
        dynamic_io_logits = ablated_scores[batch_idx, head_idx, end_pos, io_pos]
        
        # 2. Overwrite the entire query row at 'end' position with the clean cached baseline scores
        clean_scores_dev = clean_scores.to(device) # Shape: [batch, num_heads, q_seq_len, k_seq_len]
        
        # Overwrite the slice at [batch_idx, head_idx, end_pos, :]
        # Slice shape: [batch, k_seq_len] (length [10, 15])
        ablated_scores[batch_idx, head_idx, end_pos, :] = clean_scores_dev[batch_idx, head_idx, end_pos, :]
        
        # 3. Surgically re-inject only the dynamic ablated logit for the target IO token back into that row
        # Slice shape: [batch] (length 10)
        ablated_scores[batch_idx, head_idx, end_pos, io_pos] = dynamic_io_logits
        
        return ablated_scores
    return hybrid_hook

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
    print("==========================================================================")
    print("PHASE 4: COUNTERFACTUAL 'FROZEN ATTENTION LOGIT' INTERVENTION")
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
        
    # 2. Generate a batch of clean prompts
    print("\nGenerating IOI dataset (10 samples, template style A)...")
    dataset = IOIDataset(model, num_samples=10, template_style="A", seed=42)
    print(f"Dataset generated. Batch: {dataset.batch_size}, Sequence length: {dataset.seq_len}")
    
    # 3. PASS 1: Clean Baseline Pass
    print("\nExecuting PASS 1: Clean Baseline Pass...")
    with torch.no_grad():
        # clean_logits shape: [batch, seq_len, d_vocab]
        clean_logits, clean_cache = model.run_with_cache(dataset.clean_tokens)
        
    clean_logit_diffs = compute_logit_difference(clean_logits, dataset)
    mean_clean_logit_diff = clean_logit_diffs.mean().item()
    print(f"✓ Clean Baseline Mean Logit Difference: {mean_clean_logit_diff:.4f}")
    
    # Extract clean baseline pre-softmax attention logit scores for L10H10 and L11H10
    # Shape: [batch, num_heads, q_seq_len, k_seq_len] (e.g. [10, 12, 15, 15])
    clean_scores_l10 = clean_cache["blocks.10.attn.hook_attn_scores"]
    clean_scores_l11 = clean_cache["blocks.11.attn.hook_attn_scores"]
    
    # 4. PASS 2: Standard Ablation Pass (Task 3 replication)
    print("\nExecuting PASS 2: Standard Upstream Ablation Pass...")
    ablation_hooks = [
        ("blocks.9.attn.hook_z", make_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_ablation_hook(0))
    ]
    
    with model.hooks(fwd_hooks=ablation_hooks):
        with torch.no_grad():
            # ablated_logits shape: [batch, seq_len, d_vocab]
            ablated_logits, ablated_cache = model.run_with_cache(dataset.clean_tokens)
            
    ablated_logit_diffs = compute_logit_difference(ablated_logits, dataset)
    mean_ablated_logit_diff = ablated_logit_diffs.mean().item()
    print(f"✓ Standard Ablation Mean Logit Difference: {mean_ablated_logit_diff:.4f}")
    
    # 5. PASS 3: Hybrid Counterfactual Intervention Pass
    print("\nExecuting PASS 3: Simultaneous Upstream Ablation + Downstream Hybrid Intervention...")
    # Simultaneous interventions:
    # 1. Upstream ablation hooks on blocks.9.attn.hook_z and blocks.10.attn.hook_z.
    # 2. Hybrid intervention hooks on blocks.10.attn.hook_attn_scores and blocks.11.attn.hook_attn_scores.
    hybrid_hooks = [
        ("blocks.9.attn.hook_z", make_ablation_hook(9)),
        ("blocks.10.attn.hook_z", make_ablation_hook(0)),
        ("blocks.10.attn.hook_attn_scores", make_hybrid_logit_hook(clean_scores_l10, 10, dataset)),
        ("blocks.11.attn.hook_attn_scores", make_hybrid_logit_hook(clean_scores_l11, 10, dataset))
    ]
    
    with model.hooks(fwd_hooks=hybrid_hooks):
        with torch.no_grad():
            # hybrid_logits shape: [batch, seq_len, d_vocab]
            hybrid_logits, hybrid_cache = model.run_with_cache(dataset.clean_tokens)
            
    hybrid_logit_diffs = compute_logit_difference(hybrid_logits, dataset)
    mean_hybrid_logit_diff = hybrid_logit_diffs.mean().item()
    print(f"✓ Hybrid Intervention Mean Logit Difference: {mean_hybrid_logit_diff:.4f}")
    
    # 6. Extract target attention metrics across three conditions
    backup_heads = [
        {"layer": 10, "head": 10, "name": "L10H10"},
        {"layer": 11, "head": 10, "name": "L11H10"}
    ]
    
    results = {}
    json_export = {}
    
    for head_info in backup_heads:
        layer = head_info["layer"]
        head = head_info["head"]
        name = head_info["name"]
        
        # Extract attention tensors under the three states
        # Shape: [batch] (length 10)
        clean_attn = get_head_attention_to_io(clean_cache, layer, head, dataset)
        ablated_attn = get_head_attention_to_io(ablated_cache, layer, head, dataset)
        hybrid_attn = get_head_attention_to_io(hybrid_cache, layer, head, dataset)
        
        # Shape and bounding assertions as required by the verification plan
        assert list(hybrid_attn.shape) == [dataset.batch_size], (
            f"Assertion Failed: Hybrid attention tensor for {name} has incorrect shape "
            f"{list(hybrid_attn.shape)}, expected [{dataset.batch_size}]."
        )
        
        # Assert mathematical validity: values must lie strictly in range [0.0, 1.0] (with tolerance)
        assert (hybrid_attn >= -1e-7).all() and (hybrid_attn <= 1.0 + 1e-7).all(), (
            f"Assertion Failed: Counterfactual hybrid attention probabilities for {name} out of bounds: "
            f"min={hybrid_attn.min().item():.6f}, max={hybrid_attn.max().item():.6f}."
        )
        
        # Calculate mean values for reporting
        mean_clean = clean_attn.mean().item()
        mean_ablated = ablated_attn.mean().item()
        mean_hybrid = hybrid_attn.mean().item()
        
        results[name] = {
            "clean": mean_clean,
            "ablated": mean_ablated,
            "hybrid": mean_hybrid,
            "hydra_spike": mean_ablated - mean_clean,
            "hybrid_spike": mean_hybrid - mean_clean,
            "artifact_ratio": (mean_ablated - mean_hybrid) / (mean_ablated - mean_clean) if abs(mean_ablated - mean_clean) > 1e-7 else 0.0
        }
        
        # Save raw values to export dictionary
        json_export[name] = {
            "clean": clean_attn.tolist(),
            "ablated": ablated_attn.tolist(),
            "hybrid": hybrid_attn.tolist()
        }
        
    print("✓ All tensor shapes are structurally valid and execute without conflicts.")
    print("✓ All hybrid attention values lie strictly bounded between 0.0 and 1.0.")
    
    # Export results to JSON
    json_path = "attention_results.json"
    try:
        with open(json_path, "w") as f:
            json.dump(json_export, f, indent=4)
        print(f"✓ Successfully exported raw three-tier attention arrays to {json_path}")
    except Exception as e:
        print(f"⚠ Failed to export results to JSON: {e}")
        
    # 7. Print comparative Markdown evaluation tables
    print("\n" + "=" * 115)
    print("THREE-TIER EVALUATION JUXTAPOSITION: DISENTANGLING ALGORITHMIC ROUTING VS. SOFTMAX ARTIFACT")
    print("=" * 115)
    
    print("\nMODEL PERFORMANCE (LOGIT DIFFERENCE):")
    print(f"| {'Metric':<25} | {'Clean Baseline':<16} | {'Standard Ablation':<18} | {'Hybrid Intervention':<20} |")
    print(f"|{'-' * 27}|{'-' * 18}|{'-' * 20}|{'-' * 22}|")
    print(f"| {'Mean Logit Diff (IO - S)':<25} | {mean_clean_logit_diff:<16.6f} | {mean_ablated_logit_diff:<18.6f} | {mean_hybrid_logit_diff:<20.6f} |")
    
    print("\nBACKUP HEADS ATTENTION TO IO:")
    print(f"| {'Head':<8} | {'Clean Baseline':<16} | {'Standard Ablation':<18} | {'Hybrid Intervention':<20} | {'Hydra Spike (Δ)':<16} | {'Hybrid Spike (Δ)':<17} |")
    print(f"|{'-' * 10}|{'-' * 18}|{'-' * 20}|{'-' * 22}|{'-' * 18}|{'-' * 19}|")
    for name, r in results.items():
        print(f"| {name:<8} | {r['clean']:<16.6f} | {r['ablated']:<18.6f} | {r['hybrid']:<20.6f} | {r['hydra_spike']:<16.6f} | {r['hybrid_spike']:<17.6f} |")
    print("=" * 115)
    
    print("\nScientific Breakdown and Isolation Analysis:")
    for name, r in results.items():
        spike = r["hydra_spike"]
        hybrid_spike = r["hybrid_spike"]
        ratio = r["artifact_ratio"]
        
        if abs(spike) > 1e-4:
            print(f"\n  • Head {name}:")
            print(f"    - Standard Hydra Spike (Ablated - Clean): {spike:+.6f}")
            print(f"    - Active Hybrid Spike (Hybrid - Clean):   {hybrid_spike:+.6f}")
            print(f"    - Softmax Denominator Artifact Contribution: {ratio * 100:.2f}%")
            print(f"    - Active Algorithmic Routing Contribution:   {(1.0 - ratio) * 100:.2f}%")
            if ratio > 0.5:
                print(f"    [Result] This backup head's self-repair spike is PRIMARILY a passive softmax denominator shift.")
            else:
                print(f"    [Result] This backup head's self-repair spike is PRIMARILY an active algorithmic routing change.")
        else:
            print(f"\n  • Head {name} did not exhibit a significant Hydra spike (change is {spike:+.6f}). Skipping isolation analysis.")
            
    print("\nAll Phase 4 counterfactual experiments completed successfully!")
    print("==========================================================================")

if __name__ == "__main__":
    main()
