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
    # Extract the post-softmax attention pattern for the layer
    # Shape: [batch, n_heads, seq_len, seq_len]
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    device = pattern.device
    
    # Cast all indexing coordinates to pattern.device to prevent device mismatches
    batch_idx = torch.arange(dataset.batch_size, device=device)
    # Shape of dataset.end and dataset.IO is [batch]
    end_idx = dataset.end.to(device)
    io_idx = dataset.IO.to(device)
    
    # Extract attention values from 'end' to 'IO' for the specified head using advanced diagonal indexing
    # Shape of output: [batch]
    attn_values = pattern[batch_idx, head, end_idx, io_idx]
    
    return attn_values

def main():
    print("==================================================")
    print("PHASE 2: CLEAN RUN ACTIVATION CACHING & METRICS SETUP")
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
    
    # 3. Running forward pass with activation caching
    print("\nExecuting forward pass with run_with_cache()...")
    with torch.no_grad():
        # logits shape: [batch, seq_len, d_vocab]
        # cache holds all internal activations of the model
        logits, cache = model.run_with_cache(dataset.clean_tokens)
        
    print("✓ Activations cached successfully.")
    
    # 4. Compute baseline Model Performance Metric (Logit Delta at 'end' token)
    # Extract logits at final token position (index 'end')
    # shape: [batch, d_vocab]
    logits_at_end = torch.stack([logits[i, dataset.end[i].item(), :] for i in range(dataset.batch_size)])
    
    # Extract logits for correct IO and incorrect S tokens
    # shape: [batch]
    io_logits = logits_at_end[torch.arange(dataset.batch_size), dataset.io_token_ids]
    s_logits = logits_at_end[torch.arange(dataset.batch_size), dataset.s_token_ids]
    
    # Compute logit difference
    # shape: [batch]
    logit_diffs = io_logits - s_logits
    mean_logit_diff = logit_diffs.mean().item()
    print(f"\nBaseline Clean Mean Logit Difference: {mean_logit_diff:.4f}")
    
    # Assertion 1: Dynamic relative check (ensure basic performance exceeds random chance)
    assert mean_logit_diff > 1.0, (
        f"Assertion Failed: Baseline logit difference ({mean_logit_diff:.4f}) "
        f"is suspiciously low, expected model to perform well above chance."
    )
    print("✓ Assertion Passed: Baseline logit difference indicates strong model performance.")
    
    # 5. Extract attention values for Primary and Backup Name Movers
    # Hardcoded tracking targets for Primary and Backup Name Movers
    primary_heads = [(9, 9), (10, 0)]
    backup_heads = [(10, 10), (11, 10)]
    
    target_heads = [
        {"layer": 9, "head": 9, "role": "Primary Name Mover"},
        {"layer": 10, "head": 0, "role": "Primary Name Mover"},
        {"layer": 10, "head": 10, "role": "Backup Name Mover"},
        {"layer": 11, "head": 10, "role": "Backup Name Mover"}
    ]
    
    print("\nExtracting and verifying target head attention values...")
    results = []
    
    for head_info in target_heads:
        layer = head_info["layer"]
        head = head_info["head"]
        role = head_info["role"]
        head_name = f"L{layer}H{head}"
        
        # Safely extract attention probabilities
        # Shape: [batch_size]
        attn_values = get_head_attention_to_io(cache, layer, head, dataset)
        
        # Assertion 2: Shape must be exactly [batch_size]
        assert list(attn_values.shape) == [dataset.batch_size], (
            f"Assertion Failed: Head {head_name} returned shape {list(attn_values.shape)}, "
            f"expected [{dataset.batch_size}]."
        )
        
        # Assertion 3: All attention probabilities must lie strictly in [0.0, 1.0]
        # (Using float tolerance to avoid absolute precision bugs with CUDA/MPS float conversions)
        assert (attn_values >= -1e-7).all() and (attn_values <= 1.0 + 1e-7).all(), (
            f"Assertion Failed: Head {head_name} attention probabilities out of bounds: "
            f"min={attn_values.min().item():.6f}, max={attn_values.max().item():.6f}."
        )
        
        mean_attn = attn_values.mean().item()
        min_attn = attn_values.min().item()
        max_attn = attn_values.max().item()
        
        results.append({
            "name": head_name,
            "role": role,
            "mean": mean_attn,
            "min": min_attn,
            "max": max_attn
        })
        
    print("✓ All shape and bounding assertions passed for target heads.")
    
    # 6. Output beautiful terminal summary markdown table
    print("\n" + "=" * 80)
    print("BASELINE TARGET HEADS ATTENTION TO IO POSITION")
    print("=" * 80)
    print(f"| {'Head':<6} | {'Role':<20} | {'Mean Attn':<10} | {'Min Attn':<10} | {'Max Attn':<10} |")
    print(f"|{'-' * 8}|{'-' * 22}|{'-' * 12}|{'-' * 12}|{'-' * 12}|")
    for r in results:
        print(f"| {r['name']:<6} | {r['role']:<20} | {r['mean']:<10.6f} | {r['min']:<10.6f} | {r['max']:<10.6f} |")
    print("=" * 80)
    
    # Sanity checks based on expected baseline behavior
    print("\nRunning heuristic validation checks on baseline dynamics:")
    for r in results:
        if r["role"] == "Primary Name Mover":
            if r["mean"] > 0.15:
                print(f"  ✓ {r['name']} ({r['role']}) behaves as expected: High baseline attention to IO ({r['mean']:.4f}).")
            else:
                print(f"  ⚠ {r['name']} ({r['role']}) has lower attention than typical ({r['mean']:.4f}).")
        elif r["role"] == "Backup Name Mover":
            if r["mean"] < 0.08:
                print(f"  ✓ {r['name']} ({r['role']}) behaves as expected: Inactive baseline attention to IO ({r['mean']:.4f}).")
            else:
                print(f"  ⚠ {r['name']} ({r['role']}) has higher baseline attention than expected ({r['mean']:.4f}).")
                
    print("\nAll baseline caching checks completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
