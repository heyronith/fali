import torch
import sys
import os
from transformer_lens import HookedTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ioi_dataset import IOIDataset

def main():
    print("==================================================")
    print("PHASE 1: ENVIRONMENT SETUP & IOI DATASET VERIFICATION")
    print("==================================================")
    
    # 1. Load the model
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
        
    # Check the center_writing_weights config setting
    config_cww = getattr(model.cfg, "center_writing_weights", None)
    print(f"model.cfg.center_writing_weights: {config_cww}")
    
    # 2. Instantiate the dataset
    print("\nGenerating IOI dataset (10 samples, template style A)...")
    try:
        dataset = IOIDataset(model, num_samples=10, template_style="A", seed=42)
        print(f"Dataset generated successfully. Batch size: {dataset.batch_size}, Sequence length: {dataset.seq_len}")
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Assertions and structural validations
    print("\nRunning structural assertions...")
    
    # Assert token shapes
    assert dataset.clean_tokens.shape == (10, dataset.seq_len), "clean_tokens shape mismatch"
    assert dataset.corrupted_tokens.shape == (10, dataset.seq_len), "corrupted_tokens shape mismatch"
    print(f"✓ Shape validation passed: clean_tokens and corrupted_tokens are shape {list(dataset.clean_tokens.shape)}")
    
    # Assert integer indexes
    assert dataset.S1.dtype == torch.long, "S1 indices must be torch.long"
    assert dataset.IO.dtype == torch.long, "IO indices must be torch.long"
    assert dataset.S2.dtype == torch.long, "S2 indices must be torch.long"
    assert dataset.end.dtype == torch.long, "end indices must be torch.long"
    print("✓ Type validation passed: index tensors are torch.long")
    
    # Verify index mapping correctness
    for i in range(dataset.batch_size):
        clean_toks = dataset.clean_tokens[i]
        corr_toks = dataset.corrupted_tokens[i]
        
        s_id = dataset.s_token_ids[i].item()
        io_id = dataset.io_token_ids[i].item()
        c_io_id = dataset.corrupted_token_ids[i].item()
        
        # Verify token contents at indices
        assert clean_toks[dataset.S1[i]].item() == s_id, f"Sample {i}: S1 index mismatch"
        assert clean_toks[dataset.S2[i]].item() == s_id, f"Sample {i}: S2 index mismatch"
        assert clean_toks[dataset.IO[i]].item() == io_id, f"Sample {i}: IO index mismatch"
        
        assert corr_toks[dataset.S1[i]].item() == s_id, f"Sample {i}: Corrupted S1 index mismatch"
        assert corr_toks[dataset.S2[i]].item() == s_id, f"Sample {i}: Corrupted S2 index mismatch"
        assert corr_toks[dataset.IO[i]].item() == c_io_id, f"Sample {i}: Corrupted IO index mismatch"
        
        assert clean_toks[dataset.end[i]].item() == model.tokenizer.encode(" to")[0], f"Sample {i}: End index token is not 'to'"
        
    print("✓ Token value mapping checks passed: name tokens correspond exactly to S1, IO, S2, and end indices.")

    # 4. Perform baseline clean forward pass and compute performance metrics
    print("\nRunning baseline forward pass...")
    with torch.no_grad():
        # clean_logits shape: [batch, seq_len, d_vocab]
        clean_logits = model(dataset.clean_tokens)
        
        # corrupted_logits shape: [batch, seq_len, d_vocab]
        corrupted_logits = model(dataset.corrupted_tokens)
        
    # We want to measure the logits at the final token position (index 'end')
    # and compare the logit for the correct IO token vs the subject S token.
    # clean_logits_at_end shape: [batch, d_vocab]
    clean_logits_at_end = torch.stack([clean_logits[i, dataset.end[i].item(), :] for i in range(dataset.batch_size)])
    
    # Extract logits for correct IO and incorrect S tokens
    # shape: [batch]
    io_logits = clean_logits_at_end[torch.arange(dataset.batch_size), dataset.io_token_ids]
    s_logits = clean_logits_at_end[torch.arange(dataset.batch_size), dataset.s_token_ids]
    
    # Compute logit difference
    # shape: [batch]
    logit_diffs = io_logits - s_logits
    mean_logit_diff = logit_diffs.mean().item()
    accuracy = (logit_diffs > 0).float().mean().item() * 100
    
    print(f"Mean logit difference (IO - S) on clean prompts: {mean_logit_diff:.4f}")
    print(f"IO prediction accuracy: {accuracy:.1f}%")
    
    # 5. Print out structured visualization of the dataset
    print("\n" + "=" * 80)
    print("DATASET VISUALIZATION TABLE")
    print("=" * 80)
    print(f"{'No.':<4} | {'Clean Prompt':<60} | {'S1':<2} | {'IO':<2} | {'S2':<2} | {'end':<3} | {'IO Logit Diff':<8}")
    print("-" * 105)
    for i in range(min(5, dataset.batch_size)):
        prompt = dataset.clean_prompts[i]
        # Truncate prompt string if it's too long for visual display
        display_prompt = prompt if len(prompt) < 58 else prompt[:55] + "..."
        s1 = dataset.S1[i].item()
        io = dataset.IO[i].item()
        s2 = dataset.S2[i].item()
        end = dataset.end[i].item()
        ld = logit_diffs[i].item()
        print(f"{i+1:<4} | {display_prompt:<60} | {s1:<2} | {io:<2} | {s2:<2} | {end:<3} | {ld:<8.4f}")
    print("=" * 80)
    
    print("\nAll setup verifications and assertions passed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
