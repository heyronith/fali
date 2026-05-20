Task 1: Environment Setup & IOI Dataset Pipeline
Goal: Initialize the model and generate the evaluation dataset.
Implementation Details:
Load gpt2-small using transformer_lens.HookedTransformer with center_writing=True.
Implement an Indirect Object Identification (IOI) dataset generator using standard templates (e.g., "When [NameA] and [NameB] went to the store, [NameA] gave an object to [NameB]").
Hardcode the known layer/head indexes for Primary Name Mover Heads (e.g., Layer 9 Head 9, Layer 10 Head 0) and Backup Name Mover Heads (e.g., Layer 10 Head 10, Layer 11 Head 10) to serve as tracking targets.
Task 2: Clean Run Activation Caching & Metrics Setup
Goal: Establish baseline clean activations and target metrics.
Implementation Details:
Run the model on clean IOI prompts using model.run_with_cache().
Cache both the pre-softmax attention logits (blocks.{l}.attn.hook_attn_scores) and post-softmax attention patterns (blocks.{l}.attn.hook_pattern) for all layers.
Define two evaluation metrics:
Model Performance: Logit difference between the correct Indirect Object (IO) token and the Subject (S) token at the final token position.
Target Direct Metric: The attention probability allocated by the Backup Name Mover Heads to the IO token position.
Task 3: Standard Upstream Head Ablation (Baseline Hydra Effect)
Goal: Replicate and quantify the standard baseline "self-repair" spike.
Implementation Details:
Write a TransformerLens hook function to perform zero-ablation (or mean-ablation) on the output of the Primary Name Mover Heads (blocks.{l}.hook_resid_post or via hook_z manipulation).
Execute the forward pass with this ablation hook active.
Measure and record the drop in overall model logit difference and the corresponding baseline increase (the Hydra spike) in the Backup Name Mover Heads' attention probability toward the IO token.
Task 4: Counterfactual "Frozen Attention Logit" Hook Intervention
Goal: Mathematically isolate the token-level softmax denominator shift from active logit changes.
Implementation Details:
Create a custom intervention hook targeting the Backup Name Mover Heads at blocks.{l}.attn.hook_attn_scores.
During the ablated run forward pass, intercept the dynamic pre-softmax logit tensor (QK 
T
 / 
d 
k
​	
 

​	
 ).
Construct a hybrid logit tensor: Keep the raw, dynamically computed logit for the target IO token from the ablated run, but overwrite the logits of all non-target tokens with their original values cached from the clean run.
Pass this hybrid logit tensor through the standard softmax operation to generate a counterfactual attention pattern.
Task 5: Metric Evaluation, Isolation Analysis, and Reporting
Goal: Quantify the exact percentage of the Hydra Effect caused by the softmax artifact.
Implementation Details:
Run a batch execution comparing three conditions: Clean Run, Standard Ablation Run, and Hybrid Logit Intervention Run.
Calculate the attention probability assigned to the IO token under all three conditions.
Compute the artifact ratio:  
Attention 
Ablated
​	
 −Attention 
Clean
​	
 
Attention 
Hybrid
​	
 −Attention 
Clean
​	
 
​	
 .
Save results to a structured CSV/JSON file containing head-by-head breakdowns of how much of the backup spike is a passive normalization artifact versus active structural routing.