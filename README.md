# When Attention Repairs Itself: A Mechanistic Study of Backup Heads in GPT-2

This repository contains the complete, reproducible research pipeline for reverse-engineering downstream "backup" self-repair behavior (the **Hydra Effect**) in transformer attention circuits under the ablation of primary name movers. 

Our pipeline rigorously evaluates the causal, geometric, and sequential triggers of self-repair in **GPT-2 Small**, establishing that the compensation is a physically real, causally contributing, query-driven algorithmic response rather than a passive normalization artifact or specific to the zero-ablation protocol.

---

## 🎯 Project Goal

The primary goal of this project is to resolve the causal origin of the Hydra Effect—the observed increase in attention allocated by downstream backup attention heads (e.g., `L10H10`) to a target token (the Indirect Object `IO`) after primary heads (e.g., `L9H9`, `L10H0`) are ablated. We disentangle the **Target-Logit attention-logit contribution** from the **Background-Logit attention-logit contribution** using our path-stable Frozen Attention Logit Intervention (FALI) Shapley decomposition, and separately characterize physical **LayerNorm Scale-Drop Amplification** across a rigorous suite of ablation controls (zero, position-wise mean, and resampling), geometric Query-Key alignment tracking, and output-level causal mediation patching.

---

## 💻 Technical Stack & Core Setup

- **Language & Core**: Python 3.9+, PyTorch
- **Interpretability Framework**: TransformerLens (activation caching, hook-based counterfactual patching)
- **Tensor Mathematics**: `einops` and `torch.einsum` for high-dimensional representation geometry tracking
- **Model**: `gpt2-small` loaded with `center_writing_weights=True` to improve residual stream projection interpretability
- **Dataset**: A balanced cohort of $N=200$ Indirect Object Identification (IOI) templates (100 Template A, 100 Template B, seed 42)

### Dependency Installation

To install all dependencies and set up the local virtual environment:

```bash
# 1. Create a local virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Full Pipeline

To run the entire 12-step mechanistic interpretability pipeline end-to-end and perform the automatic audit check:

```bash
.venv/bin/python run_all_experiments.py
```

---

## 🌀 The 12-Step Experiment Pipeline

The master runner (`run_all_experiments.py`) executes the following 12 steps sequentially:

1. **Verify Setup & Dataset** (`experiments/verify_setup.py`): Validates PyTorch environment, loads the model, and checks that the $N=200$ IOI dataset meets expected tokenization and shape constraints.
2. **Cache Clean Baselines** (`experiments/cache_baseline.py`): Performs a clean forward pass caching all activations to establish baseline attention and logit difference scores.
3. **Ablate Primary Mover Heads** (`experiments/ablate_primary.py`): Zero-ablates `L9H9` and `L10H0` at their output `hook_z` vector, establishing basic causal degradation baselines.
4. **FALI Hybrid Intervention** (`experiments/hybrid_intervention.py`): Performs the target-logit counterfactual intervention at the sub-softmax level to mathematically separate target-directed active logit shifts from global softmax shifts.
5. **LayerNorm-Frozen Validation** (`experiments/fali_rigorous_validation.py`): Implements LayerNorm-scale freezing in the forward pass to causally measure the contribution of the LayerNorm volume surge.
6. **Model-Wide Scale Evaluation** (`experiments/scale_evaluation.py`): Systematically runs FALI across all 144 attention heads in GPT-2 Small to generate a model-wide target-logit vs. background-logit compensation map.
7. **Causal Double Ablation** (`experiments/causal_validation.py`): Implements a four-pass causal protocol (Pass A: Clean, Pass B: Primary Ablated, Pass C: Double Ablated including `L10H10`, Pass D: `L10H10` Only ablated) to isolate `L10H10`'s compensatory contribution.
8. **Ablation Controls** (`experiments/ablation_controls.py`): Compares the robustness of the self-repair attention spike and logit degradation under zero ablation, position-wise mean ablation, and resample ablation using a fixed derangement.
9. **Path-Stable FALI Decomposition** (`experiments/fali_decomposition.py`): Computes target-logit and background-logit Shapley-style decomposition of attention probability shifts to establish path-independent contributions.
10. **QK Decomposition** (`experiments/qk_decomposition.py`): Projects attention scores into Query norm, Key norm, and directional/cosine alignment components using high-dimensional representation geometry.
11. **Output Mediation** (`experiments/output_mediation.py`): Measures the downstream causal contribution of the ablated-run `L10H10` attention pattern vs. the ablated-run `L10H10` $z$-vector output in recovering the final $IO - S$ logit difference.
12. **Generate Analytical Figures** (`experiments/generate_new_figures.py`): Renders publication-ready lollipop charts, trajectory plots, disentanglement scatter plots, and trigger isolation bar charts.

---

## 📁 Results Folder Directory Map

The pipeline generates and populates the following directory structure:

- **`results/raw/`**: Detailed per-example CSV and JSON files storing high-dimensional activations, raw FALI attention deltas, raw QK norm/cosine scores, and per-token logit differences.
  - `ablation_controls_raw.json` / `ablation_controls_raw.csv`
  - `fali_decomposition_raw.json` / `fali_decomposition_raw.csv`
  - `qk_decomposition_raw.json` / `qk_decomposition_raw.csv`
  - `output_mediation_raw.json` / `output_mediation_raw.csv`
  - `fali_raw_results.json` / `fali_raw_results.csv`
  - `ln_frozen_raw_results.json` / `ln_frozen_raw_results.csv`
- **`results/summary/`**: Aggregated means, standard deviations, confidence intervals, and fractions of positive effects across the cohort.
  - `ablation_controls_summary.json` / `ablation_controls_summary.csv`
  - `fali_decomposition_summary.json` / `fali_decomposition_summary.csv`
  - `qk_decomposition_summary.json` / `qk_decomposition_summary.csv`
  - `output_mediation_summary.json` / `output_mediation_summary.csv`
  - `output_mediation_controls.json` / `output_mediation_controls.csv`
  - `attention_results.json`
  - `fali_rigorous_validation.csv`
  - `scale_evaluation_results.json`
- **`figures/`**: Diagnostic visualizations.
  - `cossim_shift_lollipop.png`: QK cosine similarity shifts across Layer 10 heads showing magnitude-independent directional realignment.
  - `causal_shift_plot.png`: Trajectory plots of clean vs. ablated attention probabilities for Layers 9--11.
  - `circuit_disentanglement_plot.png`: Scatter plot mapping standard attention spikes against FALI target-logit Shapley contributions, showing that they plot above the diagonal.
  - `isolation_bar_chart.png`: Trigger isolation bar chart comparing sequential upstream and intra-layer ablations.

---

## 🔍 Robust Causal Evidence Summary

### 1. Ablation Controls
The downstream compensation of the primary backup head `L10H10` is robust and survives beyond zero ablation. Under zero, mean, and resample ablations, the attention spike to $IO$ remains positive in $91.5\%$--$100\%$ of cases, and the causal degradation ($IO - S$ logit drop when `L10H10` is additionally ablated) remains stable.
- **Attention Spike ($\Delta_\text{Attn}$)**: $+0.1893$ (Zero), $+0.1616$ (Mean), $+0.1224$ (Resample).
- **Causal Logit Degradation**: $0.2281$ (Zero), $0.2048$ (Mean), $0.2435$ (Resample).

### 2. Path-Stable FALI / Shapley Decomposition
Attention probability shifts are dominated by target-logit pre-softmax score changes rather than background denominator inflation. The path-independent Shapley decomposition of L10H10's $+0.1893$ attention spike shows:
- **Target-Logit Shapley Contribution**: $+0.2222$ (explaining **$117.4\%$** of the total shift).
- **Background-Logit Shapley Contribution**: $-0.0328$ (**$-17.4\%$** fraction). The background logits actually oppose the attention spike on average in $96.0\%$ of samples, indicating that background-logit changes oppose the IO attention increase on average.
- **Decomposition Path Stability**: High path-independence. The absolute target-background disagreement between target-first and background-first paths is just $0.00153$ ($0.69\%$ relative to the Shapley effect).

### 3. QK Vector & Geometry Decomposition
The target-logit score increase ($+0.9632$ score delta) is query-driven and alignment-driven:
- **Query Shapley Contribution**: $+0.9770$ (**$101.4\%$** of the score delta), reflecting query vector rotation.
- **Key Shapley Contribution**: $-0.0137$ (**$-1.4\%$** of the delta).
- **Directional Cosine Shapley**: $+0.7418$ (**$77.0\%$** of the delta), indicating directional realignment in QK space.
- **Norm Shapley**: $+0.2214$ (**$23.0\%$** of the delta).

### 4. Output-Level Causal Mediation
Downstream logit recovery is causally mediated by the backup head's output. When `L9H9` and `L10H0` are ablated, the $IO - S$ logit difference drops from $+2.9732$ to $+2.1055$ (a degradation of $0.8677$). Additionally zeroing `L10H10` drops the performance further to $+1.8775$ (a causal degradation of $0.2281$).
- **Attention-Pattern Necessity**: Patching clean `L10H10` pattern into the ablated run explains **$43.6\%$** of this degradation (aggregate pattern necessity effect of $0.0994$).
- **$z$-Vector Output Necessity**: Patching clean `L10H10` $z$-vector output into the ablated run explains **$44.0\%$** of this degradation (aggregate $z$ necessity effect of $0.1002$).
- **Control Head Comparison**: In contrast, non-backup heads such as `L10H5` show no such mediation effects, establishing the structural specificity of `L10H10`.

---

## 📌 Strongest Qualified Claim

In GPT-2 Small on a 200-example mixed-template IOI cohort, ablating L9H9 and L10H0 induces a robust L10H10 increase in IO-directed attention. This effect survives zero, mean, and resample ablation controls. Path-stable FALI/Shapley decomposition shows that the attention increase is dominated by the IO-target attention-logit contribution, while background-logit changes oppose the spike on average. QK decomposition suggests the IO-target score increase is primarily query-driven and largely cosine-alignment-driven. Output mediation shows that the L10H10 attention-pattern change causally contributes to IO-S logit recovery under this intervention, explaining about 43.6% of L10H10’s zero-ablation degradation by aggregate ratio of means.

---

## 🛡 Limitations

1. **Single Architecture**: Evaluation is limited to a single 12-layer model: **GPT-2 Small**.
2. **Single Task Family**: Circuit tracking is restricted to the **Indirect Object Identification (IOI)** name-copying benchmark.
3. **Cohort Size**: The empirical evaluation is performed on a cohort of $N=200$ mixed-template prompts.
4. **Primary Ablation Focus**: Interventions focus specifically on the ablation of `L9H9` and `L10H0`.
5. **Intervention Specificity**: Causal claims are specific to this particular hook-patching setup. While the zero, mean, and resampling ablation controls confirm robust compensation across different baselines, they do not mathematically establish a general backup mechanism across arbitrary component failures.
6. **No Direct V/OV Path Patching**: Direct path patching of the Value ($V$) or Output-Value ($OV$) circuit remains future work.
7. **Model-Family Generalization**: Testing generalization to other transformer families (e.g., Pythia, Gemma, LLaMA) and broader task domains (e.g., factual recall, translation) remains future work.

---

## 📊 Claims, Evidence, and Strength Matrix

| Claim | Key Causal Evidence | Empirical Strength | Structural Limitation |
| :--- | :--- | :--- | :--- |
| **Robust Downstream Attention Spike** | `L10H10` attention spike is robust across zero (`+0.1893`), mean (`+0.1616`), and resample (`+0.1224`) ablation paradigms, occurring in $91.5\%$--$100\%$ of cohort samples. | **Strong but intervention-specific** | Evaluated on $N=200$ mixed-templates in GPT-2 Small only. |
| **Target-Logit Dominance** | Shapley shows target pre-softmax shift is $+0.2222$ ($117.4\%$ fraction) while background opposes the spike ($-17.4\%$ fraction, opposing in $96.0\%$ of samples). Absolute path disagreement is low ($0.69\%$). | **Strong in this setting** (Pre-softmax mathematical bound) | Dependent on correct token alignment in the query/key projection. |
| **Query-Driven Alignment Realignment** | QK score delta ($+0.9632$) is driven by query Shapley ($+0.9770$ / $101.4\%$ fraction) and directional Q-K cosine alignment realignment ($+0.7418$ / $77.0\%$ fraction). | **Strong in this setting** (Magnitude-independent cosine shift) | Focused on `IO` key position and `end` query position. |
| **Causal Downstream Rescue** | Double ablation causes $0.2281$ logit drop. L10H10 pattern/z necessity explain $43.6\%$ / $44.0\%$ of the degradation; L10H2, L10H3, and L10H6 are additional contributing Layer 10 heads; L10H5 is a low-effect control; L10H7 is anti-backup/suppressor-like under this intervention. | **Strong** (Causally verified via multi-pass hook patching) | Value/OV direct path patching was not performed; remaining degradation (~56%) is mediated by other backup heads. |
| **Cross-Layer Sequential Trigger** | Upstream L9H9 ablation alone reproduces the full L10H10 spike ($+0.1893$), while intra-layer partner L10H0 ablation produces zero change ($0.0000$), showing that L10H10’s attention-pattern shift is triggered by upstream L9H9 ablation, not by same-layer L10H0 z-ablation. Because same-layer attention heads compute in parallel, the absence of a same-layer L10H0 z-ablation effect is partly architecturally expected. | **Strong but intervention-specific** (Surgical double/single causal isolation) | Specific to the sequential hierarchy of Layer 9 and Layer 10. |

