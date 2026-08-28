"""
EF-1 Level B — tiny random HookedTransformer validation (mechanical, not scientific).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from extended_fali_interventions import (
    FactorState,
    ScorePartition,
    build_batched_score_partition,
    build_score_partition_masks,
    cell_table_from_utility_fn,
    char_fn_from_cell_table,
    masked_softmax,
    partition_diagnostics,
    quotient_h_values_from_cell_table,
    reconstruct_z_from_scores_values,
    score_pattern_z_pipeline,
    validate_score_partition,
)
from extended_fali_math import (
    EXTENDED_FALI_GROUPS,
    EXTENDED_FALI_PLAYERS,
    extended_fali_quotient_shapley,
    owen_value_subset,
)
from revision_config import Head
from revision_factorial import three_factor_shapley
from revision_interventions import build_zero_ablation_hooks

EF1_SEED = 4242
EF1_ARCH = dict(
    n_layers=2,
    n_heads=2,
    d_model=32,
    d_head=16,
    d_mlp=64,
    n_ctx=8,
    d_vocab=48,
)
PRIMARY: Head = (0, 0)
CANDIDATE: Head = (1, 0)
METRIC_POS_A = 10
METRIC_POS_B = 17
LivePerturbKind = Literal["none", "v", "q", "qv"]


@dataclass
class LevelBContext:
    model: HookedTransformer
    tokens: torch.Tensor
    final_query: torch.Tensor  # [batch]
    target_key: torch.Tensor  # [batch]
    valid_length: torch.Tensor  # [batch]
    partition: ScorePartition
    scores_a: torch.Tensor  # [B,Q,K] candidate head, context A
    scores_b: torch.Tensor
    values_a: torch.Tensor  # [B,K,D]
    values_b: torch.Tensor
    pattern_a: torch.Tensor
    pattern_b: torch.Tensor
    z_a: torch.Tensor  # [B,Q,D]
    z_b: torch.Tensor
    primary_hooks: List


def build_tiny_model(seed: int = EF1_SEED) -> HookedTransformer:
    torch.manual_seed(seed)
    cfg = HookedTransformerConfig(
        **EF1_ARCH,
        act_fn="gelu",
        normalization_type="LN",
        device="cpu",
        dtype=torch.float32,
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


def _metric_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Final-position logit difference METRIC_POS_A - METRIC_POS_B."""
    last = logits[:, -1, :]
    return last[:, METRIC_POS_A] - last[:, METRIC_POS_B]


def _cache_candidate(model: HookedTransformer, tokens: torch.Tensor, primary_ablate: bool):
    cand_layer, cand_head = CANDIDATE
    hooks = (
        build_zero_ablation_hooks([PRIMARY], n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads)
        if primary_ablate
        else []
    )
    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(tokens)
    scores = cache[f"blocks.{cand_layer}.attn.hook_attn_scores"][:, cand_head].contiguous()
    pattern = cache[f"blocks.{cand_layer}.attn.hook_pattern"][:, cand_head].contiguous()
    values = cache[f"blocks.{cand_layer}.attn.hook_v"][:, :, cand_head, :].contiguous()
    z = cache[f"blocks.{cand_layer}.attn.hook_z"][:, :, cand_head, :].contiguous()
    return scores, pattern, values, z


def _select_target_keys(
    pattern_a: torch.Tensor,
    final_queries: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Argmax valid attention mass on each example's final query row."""
    batch, _, _ = pattern_a.shape
    if override is not None:
        return override.to(dtype=torch.long, device=pattern_a.device)
    target_keys = torch.empty(batch, dtype=torch.long, device=pattern_a.device)
    for b in range(batch):
        fq = int(final_queries[b].item())
        vl = int(valid_lengths[b].item())
        causal = torch.arange(pattern_a.shape[-1], device=pattern_a.device) <= fq
        valid = causal & (torch.arange(pattern_a.shape[-2], device=pattern_a.device) < vl)
        row = pattern_a[b, fq].clone()
        row = row.masked_fill(~valid[fq], 0.0)
        target_keys[b] = row.argmax()
    return target_keys


def build_level_b_context(
    seed: int = EF1_SEED,
    batch: int = 3,
    *,
    target_key_override: torch.Tensor | None = None,
) -> LevelBContext:
    torch.manual_seed(seed)
    model = build_tiny_model(seed)
    tokens = torch.randint(0, EF1_ARCH["d_vocab"], (batch, 6))
    scores_a, pattern_a, values_a, z_a = _cache_candidate(model, tokens, primary_ablate=False)
    scores_b, pattern_b, values_b, z_b = _cache_candidate(model, tokens, primary_ablate=True)
    seq_len = scores_a.shape[1]
    final_query = torch.full((batch,), seq_len - 1, dtype=torch.long)
    valid_length = final_query + 1
    target_key = _select_target_keys(
        pattern_a, final_query, valid_length, override=target_key_override
    )
    partition = build_batched_score_partition(
        seq_len, final_query, target_key, valid_lengths=valid_length
    )
    validate_score_partition(partition)
    primary_hooks = build_zero_ablation_hooks(
        [PRIMARY], n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads
    )
    return LevelBContext(
        model=model,
        tokens=tokens,
        final_query=final_query,
        target_key=target_key,
        valid_length=valid_length,
        partition=partition,
        scores_a=scores_a,
        scores_b=scores_b,
        values_a=values_a,
        values_b=values_b,
        pattern_a=pattern_a,
        pattern_b=pattern_b,
        z_a=z_a,
        z_b=z_b,
        primary_hooks=primary_hooks,
    )


def batch_partition_report(ctx: LevelBContext) -> Dict:
    diag = partition_diagnostics(ctx.partition)
    return {
        "target_keys": [int(x) for x in ctx.target_key.tolist()],
        "final_queries": [int(x) for x in ctx.final_query.tolist()],
        "valid_lengths": [int(x) for x in ctx.valid_length.tolist()],
        **diag,
    }


def _inject_candidate_z(model: HookedTransformer, z_hybrid: torch.Tensor) -> Tuple[str, callable]:
    layer, head = CANDIDATE

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        out = z.clone()
        out[:, :, head, :] = z_hybrid
        return out

    return f"blocks.{layer}.attn.hook_z", hook_fn


def _zero_candidate_z(model: HookedTransformer) -> Tuple[str, callable]:
    layer, head = CANDIDATE

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        out = z.clone()
        out[:, :, head, :] = 0.0
        return out

    return f"blocks.{layer}.attn.hook_z", hook_fn


def _run_direct(
    ctx: LevelBContext,
    *,
    primary_ablate: bool,
    candidate_zero: bool,
) -> torch.Tensor:
    """Independent forward pass without hybrid-z injection."""
    hooks: List = []
    if primary_ablate:
        hooks.extend(ctx.primary_hooks)
    if candidate_zero:
        hooks.append(_zero_candidate_z(ctx.model))
    with torch.no_grad():
        with ctx.model.hooks(fwd_hooks=hooks):
            logits = ctx.model(ctx.tokens)
    return _metric_from_logits(logits)


def _run_metric(
    ctx: LevelBContext,
    state: FactorState,
    *,
    candidate_zero: bool,
    live_perturbation: LivePerturbKind = "none",
    perturb_magnitude: float = 1e6,
) -> torch.Tensor:
    _, _, z_hybrid = score_pattern_z_pipeline(
        ctx.scores_a,
        ctx.scores_b,
        ctx.values_a,
        ctx.values_b,
        ctx.partition,
        state,
    )
    cand_layer, cand_head = CANDIDATE
    hooks: List = []
    if state[4]:  # context B => primary ablated
        hooks.extend(ctx.primary_hooks)
    if candidate_zero:
        hooks.append(_zero_candidate_z(ctx.model))
    else:
        hooks.append(_inject_candidate_z(ctx.model, z_hybrid))
    if live_perturbation in ("v", "qv"):

        def v_perturb(v: torch.Tensor, hook) -> torch.Tensor:
            out = v.clone()
            out[:, :, cand_head, :] = out[:, :, cand_head, :] + perturb_magnitude
            return out

        hooks.insert(0, (f"blocks.{cand_layer}.attn.hook_v", v_perturb))
    if live_perturbation in ("q", "qv"):

        def q_perturb(q: torch.Tensor, hook) -> torch.Tensor:
            out = q.clone()
            out[:, :, cand_head, :] = out[:, :, cand_head, :] + perturb_magnitude
            return out

        hooks.insert(0, (f"blocks.{cand_layer}.attn.hook_q", q_perturb))
    with torch.no_grad():
        with ctx.model.hooks(fwd_hooks=hooks):
            logits = ctx.model(ctx.tokens)
    return _metric_from_logits(logits)


def utility_fn(ctx: LevelBContext, state: FactorState) -> float:
    injected = _run_metric(ctx, state, candidate_zero=False)
    zeroed = _run_metric(ctx, state, candidate_zero=True)
    return float((injected - zeroed).mean().item())


def reconstruction_errors(ctx: LevelBContext) -> Dict[str, float]:
    part = ctx.partition
    p_from_a = masked_softmax(
        ctx.scores_a.masked_fill(~part.valid_mask, float("-inf")), part.valid_mask
    )
    p_from_b = masked_softmax(
        ctx.scores_b.masked_fill(~part.valid_mask, float("-inf")), part.valid_mask
    )
    z_from_a = reconstruct_z_from_scores_values(p_from_a, ctx.values_a)
    z_from_b = reconstruct_z_from_scores_values(p_from_b, ctx.values_b)
    invalid = ~part.valid_mask
    return {
        "pattern_a": float((p_from_a - ctx.pattern_a).abs().max()),
        "pattern_b": float((p_from_b - ctx.pattern_b).abs().max()),
        "z_a": float((z_from_a - ctx.z_a).abs().max()),
        "z_b": float((z_from_b - ctx.z_b).abs().max()),
        "masked_prob_max": float(
            torch.cat(
                [
                    p_from_a.masked_select(invalid),
                    p_from_b.masked_select(invalid),
                ]
            ).max()
            if invalid.any()
            else torch.tensor(0.0)
        ),
    }


def independent_endpoint_values(ctx: LevelBContext) -> Dict[str, float]:
    """Four direct forward passes — no hybrid-z injection."""
    a_present = _run_direct(ctx, primary_ablate=False, candidate_zero=False)
    a_zero = _run_direct(ctx, primary_ablate=False, candidate_zero=True)
    b_present = _run_direct(ctx, primary_ablate=True, candidate_zero=False)
    b_zero = _run_direct(ctx, primary_ablate=True, candidate_zero=True)
    intact = float((a_present - a_zero).mean().item())
    conditional = float((b_present - b_zero).mean().item())
    gain = conditional - intact
    table = cell_table_from_utility_fn(lambda s: utility_fn(ctx, s))
    u0 = table[(0, 0, 0, 0, 0)]
    u1 = table[(1, 1, 1, 1, 1)]
    return {
        "intact_direct": intact,
        "conditional_direct": conditional,
        "conditional_gain_direct": gain,
        "U00000": u0,
        "U11111": u1,
        "u00000_vs_intact": abs(u0 - intact),
        "u11111_vs_conditional": abs(u1 - conditional),
        "gain_vs_delta_u": abs((u1 - u0) - gain),
    }


def endpoint_errors(ctx: LevelBContext) -> Dict[str, float]:
    vals = independent_endpoint_values(ctx)
    return {
        "u00000_vs_intact": vals["u00000_vs_intact"],
        "u11111_vs_conditional": vals["u11111_vs_conditional"],
        "gain_vs_delta_u": vals["gain_vs_delta_u"],
    }


def endpoint_activation_errors(ctx: LevelBContext) -> Dict[str, float]:
    """All-A / all-B hybrid z must match independent present-run cached z."""
    _, _, z_hybrid_a = score_pattern_z_pipeline(
        ctx.scores_a,
        ctx.scores_b,
        ctx.values_a,
        ctx.values_b,
        ctx.partition,
        (0, 0, 0, 0, 0),
    )
    _, _, z_hybrid_b = score_pattern_z_pipeline(
        ctx.scores_a,
        ctx.scores_b,
        ctx.values_a,
        ctx.values_b,
        ctx.partition,
        (1, 1, 1, 1, 1),
    )
    return {
        "all_a_z_error": float((z_hybrid_a - ctx.z_a).abs().max()),
        "all_b_z_error": float((z_hybrid_b - ctx.z_b).abs().max()),
    }


def injection_scope_errors(ctx: LevelBContext, state: FactorState = (1, 0, 0, 0, 0)) -> Dict[str, float]:
    """Verify only CANDIDATE head is modified at hook_z."""
    _, _, z_hybrid = score_pattern_z_pipeline(
        ctx.scores_a,
        ctx.scores_b,
        ctx.values_a,
        ctx.values_b,
        ctx.partition,
        state,
    )
    cand_layer, cand_head = CANDIDATE
    captured: Dict[str, torch.Tensor] = {}
    hooks: List = []
    if state[4]:
        hooks.extend(ctx.primary_hooks)

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        captured["before"] = z.clone()
        out = z.clone()
        out[:, :, cand_head, :] = z_hybrid
        captured["after"] = out.clone()
        return out

    hooks.append((f"blocks.{cand_layer}.attn.hook_z", hook_fn))
    with torch.no_grad():
        with ctx.model.hooks(fwd_hooks=hooks):
            ctx.model(ctx.tokens)
    before = captured["before"]
    after = captured["after"]
    noncandidate_max = 0.0
    for h in range(ctx.model.cfg.n_heads):
        if h == cand_head:
            continue
        err = float((before[:, :, h, :] - after[:, :, h, :]).abs().max())
        noncandidate_max = max(noncandidate_max, err)
    candidate_err = float((after[:, :, cand_head, :] - z_hybrid).abs().max())
    return {
        "candidate_z_error": candidate_err,
        "noncandidate_z_error": noncandidate_max,
    }


def live_leak_errors(ctx: LevelBContext, state: FactorState = (1, 0, 0, 0, 0)) -> Dict[str, float]:
    base = _run_metric(ctx, state, candidate_zero=False, live_perturbation="none")
    v_err = _run_metric(ctx, state, candidate_zero=False, live_perturbation="v")
    q_err = _run_metric(ctx, state, candidate_zero=False, live_perturbation="q")
    qv_err = _run_metric(ctx, state, candidate_zero=False, live_perturbation="qv")
    return {
        "v_leak_error": float((base - v_err).abs().max()),
        "q_leak_error": float((base - q_err).abs().max()),
        "qv_leak_error": float((base - qv_err).abs().max()),
    }


def no_live_leak_error(ctx: LevelBContext, state: FactorState = (1, 0, 0, 0, 0)) -> float:
    return live_leak_errors(ctx, state)["v_leak_error"]


def quotient_bridge_error(ctx: LevelBContext) -> float:
    table = cell_table_from_utility_fn(lambda s: utility_fn(ctx, s))
    h = quotient_h_values_from_cell_table(table)
    phi_p, phi_v, phi_c, _ = three_factor_shapley(
        h["H000"], h["H100"], h["H010"], h["H110"], h["H001"], h["H101"], h["H011"], h["H111"]
    )
    owen = owen_value_subset(EXTENDED_FALI_PLAYERS, EXTENDED_FALI_GROUPS, char_fn_from_cell_table(table))
    q = extended_fali_quotient_shapley(char_fn_from_cell_table(table))
    return max(
        abs(sum(owen[p] for p in ("T", "R", "O")) - phi_p),
        abs(owen["V"] - phi_v),
        abs(owen["C"] - phi_c),
        abs(q["SCORE"] - phi_p),
    )
