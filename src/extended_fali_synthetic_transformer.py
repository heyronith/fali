"""
EF-1 Level B — tiny random HookedTransformer validation (mechanical, not scientific).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from extended_fali_interventions import (
    FactorState,
    ScorePartition,
    build_score_partition_masks,
    cell_table_from_utility_fn,
    char_fn_from_cell_table,
    masked_softmax,
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


@dataclass
class LevelBContext:
    model: HookedTransformer
    tokens: torch.Tensor
    target_key: torch.Tensor  # [batch] key index per example at final query
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


def build_level_b_context(seed: int = EF1_SEED, batch: int = 2) -> LevelBContext:
    torch.manual_seed(seed)
    model = build_tiny_model(seed)
    tokens = torch.randint(0, EF1_ARCH["d_vocab"], (batch, 6))
    scores_a, pattern_a, values_a, z_a = _cache_candidate(model, tokens, primary_ablate=False)
    scores_b, pattern_b, values_b, z_b = _cache_candidate(model, tokens, primary_ablate=True)
    seq_len = scores_a.shape[1]
    final_query = seq_len - 1
    # target = argmax pattern mass on final row among valid keys (deterministic)
    valid = build_score_partition_masks(seq_len, final_query, final_query - 1).valid_mask
    row = pattern_a[:, final_query].clone()
    row = row.masked_fill(~valid[final_query].unsqueeze(0), 0.0)
    target_key = row.argmax(dim=-1)
    # rebuild partition per-example using first target for shared partition (use mode)
    tk = int(target_key[0].item())
    partition = build_score_partition_masks(seq_len, final_query, tk)
    validate_score_partition(partition)
    primary_hooks = build_zero_ablation_hooks([PRIMARY], n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads)
    return LevelBContext(
        model=model,
        tokens=tokens,
        target_key=target_key,
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


def _run_metric(
    ctx: LevelBContext,
    state: FactorState,
    *,
    candidate_zero: bool,
    live_perturbation: float = 0.0,
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
    hooks = []
    if state[4]:  # context B => primary ablated
        hooks.extend(ctx.primary_hooks)
    if candidate_zero:
        hooks.append(_zero_candidate_z(ctx.model))
    else:
        hooks.append(_inject_candidate_z(ctx.model, z_hybrid))
    if live_perturbation != 0.0:

        def v_perturb(v: torch.Tensor, hook) -> torch.Tensor:
            out = v.clone()
            out[:, :, cand_head, :] = out[:, :, cand_head, :] + live_perturbation
            return out

        hooks.insert(0, (f"blocks.{cand_layer}.attn.hook_v", v_perturb))
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
    p_from_a = masked_softmax(ctx.scores_a.masked_fill(~part.valid_mask.unsqueeze(0), float("-inf")), part.valid_mask)
    p_from_b = masked_softmax(ctx.scores_b.masked_fill(~part.valid_mask.unsqueeze(0), float("-inf")), part.valid_mask)
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
                    p_from_a.masked_select(invalid.unsqueeze(0)),
                    p_from_b.masked_select(invalid.unsqueeze(0)),
                ]
            ).max()
            if invalid.any()
            else torch.tensor(0.0)
        ),
    }


def endpoint_errors(ctx: LevelBContext) -> Dict[str, float]:
    table = cell_table_from_utility_fn(lambda s: utility_fn(ctx, s))
    u0 = table[(0, 0, 0, 0, 0)]
    u1 = table[(1, 1, 1, 1, 1)]
    intact = utility_fn(ctx, (0, 0, 0, 0, 0))  # same as u0 by construction
    # Direct intact/conditional from forward semantics
    inj_a = _run_metric(ctx, (0, 0, 0, 0, 0), candidate_zero=False)
    zero_a = _run_metric(ctx, (0, 0, 0, 0, 0), candidate_zero=True)
    inj_b = _run_metric(ctx, (1, 1, 1, 1, 1), candidate_zero=False)
    zero_b = _run_metric(ctx, (1, 1, 1, 1, 1), candidate_zero=True)
    intact_direct = float((inj_a - zero_a).mean())
    cond_direct = float((inj_b - zero_b).mean())
    gain_direct = cond_direct - intact_direct
    return {
        "u00000_vs_intact": abs(u0 - intact_direct),
        "u11111_vs_conditional": abs(u1 - cond_direct),
        "gain_vs_delta_u": abs((u1 - u0) - gain_direct),
    }


def no_live_leak_error(ctx: LevelBContext, state: FactorState = (1, 0, 0, 0, 0)) -> float:
    base = _run_metric(ctx, state, candidate_zero=False, live_perturbation=0.0)
    pert = _run_metric(ctx, state, candidate_zero=False, live_perturbation=1e6)
    return float((base - pert).abs().max())


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
