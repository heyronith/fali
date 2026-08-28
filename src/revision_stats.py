"""
Reusable statistics for the FALI revision pipeline.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def summarize_array(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)) if x.size else float("nan"),
        "fraction_positive": float(np.mean(x > 0.0)) if x.size else float("nan"),
    }


def stable_sha256_seed(*parts: object, mod: int = 2**31 - 1) -> int:
    """Deterministic RNG seed from SHA256 of joined string parts (not Python hash())."""
    s = "|".join(map(str, parts))
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % mod


def stratified_bootstrap_ci(
    values: np.ndarray,
    strata: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Paired/one-sample mean CI via stratified bootstrap.

    Preserves stratum counts (e.g., template A/B group sizes) on every resample.
    Returns (mean, ci_low, ci_high).
    """
    values = np.asarray(values, dtype=np.float64)
    strata = np.asarray(strata)
    assert values.shape[0] == strata.shape[0]
    assert values.size > 0

    rng = np.random.default_rng(seed)
    unique, counts = np.unique(strata, return_counts=True)
    stratum_indices = {u: np.where(strata == u)[0] for u in unique}

    means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        chosen = []
        for u, c in zip(unique, counts):
            idxs = stratum_indices[u]
            draw = rng.choice(idxs, size=c, replace=True)
            chosen.append(draw)
        sample_idx = np.concatenate(chosen)
        means[b] = float(np.mean(values[sample_idx]))

    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(np.mean(values)), low, high


def within_template_stratified_bootstrap_indices(
    strata: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    Phase-4 A: within-template stratified bootstrap over all 30 strata.

    Each draw preserves every stratum label and resamples examples within stratum.
    Returns index array of shape (n_bootstrap, n).
    """
    strata = np.asarray(strata)
    n = strata.shape[0]
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(strata, return_counts=True)
    stratum_indices = {u: np.where(strata == u)[0] for u in unique}
    out = np.empty((n_bootstrap, n), dtype=np.int64)
    for b in range(n_bootstrap):
        chosen = []
        for u, c in zip(unique, counts):
            idxs = stratum_indices[u]
            chosen.append(rng.choice(idxs, size=int(c), replace=True))
        out[b] = np.concatenate(chosen)
    return out


def within_template_stratified_bootstrap_ci(
    values: np.ndarray,
    strata: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Mean CI via within-30-stratum bootstrap (alias of stratified_bootstrap_ci)."""
    return stratified_bootstrap_ci(
        values, strata, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
    )


def template_cluster_bootstrap_indices(
    low_level_template_id: np.ndarray,
    pattern_type: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    n_templates: int = 15,
) -> np.ndarray:
    """
    Phase-4 B: template-cluster bootstrap.

    Resample the 15 low-level templates as clusters (with replacement).
    For each sampled template id, retain BOTH BABA and ABBA strata, then
    resample prompt instances within those selected strata to restore the
    original total count structure (n_templates clusters × both patterns).

    Returns index array of shape (n_bootstrap, n) where n = len(values).
    """
    tid = np.asarray(low_level_template_id)
    pattern = np.asarray(pattern_type)
    assert tid.shape[0] == pattern.shape[0]
    n = tid.shape[0]
    assert set(np.unique(tid).tolist()) <= set(range(n_templates))

    strata = np.array([f"T{int(t):02d}_{p}" for t, p in zip(tid, pattern)])
    stratum_indices = {u: np.where(strata == u)[0] for u in np.unique(strata)}
    template_ids = np.arange(n_templates)
    counts_by_key = {}
    for t in template_ids:
        for p in ("BABA", "ABBA"):
            key = f"T{t:02d}_{p}"
            counts_by_key[key] = int(np.sum(strata == key))

    rng = np.random.default_rng(seed)
    out = np.empty((n_bootstrap, n), dtype=np.int64)
    for b in range(n_bootstrap):
        drawn_templates = rng.choice(template_ids, size=n_templates, replace=True)
        chosen = []
        for t in drawn_templates:
            for p in ("BABA", "ABBA"):
                key = f"T{int(t):02d}_{p}"
                c = counts_by_key[key]
                if c == 0:
                    raise RuntimeError(f"Empty stratum {key} in cluster bootstrap")
                idxs = stratum_indices[key]
                chosen.append(rng.choice(idxs, size=c, replace=True))
        sample_idx = np.concatenate(chosen)
        assert sample_idx.shape[0] == n
        out[b] = sample_idx
    return out


def template_cluster_bootstrap_ci(
    values: np.ndarray,
    low_level_template_id: np.ndarray,
    pattern_type: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    n_templates: int = 15,
) -> Tuple[float, float, float]:
    """Mean CI via template-cluster bootstrap (BABA+ABBA paired; vectorized)."""
    values = np.asarray(values, dtype=np.float64)
    tid = np.asarray(low_level_template_id).astype(np.int64)
    pattern = np.asarray(pattern_type)
    n = values.shape[0]
    patterns = ("BABA", "ABBA")
    pat_to_i = {p: i for i, p in enumerate(patterns)}

    # Infer balanced cell size
    counts = []
    for t in range(n_templates):
        for p in patterns:
            c = int(((tid == t) & (pattern == p)).sum())
            if c == 0:
                raise RuntimeError(f"Empty stratum T{t:02d}_{p}")
            counts.append(c)
    if len(set(counts)) != 1:
        # Fall back to index-based method for unbalanced designs
        idx = template_cluster_bootstrap_indices(
            tid, pattern, n_bootstrap=n_bootstrap, seed=seed, n_templates=n_templates
        )
        means = np.mean(values[idx], axis=1)
        return (
            float(np.mean(values)),
            float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)),
        )
    n_per = counts[0]

    data = np.empty((n_templates, 2, n_per), dtype=np.float64)
    counters = np.zeros((n_templates, 2), dtype=np.int64)
    for i in range(n):
        ti = int(tid[i])
        pi = pat_to_i[str(pattern[i])]
        k = counters[ti, pi]
        data[ti, pi, k] = values[i]
        counters[ti, pi] = k + 1

    rng = np.random.default_rng(seed)
    drawn_t = rng.integers(0, n_templates, size=(n_bootstrap, n_templates))
    ex_idx = rng.integers(0, n_per, size=(n_bootstrap, n_templates, 2, n_per))
    shape = (n_bootstrap, n_templates, 2, n_per)
    t_b = np.broadcast_to(drawn_t[:, :, None, None], shape)
    p_b = np.broadcast_to(np.arange(2)[None, None, :, None], shape)
    sampled = data[t_b, p_b, ex_idx]
    means = sampled.reshape(n_bootstrap, -1).mean(axis=1)
    return (
        float(np.mean(values)),
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def hierarchical_template_cluster_bootstrap_means(
    values: np.ndarray,
    seed_ids: np.ndarray,
    low_level_template_id: np.ndarray,
    pattern_type: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    n_templates: int = 15,
    return_draw_arrays: bool = False,
):
    """
    Hierarchical template-cluster bootstrap for multi-seed aggregate.

    Per draw:
      1. sample 15 template clusters with replacement (SHARED across seeds);
      2. retain both BABA and ABBA for each sampled template;
      3. within every sampled seed × template × pattern cell, resample its prompts
         INDEPENDENTLY per seed (do not reuse within-cell example indices across seeds);
      4. compute each seed's mean;
      5. average the seed means equally (seeds themselves are not resampled).

    Returns bootstrap means of shape (n_bootstrap,).
    If return_draw_arrays=True, also returns (drawn_t, ex_idx) where
      drawn_t: (n_bootstrap, n_templates)
      ex_idx:  (n_bootstrap, n_seeds, n_templates, 2, n_per_cell)
    """
    values = np.asarray(values, dtype=np.float64)
    seed_ids = np.asarray(seed_ids)
    tid = np.asarray(low_level_template_id).astype(np.int64)
    pattern = np.asarray(pattern_type)
    n = values.shape[0]
    assert seed_ids.shape[0] == n == tid.shape[0] == pattern.shape[0]

    unique_seeds = np.array(sorted(np.unique(seed_ids), key=lambda x: int(x)))
    n_seeds = len(unique_seeds)
    patterns = ("BABA", "ABBA")
    pat_to_i = {p: i for i, p in enumerate(patterns)}

    first = None
    for s in unique_seeds:
        for t in range(n_templates):
            for p in patterns:
                m = (seed_ids == s) & (tid == t) & (pattern == p)
                c = int(m.sum())
                if c == 0:
                    raise RuntimeError(f"Empty cell seed={s} T={t} {p}")
                if first is None:
                    first = c
                elif c != first:
                    raise RuntimeError(
                        f"Unbalanced cell sizes: expected {first}, got {c} at seed={s} T={t} {p}"
                    )
    n_per_cell = int(first)

    # data[seed_i, template, pattern_i, example]
    data = np.empty((n_seeds, n_templates, 2, n_per_cell), dtype=np.float64)
    seed_to_i = {s: i for i, s in enumerate(unique_seeds)}
    counters = np.zeros((n_seeds, n_templates, 2), dtype=np.int64)
    for i in range(n):
        si = seed_to_i[seed_ids[i]]
        ti = int(tid[i])
        pi = pat_to_i[str(pattern[i])]
        k = counters[si, ti, pi]
        data[si, ti, pi, k] = values[i]
        counters[si, ti, pi] = k + 1
    if not np.all(counters == n_per_cell):
        raise RuntimeError(f"Cell fill mismatch: {counters}")

    rng = np.random.default_rng(seed)
    # Shared template clusters across seeds within each bootstrap draw
    drawn_t = rng.integers(0, n_templates, size=(n_bootstrap, n_templates))
    # Independent within-cell example draws per seed cell
    ex_idx = rng.integers(
        0, n_per_cell, size=(n_bootstrap, n_seeds, n_templates, 2, n_per_cell)
    )

    seed_means = np.empty((n_bootstrap, n_seeds), dtype=np.float64)
    p_idx = np.arange(2)[None, None, :, None]  # (1, 1, 2, 1)
    shape_tp = (n_bootstrap, n_templates, 2, n_per_cell)
    t_b = np.broadcast_to(drawn_t[:, :, None, None], shape_tp)
    p_b = np.broadcast_to(p_idx, shape_tp)
    for si in range(n_seeds):
        sampled = data[si][t_b, p_b, ex_idx[:, si]]
        seed_means[:, si] = sampled.reshape(n_bootstrap, -1).mean(axis=1)

    out = seed_means.mean(axis=1)
    if return_draw_arrays:
        return out, drawn_t, ex_idx
    return out


def hierarchical_template_cluster_bootstrap_means_buggy_shared_ex_idx(
    values: np.ndarray,
    seed_ids: np.ndarray,
    low_level_template_id: np.ndarray,
    pattern_type: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    n_templates: int = 15,
    return_draw_arrays: bool = False,
):
    """
    Pre-fix Phase-4B implementation retained for regression tests only.

    BUG: reuses one ex_idx[bootstrap, template, pattern, example] across all seeds.
    """
    values = np.asarray(values, dtype=np.float64)
    seed_ids = np.asarray(seed_ids)
    tid = np.asarray(low_level_template_id).astype(np.int64)
    pattern = np.asarray(pattern_type)
    unique_seeds = np.array(sorted(np.unique(seed_ids), key=lambda x: int(x)))
    n_seeds = len(unique_seeds)
    patterns = ("BABA", "ABBA")
    pat_to_i = {p: i for i, p in enumerate(patterns)}
    n_per_cell = None
    for s in unique_seeds:
        for t in range(n_templates):
            for p in patterns:
                c = int(((seed_ids == s) & (tid == t) & (pattern == p)).sum())
                if n_per_cell is None:
                    n_per_cell = c
                assert c == n_per_cell
    n_per_cell = int(n_per_cell)
    data = np.empty((n_seeds, n_templates, 2, n_per_cell), dtype=np.float64)
    seed_to_i = {s: i for i, s in enumerate(unique_seeds)}
    counters = np.zeros((n_seeds, n_templates, 2), dtype=np.int64)
    for i in range(len(values)):
        si = seed_to_i[seed_ids[i]]
        ti = int(tid[i])
        pi = pat_to_i[str(pattern[i])]
        k = counters[si, ti, pi]
        data[si, ti, pi, k] = values[i]
        counters[si, ti, pi] = k + 1
    rng = np.random.default_rng(seed)
    drawn_t = rng.integers(0, n_templates, size=(n_bootstrap, n_templates))
    # BUG: no seed axis
    ex_idx = rng.integers(0, n_per_cell, size=(n_bootstrap, n_templates, 2, n_per_cell))
    seed_means = np.empty((n_bootstrap, n_seeds), dtype=np.float64)
    shape = (n_bootstrap, n_templates, 2, n_per_cell)
    t_b = np.broadcast_to(drawn_t[:, :, None, None], shape)
    p_b = np.broadcast_to(np.arange(2)[None, None, :, None], shape)
    for si in range(n_seeds):
        sampled = data[si][t_b, p_b, ex_idx]
        seed_means[:, si] = sampled.reshape(n_bootstrap, -1).mean(axis=1)
    out = seed_means.mean(axis=1)
    if return_draw_arrays:
        return out, drawn_t, ex_idx
    return out



def hierarchical_template_cluster_bootstrap_ci(
    values: np.ndarray,
    seed_ids: np.ndarray,
    low_level_template_id: np.ndarray,
    pattern_type: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    n_templates: int = 15,
) -> Tuple[float, float, float]:
    """
    Point estimate = equal average of per-seed means.
    CI = hierarchical template-cluster bootstrap (PRIMARY for broad-template claims).
    """
    values = np.asarray(values, dtype=np.float64)
    seed_ids = np.asarray(seed_ids)
    unique_seeds = np.unique(seed_ids)
    seed_means = [float(np.mean(values[seed_ids == s])) for s in unique_seeds]
    point = float(np.mean(seed_means))
    boot = hierarchical_template_cluster_bootstrap_means(
        values,
        seed_ids,
        low_level_template_id,
        pattern_type,
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_templates=n_templates,
    )
    low = float(np.quantile(boot, alpha / 2.0))
    high = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return point, low, high


def pearson_spearman(
    x: Sequence[float],
    y: Sequence[float],
) -> Dict[str, Optional[float]]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 2 or y_arr.size < 2:
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
        }
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
        }
    pr = stats.pearsonr(x_arr, y_arr)
    sr = stats.spearmanr(x_arr, y_arr)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }
