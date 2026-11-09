"""Paired statistics for comparing two systems on the same items."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def paired_bootstrap_ci(
    base: np.ndarray, cand: np.ndarray, b: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """CI for mean(cand) - mean(base), resampling items so each pair stays together."""
    rng = np.random.default_rng(seed)
    d = cand.astype(float) - base.astype(float)
    idx = rng.integers(0, len(d), size=(b, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def mcnemar_p(base: np.ndarray, cand: np.ndarray) -> float:
    """Exact two-sided McNemar test on the items where the two systems disagree."""
    b = int(np.sum(base & ~cand))
    c = int(np.sum(~base & cand))
    if b + c == 0:
        return 1.0
    return float(binomtest(c, b + c, 0.5).pvalue)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
