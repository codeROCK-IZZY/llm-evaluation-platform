"""The release gate: does the candidate regress, overall or on any slice, beyond what noise explains?

The gate blocks when a slice (or the whole set) is worse by more than a practical margin *and* the
paired McNemar test, Holm-corrected across every comparison, says the drop is real. Three common
alternatives are kept for comparison:

  overall mean   block if the overall pass rate drops at all (blind to a regression in one slice)
  any slice      block if any slice's pass rate drops at all (blocks on noise)
  uncorrected    the statistical gate without the multiple-comparison correction
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .golden import SLICES
from .stats import holm, mcnemar_p, paired_bootstrap_ci

ALPHA = 0.05
MARGIN = 0.02


@dataclass
class Comparison:
    scope: str
    n: int
    base: float
    cand: float
    diff: float
    ci: tuple[float, float]
    p: float
    p_adj: float = 1.0


@dataclass
class Decision:
    blocked: bool
    comparisons: list[Comparison] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def compare(slices: list[str], base: np.ndarray, cand: np.ndarray, ci: bool = True) -> list[Comparison]:
    out = []
    arr = np.array(slices)
    for scope in ["overall", *SLICES]:
        m = np.ones(len(arr), bool) if scope == "overall" else arr == scope
        b, c = base[m], cand[m]
        interval = paired_bootstrap_ci(b, c) if ci else (float("nan"), float("nan"))
        out.append(
            Comparison(
                scope,
                int(m.sum()),
                float(b.mean()),
                float(c.mean()),
                float(c.mean() - b.mean()),
                interval,
                mcnemar_p(b, c),
            )
        )
    for comp, p in zip(out, holm([c.p for c in out]), strict=True):
        comp.p_adj = p
    return out


def decide(comps: list[Comparison], rule: str = "statistical") -> Decision:
    d = Decision(False, comps)
    for c in comps:
        if rule == "statistical":
            hit = c.diff < -MARGIN and c.p_adj < ALPHA
        elif rule == "uncorrected":
            hit = c.diff < -MARGIN and c.p < ALPHA
        elif rule == "overall mean":
            hit = c.scope == "overall" and c.diff < 0
        elif rule == "any slice":
            hit = c.diff < 0
        else:
            raise ValueError(rule)
        if hit:
            d.blocked = True
            d.reasons.append(f"{c.scope}: {c.diff:+.1%} (p_adj={c.p_adj:.3g})")
    return d


RULES = ["statistical", "uncorrected", "overall mean", "any slice"]
