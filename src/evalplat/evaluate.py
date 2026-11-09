"""The evaluation of the evaluation: judges against humans, pairwise bias, and the gate's error rates."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import gate, golden, judges
from .systems import Answer, System, run

BASE_RECALL = 0.93
BASELINE = System("v1", dict.fromkeys(golden.SLICES, BASE_RECALL), verbosity=2.0)
SCENARIOS: dict[str, dict[str, float]] = {
    "no change (A/A)": {},
    "better everywhere (+3 pts recall)": dict.fromkeys(golden.SLICES, BASE_RECALL + 0.03),
    "better overall, returns -10 pts": {
        **dict.fromkeys(golden.SLICES, BASE_RECALL + 0.05),
        "returns": BASE_RECALL - 0.10,
    },
    "better overall, returns -5 pts": {
        **dict.fromkeys(golden.SLICES, BASE_RECALL + 0.05),
        "returns": BASE_RECALL - 0.05,
    },
}
SHOULD_BLOCK = {
    "no change (A/A)": False,
    "better everywhere (+3 pts recall)": False,
    "better overall, returns -10 pts": True,
    "better overall, returns -5 pts": True,
}


@dataclass
class Result:
    items: int
    trials: int
    annotators: dict[str, float] = field(default_factory=dict)
    judges: dict[str, dict[str, float]] = field(default_factory=dict)
    learned_phrasings: list[str] = field(default_factory=list)
    pairwise: dict[str, float] = field(default_factory=dict)
    gate: dict[str, dict[str, float]] = field(default_factory=dict)  # scenario -> rule -> block rate
    power: dict[str, float] = field(default_factory=dict)  # golden set size -> detection
    release: list[dict[str, Any]] = field(default_factory=list)
    release_decisions: dict[str, bool] = field(default_factory=dict)
    seconds: float = 0.0


def candidate(name: str, recall: dict[str, float]) -> System:
    full = dict.fromkeys(golden.SLICES, BASE_RECALL) | recall
    return System(name, full, verbosity=2.0)


def _judge_stats(grades: list[bool], human: list[bool], answers: list[Answer]) -> dict[str, float]:
    g = np.array(grades)
    truth = np.array([a.correct for a in answers])
    long = np.array([a.words > 30 for a in answers])
    wrong = ~truth

    def rate(m: np.ndarray) -> float:
        return float(g[m].mean()) if m.any() else 0.0

    return {
        "kappa_vs_humans": judges.cohen_kappa(grades, human),
        "accuracy_vs_truth": float((g == truth).mean()),
        "pass_rate": float(g.mean()),
        "wrong_passed_long": rate(wrong & long),
        "wrong_passed_short": rate(wrong & ~long),
    }


def run_eval(n_items: int = 1200, trials: int = 200) -> Result:
    t0 = time.perf_counter()
    r = Result(n_items, trials)
    items = golden.golden_set(n_items)
    by = {i.id: i for i in items}
    ann = judges.Annotators()
    mj = judges.ModelJudge()

    # ---- judges, calibrated on half the items and scored on the other half
    pool: list[Answer] = []
    for sys_, seed in (
        (BASELINE, 1),
        (candidate("v2", {"returns": 0.85}), 2),
        (System("v0", dict.fromkeys(golden.SLICES, 0.80), hallucination=0.08, verbosity=4.0), 3),
    ):
        pool += run(sys_, items, seed)
    cal_ids = {i.id for i in items[: n_items // 2]}
    cal = [a for a in pool if a.item_id in cal_ids]
    hold = [a for a in pool if a.item_id not in cal_ids]
    hum_cal = [ann.consensus(a) for a in cal]
    hum = [ann.consensus(a) for a in hold]
    labels = [ann.labels(a) for a in hold]
    r.annotators = {
        "fleiss_kappa": judges.fleiss_kappa(labels),
        **{
            f"annotator_{k + 1}_kappa_vs_majority": judges.cohen_kappa([x[k] for x in labels], hum)
            for k in range(len(ann.errors))
        },
        "majority_kappa_vs_truth": judges.cohen_kappa(hum, [a.correct for a in hold]),
    }
    thr = max(
        np.arange(0.2, 0.9, 0.05),
        key=lambda t: judges.cohen_kappa([judges.token_f1(by[a.item_id], a) >= t for a in cal], hum_cal),
    )
    learned = judges.Rubric().learn([(by[a.item_id], a) for a in cal], hum_cal)
    r.learned_phrasings = sorted(learned.learned)
    graders: dict[str, judges.Grader] = {
        "exact match": judges.exact_match,
        f"token F1 >= {thr:.2f}": lambda it, a: judges.token_f1(it, a) >= thr,
        "rubric (listed phrasings)": judges.Rubric().grade,
        "model judge": mj.grade,
        "rubric (calibrated)": learned.grade,
    }
    for name, f in graders.items():
        r.judges[name] = _judge_stats([f(by[a.item_id], a) for a in hold], hum, hold)

    # ---- pairwise preference: the model judge read in one order, and in both
    v1 = run(BASELINE, items, 21)
    v0 = run(System("v0", dict.fromkeys(golden.SLICES, 0.80), hallucination=0.08, verbosity=4.0), items, 22)
    agree_single = agree_deb = decided = first = n = 0
    for k, (a, b) in enumerate(zip(v1, v0, strict=True)):
        it = by[a.item_id]
        h = ann.prefer(it, a, b)
        if h is None:
            continue  # only pairs where the annotators have a preference
        n += 1
        better, worse = (a, b) if h == 0 else (b, a)
        # alternate which answer is read first, as a single-order judge sees them in practice
        x, y = (better, worse) if k % 2 else (worse, better)
        picked = x if mj.prefer(it, x, y) == 0 else y
        agree_single += picked is better
        first += mj.prefer(it, worse, better) == 0
        d = mj.prefer_debiased(it, better, worse)
        if d is not None:
            decided += 1
            agree_deb += d == 0
    r.pairwise = {
        "single_order_agreement": agree_single / n,
        "picked_first_when_worse_first": first / n,
        "debiased_agreement_on_decided": agree_deb / max(decided, 1),
        "debiased_decided_share": decided / n,
    }

    # ---- the gate, on releases whose true effect is known
    grade = learned.grade
    for scen, recall in SCENARIOS.items():
        counts = dict.fromkeys(gate.RULES, 0)
        for t in range(trials):
            base = run(BASELINE, items, 1000 + t)
            cand = run(candidate("cand", recall), items, 5000 + t)
            gb = np.array([grade(by[a.item_id], a) for a in base])
            gc = np.array([grade(by[a.item_id], a) for a in cand])
            comps = gate.compare([a.slice for a in base], gb, gc, ci=False)
            for rule in gate.RULES:
                counts[rule] += gate.decide(comps, rule).blocked
        r.gate[scen] = {rule: v / trials for rule, v in counts.items()}

    scen = "better overall, returns -10 pts"
    for size in (150, 300, 600, 1200):
        its = golden.golden_set(size, seed=31)
        byi = {i.id: i for i in its}
        hits = 0
        for t in range(trials // 2):
            base = run(BASELINE, its, 9000 + t)
            cand = run(candidate("cand", SCENARIOS[scen]), its, 9500 + t)
            gb = np.array([grade(byi[a.item_id], a) for a in base])
            gc = np.array([grade(byi[a.item_id], a) for a in cand])
            hits += gate.decide(gate.compare([a.slice for a in base], gb, gc, ci=False)).blocked
        r.power[str(size)] = hits / (trials // 2)

    # ---- one release, in full
    base = run(BASELINE, items, 77)
    cand = run(candidate("v2", SCENARIOS[scen]), items, 78)
    gb = np.array([grade(by[a.item_id], a) for a in base])
    gc = np.array([grade(by[a.item_id], a) for a in cand])
    comps = gate.compare([a.slice for a in base], gb, gc)
    r.release = [asdict(x) for x in comps]
    r.release_decisions = {rule: gate.decide(comps, rule).blocked for rule in gate.RULES}
    r.seconds = time.perf_counter() - t0
    return r


def check(r: Result) -> list[str]:
    p = []
    cal = r.judges["rubric (calibrated)"]
    if cal["kappa_vs_humans"] < 0.8:
        p.append(f"calibrated rubric kappa {cal['kappa_vs_humans']:.2f} < 0.80")
    if cal["kappa_vs_humans"] <= r.judges["rubric (listed phrasings)"]["kappa_vs_humans"]:
        p.append("calibration did not improve the rubric")
    g = r.gate
    if g["no change (A/A)"]["statistical"] > 0.05:
        p.append(f"false alarms on A/A {g['no change (A/A)']['statistical']:.3f} > 0.05")
    if g["better everywhere (+3 pts recall)"]["statistical"] > 0.05:
        p.append("the gate blocks real improvements")
    if g["better overall, returns -10 pts"]["statistical"] < 0.8:
        p.append(
            f"power on a 10-point slice regression {g['better overall, returns -10 pts']['statistical']:.2f} < 0.8"
        )
    if g["better overall, returns -10 pts"]["overall mean"] > 0.2:
        p.append("the overall-mean baseline should miss slice regressions; the scenario is broken")
    if r.pairwise["debiased_agreement_on_decided"] <= r.pairwise["single_order_agreement"]:
        p.append("reading both orders did not improve pairwise agreement")
    return p


def to_json(r: Result) -> str:
    return json.dumps(asdict(r), indent=2) + "\n"


def _p(x: float) -> str:
    return f"{x * 100:.1f}%"


def to_markdown(r: Result) -> str:
    L = [
        "# Evaluation",
        "",
        f"{r.items} golden items in {len(golden.SLICES)} slices, {r.trials} simulated releases per "
        "scenario. Generated by `evalp evaluate`.",
        "",
        "## Judges against human labels (held-out half)",
        "",
        "| judge | kappa vs humans | accuracy vs truth | pass rate | wrong answers passed: long / short |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in r.judges.items():
        L.append(
            f"| {name} | {s['kappa_vs_humans']:.3f} | {_p(s['accuracy_vs_truth'])} | {_p(s['pass_rate'])} | "
            f"{_p(s['wrong_passed_long'])} / {_p(s['wrong_passed_short'])} |"
        )
    a = r.annotators
    L += [
        "",
        f"Three annotators: Fleiss kappa {a['fleiss_kappa']:.3f}; their majority agrees with the truth at kappa "
        f"{a['majority_kappa_vs_truth']:.3f}.",
        "",
        f"The calibrated rubric learned {len(r.learned_phrasings)} phrasings from answers the annotators passed and "
        "the rubric failed, on the calibration half only:",
        "",
    ]
    L += [f"- `{t}`" for t in r.learned_phrasings]
    pw = r.pairwise
    L += [
        "",
        "## Pairwise preference (model judge)",
        "",
        "| | |",
        "|---|---:|",
        f"| agreement with humans, one reading order | {_p(pw['single_order_agreement'])} |",
        f"| picked the first answer when it was the worse one | {_p(pw['picked_first_when_worse_first'])} |",
        f"| agreement when both orders agree | {_p(pw['debiased_agreement_on_decided'])} |",
        f"| share of pairs where both orders agree | {_p(pw['debiased_decided_share'])} |",
        "",
        "## The release gate on releases with a known effect",
        "",
        "Share of releases blocked (should be 0% for the first two rows and 100% for the last two):",
        "",
        "| scenario | " + " | ".join(gate.RULES) + " |",
        "|---|" + "---:|" * len(gate.RULES),
    ]
    for scen, rules in r.gate.items():
        L.append(f"| {scen} | " + " | ".join(_p(rules[x]) for x in gate.RULES) + " |")
    L += [
        "",
        "Detection of the 10-point slice regression by golden set size (statistical gate):",
        "",
        "| items | detected |",
        "|---:|---:|",
    ]
    L += [f"| {k} | {_p(v)} |" for k, v in r.power.items()]
    L += [
        "",
        "## One sampled release: better overall, returns -10 pts",
        "",
        "A single draw from the scenario above, not chosen:",
        "",
        "| scope | n | baseline | candidate | diff | 95% CI | p (Holm) |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for c in r.release:
        L.append(
            f"| {c['scope']} | {c['n']} | {_p(c['base'])} | {_p(c['cand'])} | {c['diff'] * 100:+.1f} pts | "
            f"[{c['ci'][0] * 100:+.1f}, {c['ci'][1] * 100:+.1f}] | {c['p_adj']:.3g} |"
        )
    L += [
        "",
        "Decisions: "
        + ", ".join(f"{k}: {'blocked' if v else 'shipped'}" for k, v in r.release_decisions.items())
        + ".",
        "",
    ]
    return "\n".join(L)
