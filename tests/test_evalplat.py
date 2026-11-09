from __future__ import annotations

import random

import numpy as np
import pytest

from evalplat import gate, golden, judges
from evalplat.cli import main
from evalplat.evaluate import BASELINE, candidate
from evalplat.stats import holm, mcnemar_p, paired_bootstrap_ci
from evalplat.systems import Answer, System, run


@pytest.fixture(scope="module")
def items() -> list[golden.Item]:
    return golden.golden_set(300, seed=3)


def answer(item: golden.Item, text: str, correct: bool) -> Answer:
    return Answer(item.id, item.slice, text, correct, len(item.facts) if correct else 0, len(text.split()))


def test_golden_set_is_balanced_and_deterministic(items: list[golden.Item]) -> None:
    assert [i.slice for i in items[:5]] == golden.SLICES
    assert golden.golden_set(300, seed=3) == items
    assert all(len(i.facts) == 2 for i in items)


def test_answers_carry_their_true_label(items: list[golden.Item]) -> None:
    perfect = run(System("p", dict.fromkeys(golden.SLICES, 1.0), hallucination=0.0), items, 1)
    assert all(a.facts_right == 2 for a in perfect)
    hopeless = run(candidate("h", dict.fromkeys(golden.SLICES, 0.0)), items, 1)
    assert not any(a.correct for a in hopeless)


def test_rubric_reads_listed_phrasings_and_rejects_wrong_facts(items: list[golden.Item]) -> None:
    it = items[1]
    good = " ".join(f.listed[0] + "." for f in it.facts)
    wrong = it.facts[0].wrong + ". " + it.facts[1].listed[0] + "."
    para = it.facts[0].paraphrase + ". " + it.facts[1].listed[0] + "."
    r = judges.Rubric()
    assert r.grade(it, answer(it, good, True))
    assert not r.grade(it, answer(it, wrong, False))
    assert not r.grade(it, answer(it, para, True))  # the listed-only rubric misses the paraphrase


def test_rubric_learns_templates_from_disagreements(items: list[golden.Item]) -> None:
    pairs = []
    for it in items[:60]:
        text = "Thanks for getting in touch. " + " ".join(f.paraphrase + "." for f in it.facts)
        pairs.append((it, answer(it, text, True)))
    learned = judges.Rubric().learn(pairs, [True] * len(pairs))
    assert "within <n> days weekends excluded" in learned.learned
    assert not any("thanks" in t for t in learned.learned)
    later = next(i for i in items[100:] if i.slice == "shipping")
    text = later.facts[0].paraphrase + ". " + later.facts[1].listed[0] + "."
    assert learned.grade(later, answer(later, text, True))


def test_kappa() -> None:
    x = [True, True, False, False]
    assert judges.cohen_kappa(x, x) == 1.0
    assert judges.cohen_kappa(x, [True, False, True, False]) == 0.0
    assert judges.fleiss_kappa([[True, True, True], [False, False, False]]) == 1.0


def test_model_judge_prefers_long_wrong_answers(items: list[golden.Item]) -> None:
    mj = judges.ModelJudge(error=0.0)
    it = items[0]
    short = answer(it, "No idea.", False)
    long = answer(it, " ".join(["Thanks for asking about this, it is a good question."] * 8), False)
    rate_long = np.mean([judges.ModelJudge(error=0.0, seed=s).grade(it, long) for s in range(200)])
    assert not mj.grade(it, short)
    assert rate_long > 0.2


def test_stats() -> None:
    rng = np.random.default_rng(0)
    b = rng.random(400) < 0.7
    assert mcnemar_p(b, b) == 1.0
    worse = b & (rng.random(400) < 0.8)
    assert mcnemar_p(b, worse) < 0.001
    lo, hi = paired_bootstrap_ci(b, worse)
    assert hi < 0 < 1 + lo
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_gate_blocks_a_slice_regression_the_mean_hides() -> None:
    its = golden.golden_set(1200, seed=5)
    by = {i.id: i for i in its}
    recall = {**dict.fromkeys(golden.SLICES, 0.98), "returns": 0.80}
    base, cand = run(BASELINE, its, 1), run(candidate("c", recall), its, 2)
    grade = judges.Rubric().grade
    b = np.array([grade(by[a.item_id], a) for a in base])
    c = np.array([grade(by[a.item_id], a) for a in cand])
    comps = gate.compare([a.slice for a in base], b, c, ci=False)
    assert gate.decide(comps).blocked
    assert not gate.decide(comps, "overall mean").blocked
    assert gate.decide(comps).reasons[0].startswith("returns")


def test_gate_ships_an_unchanged_release() -> None:
    its = golden.golden_set(600, seed=6)
    by = {i.id: i for i in its}
    grade = judges.Rubric().grade
    blocked = 0
    for t in range(20):
        b = np.array([grade(by[a.item_id], a) for a in run(BASELINE, its, 100 + t)])
        c = np.array([grade(by[a.item_id], a) for a in run(candidate("same", {}), its, 200 + t)])
        blocked += gate.decide(gate.compare([i.slice for i in its], b, c, ci=False)).blocked
    assert blocked <= 2


def test_cli_compare_exit_codes() -> None:
    assert main(["compare"]) == 0
    assert main(["compare", "--slice", "returns", "--slice-recall", "0.6"]) == 1
    random.seed(0)
