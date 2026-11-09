"""Judges that grade answers, and the simulated human annotators they are calibrated against.

  exact match     the answer equals the reference (normalised)
  token F1        word overlap with the reference above a threshold chosen on a calibration split
  rubric          every required fact present in a listed phrasing, no wrong version of it
  model judge     a stand-in for an LLM grader, with the biases measured in published studies:
                  noise, a preference for longer answers, and in pairwise mode a preference for
                  whichever answer it reads first

Annotators see the truth through their own error rate; the label used for calibration is the majority
of three.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from .golden import Fact, Item
from .systems import Answer


def words(s: str) -> list[str]:
    return re.findall(r"£|\d+(?:\.\d+)?|[a-z0-9]+", s.lower())


def exact_match(item: Item, a: Answer) -> bool:
    return words(a.text) == words(item.reference)


def token_f1(item: Item, a: Answer) -> float:
    p, r = Counter(words(a.text)), Counter(words(item.reference))
    common = sum((p & r).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def rubric(item: Item, a: Answer) -> bool:
    return Rubric().grade(item, a)


NUM = re.compile(r"\d+(?:\.\d+)?")


def _template(sentence: str, numbers: list[str]) -> str | None:
    t = " ".join(words(sentence))
    if not numbers:
        return t
    for n in numbers:
        if n not in t.split():
            return None
        t = re.sub(rf"\b{re.escape(n)}\b", "<n>", t)
    return t


@dataclass
class Rubric:
    """Accepts a fact in a listed phrasing, or in a phrasing learned from calibration disagreements.

    Learned phrasings are templates with the fact's numbers abstracted ("within <n> days weekends
    excluded"), so one reviewed disagreement covers every item that states the same fact that way.
    """

    learned: frozenset[str] = frozenset()

    def _fact_ok(self, t: str, f: Fact) -> bool:
        if any(" ".join(words(x)) in t for x in f.listed):
            return True
        nums = NUM.findall(f.listed[0])
        for tpl in self.learned:
            if tpl.count("<n>") != len(nums):
                continue
            s = tpl
            for n in nums:
                s = s.replace("<n>", n, 1)
            if s in t:
                return True
        return False

    def grade(self, item: Item, a: Answer) -> bool:
        t = " ".join(words(a.text))
        return all(" ".join(words(f.wrong)) not in t and self._fact_ok(t, f) for f in item.facts)

    def learn(self, pairs: list[tuple[Item, Answer]], human: list[bool], min_support: int = 2) -> Rubric:
        """Turn answers humans passed but the rubric failed into phrasing templates seen at least twice."""
        seen: Counter[str] = Counter()
        # boilerplate: sentences that turn up in answers to more than one slice say nothing about a fact
        slices_of: dict[str, set[str]] = {}
        for item, a in pairs:
            for sent in re.split(r"(?<=[.!?])\s+", a.text):
                slices_of.setdefault(" ".join(words(sent)), set()).add(item.slice)
        for (item, a), ok in zip(pairs, human, strict=True):
            if not ok or self.grade(item, a):
                continue
            t = " ".join(words(a.text))
            for f in item.facts:
                if self._fact_ok(t, f):
                    continue
                nums = NUM.findall(f.listed[0])
                for sent in re.split(r"(?<=[.!?])\s+", a.text):
                    plain = " ".join(words(sent))
                    if len(slices_of.get(plain, ())) > 1 or any(
                        " ".join(words(x)) in plain for g in item.facts for x in g.listed
                    ):
                        continue
                    tpl = _template(sent, nums)
                    if tpl and ((nums and "<n>" in tpl) or (not nums and not NUM.search(tpl))):
                        seen[tpl] += 1
        return Rubric(self.learned | {t for t, c in seen.items() if c >= min_support})


@dataclass(frozen=True)
class ModelJudge:
    """A grader that is right most of the time and leans towards long answers."""

    error: float = 0.08
    length_bias: float = 0.010  # added chance of "pass" per word over 25 words, for wrong answers
    position_bias: float = 0.30  # in pairwise mode, chance of picking the first answer regardless
    seed: int = 5

    def grade(self, item: Item, a: Answer) -> bool:
        rng = random.Random(f"{self.seed}:{item.id}:{a.text}")
        verdict = a.correct
        if rng.random() < self.error:
            verdict = not verdict
        if not verdict and rng.random() < self.length_bias * max(0, a.words - 25):
            verdict = True
        return verdict

    def prefer(self, item: Item, first: Answer, second: Answer) -> int:
        """0 if the first answer is better, 1 if the second; read in this order."""
        rng = random.Random(f"{self.seed}:p:{item.id}:{first.text}|{second.text}")
        if rng.random() < self.position_bias:
            return 0
        s1 = self.grade(item, first) + 0.1 * first.facts_right
        s2 = self.grade(item, second) + 0.1 * second.facts_right
        if s1 == s2:
            return rng.randrange(2)
        return 0 if s1 > s2 else 1

    def prefer_debiased(self, item: Item, a: Answer, b: Answer) -> int | None:
        """Read both orders; a verdict only when they agree, otherwise a tie (None)."""
        x = self.prefer(item, a, b)
        y = 1 - self.prefer(item, b, a)
        return x if x == y else None


@dataclass(frozen=True)
class Annotators:
    errors: tuple[float, ...] = (0.04, 0.07, 0.10)
    seed: int = 9

    def labels(self, a: Answer) -> list[bool]:
        out = []
        for k, e in enumerate(self.errors):
            rng = random.Random(f"{self.seed}:{k}:{a.item_id}:{a.text}")
            out.append(a.correct if rng.random() >= e else not a.correct)
        return out

    def consensus(self, a: Answer) -> bool:
        return sum(self.labels(a)) * 2 > len(self.errors)

    def prefer(self, item: Item, a: Answer, b: Answer) -> int | None:
        """Majority preference: correctness first, then facts stated; None when the annotators see a tie."""
        ka, kb = (self.consensus(a), a.facts_right), (self.consensus(b), b.facts_right)
        if ka == kb:
            return None
        return 0 if ka > kb else 1


def cohen_kappa(x: list[bool], y: list[bool]) -> float:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y, strict=True)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(rows: list[list[bool]]) -> float:
    n = len(rows[0])
    p_yes = sum(sum(r) for r in rows) / (len(rows) * n)
    pe = p_yes**2 + (1 - p_yes) ** 2
    pi = [(sum(r) * (sum(r) - 1) + (n - sum(r)) * (n - sum(r) - 1)) / (n * (n - 1)) for r in rows]
    pbar = sum(pi) / len(pi)
    return (pbar - pe) / (1 - pe)


Grader = Callable[[Item, Answer], bool]
