"""Simulated versions of the assistant under test, each with known per-slice quality.

An answer states each required fact correctly, wrongly (a hallucination) or not at all, in one of the
listed phrasings or a paraphrase, padded with filler. Because the generator knows which facts it got
right, every answer carries its true label, and every judge and every gate can be graded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .golden import SLICES, Item

FILLER = [
    "Thanks for getting in touch.",
    "I hope that helps!",
    "Let me know if there is anything else.",
    "Happy to explain further.",
    "Our team is always glad to help with questions like this.",
    "You can also find this in the help centre.",
    "Have a great day.",
    "Many customers ask about this, so you're in good company.",
]


@dataclass(frozen=True)
class Answer:
    item_id: str
    slice: str
    text: str
    correct: bool  # every fact stated correctly, nothing wrong
    facts_right: int
    words: int


@dataclass(frozen=True)
class System:
    name: str
    recall: dict[str, float]  # per slice: chance each fact is stated
    hallucination: float = 0.03  # chance a stated fact is stated wrongly
    paraphrase: float = 0.2  # chance a correct fact uses the unlisted phrasing
    verbosity: float = 1.0  # mean filler sentences
    extra: dict[str, float] = field(default_factory=dict)

    def answer(self, item: Item, rng: random.Random) -> Answer:
        parts: list[str] = []
        right = 0
        ok = True
        for f in item.facts:
            if rng.random() > self.recall[item.slice]:
                ok = False
                continue
            if rng.random() < self.hallucination:
                parts.append(f.wrong)
                ok = False
                continue
            parts.append(f.paraphrase if rng.random() < self.paraphrase else rng.choice(f.listed))
            right += 1
        k = min(len(FILLER), max(0, int(rng.expovariate(1 / self.verbosity) + 0.5))) if self.verbosity else 0
        sents = [p[:1].upper() + p[1:] + "." for p in parts]
        if not sents:
            sents = ["I'm not sure about that, please contact support."]
        text = " ".join([*rng.sample(FILLER, k // 2), *sents, *rng.sample(FILLER, k - k // 2)])
        return Answer(item.id, item.slice, text, ok and right == len(item.facts), right, len(text.split()))


def flat(name: str, recall: float, **kw: float) -> System:
    return System(name, dict.fromkeys(SLICES, recall), **kw)  # type: ignore[arg-type]


def run(system: System, items: list[Item], seed: int) -> list[Answer]:
    rng = random.Random(f"{system.name}:{seed}")
    return [system.answer(it, rng) for it in items]
