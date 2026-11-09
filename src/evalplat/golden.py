"""A golden set for a customer-support assistant: questions, the facts a correct answer must state, and
the wrong versions of those facts that make an answer incorrect.

Each fact has the phrasings a rubric lists and one more phrasing it doesn't (a paraphrase), so a
rubric judge can be wrong in a realistic way: it misses correct answers worded differently.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SLICES = ["billing", "shipping", "returns", "account", "product"]


@dataclass(frozen=True)
class Fact:
    listed: tuple[str, ...]  # phrasings the rubric accepts
    paraphrase: str  # a correct phrasing the rubric does not list
    wrong: str  # the same fact stated incorrectly


@dataclass(frozen=True)
class Item:
    id: str
    slice: str
    question: str
    facts: tuple[Fact, ...]

    @property
    def reference(self) -> str:
        return " ".join(f.listed[0][:1].upper() + f.listed[0][1:] + "." for f in self.facts)


def _days(n: int) -> Fact:
    return Fact(
        (f"{n} working days", f"{n} business days"), f"within {n} days, weekends excluded", f"{n + 3} working days"
    )


def _money(x: int, what: str) -> Fact:
    return Fact(
        (f"{what} is £{x}", f"{what} costs £{x}"),
        f"you pay £{x} for {what.rsplit(maxsplit=1)[-1]}",
        f"{what} is £{x + 5}",
    )


REGIONS = ["Scotland", "Wales", "Northern Ireland", "the Channel Islands", "London", "the Highlands", "Cornwall"]
PLANS = ["Basic", "Plus", "Pro", "Family", "Student"]
PRODUCTS = [
    "the Ridge 2 tent",
    "the Alpine stove",
    "the Trail 40 backpack",
    "the Summit sleeping bag",
    "the Brook water filter",
    "the Ember headtorch",
]


def _billing(rng: random.Random, i: int) -> Item:
    plan = rng.choice(PLANS)
    price = rng.choice([4, 6, 9, 12, 15, 19])
    day = rng.choice([1, 5, 15, 28])
    q = rng.choice(
        [
            f"How much is the {plan} plan and when am I charged?",
            f"What does {plan} cost per month, and which day is the payment taken?",
        ]
    )
    facts = (
        _money(price, f"the {plan} plan"),
        Fact(
            (f"charged on day {day} of each month", f"billed on the {day} of every month"),
            f"the payment leaves your account each month on day {day}",
            f"charged on day {day + 1} of each month",
        ),
    )
    return Item(f"bil-{i}", "billing", q, facts)


def _shipping(rng: random.Random, i: int) -> Item:
    region = rng.choice(REGIONS)
    n = rng.choice([2, 3, 4, 5])
    fee = rng.choice([3, 4, 5, 7])
    q = rng.choice(
        [
            f"How long does delivery to {region} take and what does it cost?",
            f"What's the delivery time and fee for {region}?",
        ]
    )
    return Item(f"shp-{i}", "shipping", q, (_days(n), _money(fee, "delivery")))


def _returns(rng: random.Random, i: int) -> Item:
    n = rng.choice([14, 30, 60])
    product = rng.choice(PRODUCTS)
    q = rng.choice(
        [
            f"Can I return {product}, and how long do I have?",
            f"What is the return window for {product} and do I need the receipt?",
        ]
    )
    facts = (
        Fact(
            (f"returns are accepted within {n} days", f"you have {n} days to return it"),
            f"send it back no later than {n} days after delivery",
            f"returns are accepted within {n + 30} days",
        ),
        Fact(
            ("a proof of purchase is required", "you need the receipt"),
            "keep the order confirmation because we ask for it",
            "no receipt is needed",
        ),
    )
    return Item(f"ret-{i}", "returns", q, facts)


def _account(rng: random.Random, i: int) -> Item:
    n = rng.choice([15, 30, 60])
    q = rng.choice(
        [
            "How do I reset my password?",
            "I forgot my password, what should I do?",
            "How can I get back into my account if I lose my password?",
        ]
    )
    facts = (
        Fact(
            ("use the forgot password link", "click forgot password"),
            "choose the reset option on the sign-in page",
            "call support to reset it",
        ),
        Fact(
            (f"the reset link expires after {n} minutes", f"the link is valid for {n} minutes"),
            f"you have {n} minutes before the email link stops working",
            f"the reset link expires after {n * 4} minutes",
        ),
    )
    return Item(f"acc-{i}", "account", q, facts)


def _product(rng: random.Random, i: int) -> Item:
    product = rng.choice(PRODUCTS)
    years = rng.choice([1, 2, 3, 5])
    weight = rng.choice([450, 800, 1200, 2100])
    q = rng.choice(
        [
            f"What is the warranty on {product} and how heavy is it?",
            f"How much does {product} weigh, and how long is it guaranteed?",
        ]
    )
    facts = (
        Fact(
            (f"the warranty is {years} years", f"it is guaranteed for {years} years"),
            f"faults are covered for {years} years from purchase",
            f"the warranty is {years + 2} years",
        ),
        Fact(
            (f"it weighs {weight} g", f"the weight is {weight} grams"),
            f"it comes in at {weight / 1000:g} kg",
            f"it weighs {weight + 300} g",
        ),
    )
    return Item(f"prd-{i}", "product", q, facts)


MAKERS = {"billing": _billing, "shipping": _shipping, "returns": _returns, "account": _account, "product": _product}


def golden_set(n: int = 600, seed: int = 11) -> list[Item]:
    rng = random.Random(seed)
    return [MAKERS[SLICES[i % len(SLICES)]](rng, i) for i in range(n)]
