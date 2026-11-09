"""`evalp`: compare two systems on the golden set, or run the evaluation of the evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cmd_compare(args: argparse.Namespace) -> int:
    import numpy as np

    from . import gate, golden, judges
    from .evaluate import BASELINE, candidate
    from .systems import run

    items = golden.golden_set(args.items)
    by = {i.id: i for i in items}
    recall = {args.slice: args.slice_recall} if args.slice else {}
    base = run(BASELINE, items, args.seed)
    cand = run(candidate("cand", recall), items, args.seed + 1)
    # calibrate the rubric on a separate set of answers labelled by the annotators, then grade with it
    calib = run(BASELINE, golden.golden_set(600, seed=99), 7)
    cal_items = {i.id: i for i in golden.golden_set(600, seed=99)}
    ann = judges.Annotators()
    grade = judges.Rubric().learn([(cal_items[a.item_id], a) for a in calib], [ann.consensus(a) for a in calib]).grade
    b = np.array([grade(by[a.item_id], a) for a in base])
    c = np.array([grade(by[a.item_id], a) for a in cand])
    d = gate.decide(gate.compare([a.slice for a in base], b, c))
    for x in d.comparisons:
        print(
            f"{x.scope:9} {x.base:6.1%} -> {x.cand:6.1%}  {x.diff * 100:+5.1f} pts  "
            f"CI [{x.ci[0] * 100:+.1f}, {x.ci[1] * 100:+.1f}]  p_adj {x.p_adj:.3g}"
        )
    print("BLOCKED: " + "; ".join(d.reasons) if d.blocked else "ship")
    return 1 if d.blocked else 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import check, run_eval, to_json, to_markdown
    from .report import render

    r = run_eval(n_items=args.items, trials=args.trials)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(to_json(r), newline="\n", encoding="utf-8")
    (out / "evaluation.md").write_text(to_markdown(r), newline="\n", encoding="utf-8")
    (out / "report.html").write_text(render(r), newline="\n", encoding="utf-8")
    print(f"{args.trials} trials per scenario ({r.seconds:.0f}s) -> {out}")
    if args.gate:
        problems = check(r)
        for p in problems:
            print(f"GATE FAILED: {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalp", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="baseline against a candidate, slice by slice, with the gate's decision")
    c.add_argument("--items", type=int, default=1200)
    c.add_argument("--slice", choices=["billing", "shipping", "returns", "account", "product"])
    c.add_argument("--slice-recall", type=float, default=0.83)
    c.add_argument("--seed", type=int, default=1)
    c.set_defaults(fn=cmd_compare)
    e = sub.add_parser("evaluate", help="judges against humans, pairwise bias, the gate's error rates")
    e.add_argument("--items", type=int, default=1200)
    e.add_argument("--trials", type=int, default=200)
    e.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    e.add_argument("--gate", action="store_true")
    e.set_defaults(fn=cmd_evaluate)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
