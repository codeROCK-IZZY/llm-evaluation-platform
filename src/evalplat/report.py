"""A self-contained HTML report of the evaluation."""

from __future__ import annotations

import html

from . import gate
from .evaluate import SHOULD_BLOCK, Result

CSS = """
:root { --bg:#0b0e13; --panel:#121720; --line:#232c3a; --ink:#e8ecf2; --ink2:#9aa6b8; --ok:#3fcf8e; --bad:#ff6b6b;
  --mid:#ffb454; --bar:#7fb4ff; font-family: Inter, system-ui, sans-serif; }
* { box-sizing: border-box; } body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5; }
main { max-width: 1180px; margin: 0 auto; padding: 40px 28px 64px; }
h1 { font-size: 30px; letter-spacing:-0.02em; margin: 0 0 6px; } h2 { font-size: 19px; margin: 34px 0 14px; }
.sub, .muted { color: var(--ink2); } .sub { margin: 0 0 24px; }
.tiles { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.tile b { display:block; font-size: 28px; letter-spacing:-0.02em; } .tile span { color: var(--ink2); font-size: 13px; }
table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; overflow:hidden; font-size: 13.5px; }
th, td { padding: 8px 12px; border-bottom:1px solid var(--line); text-align:right; }
th:first-child, td:first-child { text-align:left; }
th { color: var(--ink2); font-weight:500; font-size:12px; }
td.good { color: var(--ok); font-weight:600; } td.bad { color: var(--bad); font-weight:600; }
td.mid { color: var(--mid); font-weight:600; }
.cols { display:grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.bar { display:inline-block; height: 8px; border-radius: 4px; background: var(--bar); vertical-align: middle; }
"""


def _p(x: float) -> str:
    return f"{x * 100:.1f}%"


def _gate_cls(x: float, should: bool) -> str:
    err = (1 - x) if should else x
    return "good" if err <= 0.05 else ("mid" if err <= 0.3 else "bad")


def render(r: Result) -> str:
    cal = r.judges["rubric (calibrated)"]
    raw = r.judges["rubric (listed phrasings)"]
    aa = r.gate["no change (A/A)"]
    reg = r.gate["better overall, returns -10 pts"]
    tiles = [
        (
            f"{cal['kappa_vs_humans']:.2f}",
            f"kappa of the calibrated rubric vs humans (before: {raw['kappa_vs_humans']:.2f})",
        ),
        (_p(aa["statistical"]), f"false alarms on unchanged releases (any-slice rule: {_p(aa['any slice'])})"),
        (_p(reg["statistical"]), f"of slice regressions caught (overall-mean rule: {_p(reg['overall mean'])})"),
        (
            _p(r.pairwise["debiased_agreement_on_decided"]),
            f"pairwise agreement reading both orders (one order: {_p(r.pairwise['single_order_agreement'])})",
        ),
    ]
    t = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in tiles)
    judge_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['kappa_vs_humans']:.3f}</td><td>{_p(v['accuracy_vs_truth'])}</td>"
        f"<td>{_p(v['wrong_passed_long'])} / {_p(v['wrong_passed_short'])}</td>"
        f'<td><span class="bar" style="width:{max(2, int(v["kappa_vs_humans"] * 160))}px"></span></td></tr>'
        for k, v in r.judges.items()
    )
    head = "".join(f"<th>{html.escape(x)}</th>" for x in gate.RULES)
    gate_rows = "".join(
        f"<tr><td>{html.escape(s)}</td>"
        + "".join(f'<td class="{_gate_cls(v[x], SHOULD_BLOCK[s])}">{_p(v[x])}</td>' for x in gate.RULES)
        + "</tr>"
        for s, v in r.gate.items()
    )
    power = "".join(f"<tr><td>{k}</td><td>{_p(v)}</td></tr>" for k, v in r.power.items())
    pw = r.pairwise
    pair = "".join(
        f"<tr><td>{a}</td><td>{_p(b)}</td></tr>"
        for a, b in (
            ("agreement, one reading order", pw["single_order_agreement"]),
            ("picked the first answer when it was worse", pw["picked_first_when_worse_first"]),
            ("agreement when both orders agree", pw["debiased_agreement_on_decided"]),
            ("pairs where both orders agree", pw["debiased_decided_share"]),
        )
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>LLM Evaluation Platform: evaluation</title><style>{CSS}</style></head><body><main>
<h1>Evaluating the evaluation</h1>
<p class="sub">{r.items:,} golden items, simulated systems whose true quality is known, three noisy annotators, and
{r.trials} simulated releases per scenario: every judge and every gate rule is graded against the truth.</p>
<div class="tiles">{t}</div>
<h2>Judges against human labels (held-out half)</h2>
<table><tr><th>judge</th><th>kappa vs humans</th><th>accuracy vs truth</th><th>wrong passed: long / short</th>
<th></th></tr>
{judge_rows}</table>
<h2>Release gate: share of releases blocked</h2>
<table><tr><th>scenario</th>{head}</tr>{gate_rows}</table>
<p class="muted">Green: the right decision at least 95% of the time. The first two scenarios should ship, the last two
should be blocked.</p>
<div class="cols"><div><h2>Detection by golden set size</h2>
<table><tr><th>items (10-point slice regression)</th><th>detected</th></tr>{power}</table></div>
<div><h2>Pairwise judging</h2><table><tr><th>model judge</th><th></th></tr>{pair}</table></div></div>
</main></body></html>
"""
