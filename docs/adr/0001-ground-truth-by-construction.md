# ADR 1: Simulated systems whose true quality is known

## Context

An evaluation platform makes two kinds of claim: that a judge agrees with people, and that a gate
blocks bad releases without blocking good ones. Neither can be checked with real model outputs alone,
because nobody knows the true quality of a real release. "The gate blocked it" doesn't show the gate
was right.

## Decision

Answers come from simulated systems with known, per-slice quality: each answer states each required
fact correctly, wrongly or not at all, in a listed phrasing or a paraphrase. Every answer therefore
carries its true label. Three annotators see that truth through their own error rates (4%, 7%, 10%),
and the majority of three is the human label judges are calibrated against. A release scenario is a
change in the systems' parameters, so whether it *should* be blocked is known before the gate runs.

## Consequences

- The gate's false-alarm rate and power are measured, not assumed: hundreds of simulated releases
  per scenario, each graded against what the scenario actually changed.
- The numbers describe this simulation. The real-world counterpart is the same procedure on a real
  system: collect human labels for a calibration split, measure each judge's kappa before trusting
  it, and size the golden set from a power analysis.
- Swapping in a real model judge or real outputs needs only a new `Grader` or a list of `Answer`s.
