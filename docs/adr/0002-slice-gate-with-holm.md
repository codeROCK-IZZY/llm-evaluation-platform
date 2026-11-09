# ADR 2: Gate on every slice, with a paired test and a multiple-comparison correction

## Context

Common release gates fail in opposite directions:

- **"The overall score must not drop"** ships a release that improves four slices and breaks a fifth.
  The average hides the regression.
- **"No slice may drop"** blocks almost every release. With 240 items per slice, sampling noise alone
  moves some slice down on nearly every comparison.
- **A significance test per slice without correction** runs six tests per release. Its false-alarm
  rate is roughly six times the nominal level.

## Decision

For the whole set and each slice, compare baseline and candidate on the *same items*. Use the exact
McNemar test, which looks only at the items where the two systems disagree, and adjust the six
p-values with Holm-Bonferroni. Block when some scope is worse by more than a practical margin
(2 points) *and* its adjusted p-value is below 0.05. Report a paired bootstrap confidence interval
next to every difference.

## Consequences

- False alarms stay under 5% on unchanged releases, and real improvements are never blocked.
- Power depends on items per slice, so the golden set is sized from the measured power curve. Here,
  1,200 items are needed to reliably catch a 10-point regression in one slice of five, and 600 items
  catch it about 63% of the time.
- Smaller regressions (5 points) are caught less than half the time at 1,200 items. Catching them
  reliably needs a larger golden set; the gate does not guarantee it.
