# Known limitations

- **Simulated answers and annotators.** The golden set, the systems and the human labels are
  generated, so the true quality of every answer is known ([ADR 1](adr/0001-ground-truth-by-construction.md)).
  The numbers describe this simulation, not a particular product.
- **The model judge is a stand-in.** It reproduces three biases reported for LLM graders: random
  error, a lean towards long answers, and a preference for the first answer it reads. It is not a
  real LLM; a real judge plugs in as a `Grader` and should be calibrated the same way.
- **The calibrated rubric learns templates, not meaning.** It still misses paraphrases that change
  the numbers' form (for example "1.2 kg" for "1200 g"), which is why the product slice scores lower.
  New phrasings need another round of human review.
- **Binary grades.** Answers pass or fail. Partial credit and graded rubrics would need a different
  test, such as a paired permutation test on scores.
- **Small regressions need big golden sets.** At 1,200 items the gate catches a 5-point drop in one
  slice less than half the time.
- **Slices are fixed in advance.** A regression confined to a sub-population nobody sliced on is
  invisible to the gate.
