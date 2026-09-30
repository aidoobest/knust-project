# Changelog

## Round 2 (this commit)
Fixes applied following CAN-DO-VEX-DAT Testbed feedback, round 1 (commit 0e30b0c):

- **Denominator mismatch (GATE-2, Q10):** the confidence-threshold arm was
  previously scored on 147/154 instances while every comparator used 154.
  Fixed: all headline comparisons now use full-coverage argmax scoring
  (`evaluate_full_coverage`). The threshold mechanism is reported separately
  as a risk-coverage result (`selective_subset_from_full`).
- **Seed regime (Q6, Q8, Q23a):** increased from 3 seeds on one fixed
  partition to 10 seeds, each independently redrawing the data partition
  (`grouped_split(df, seed)` called per-seed, not once).
- **Missing ablation cell (Q20):** added the single-layer-head+threshold
  condition, derived post-hoc from saved probabilities.
- **McNemar's odds-ratio bug:** corrected an inverted zero-count branch;
  added explicit Haldane-Anscombe correction, disclosed per-comparison.
- **Interpretability (Q13, Q14):** Integrated Gradients now runs on the
  full Misconception-labelled test subset (37 instances, not 20), reports
  signed attributions, and suppresses tokens with fewer than 5 supporting
  instances.
- **Reproducibility (Q4, Q26):** per-seed, per-model predictions are now
  persisted to `artifacts/` as JSON.
- **Secondary ablation added:** capacity-reduced head variant
  (`EngineeredDistilBERT_v2`) to test the capacity-vs-training-size
  mismatch diagnosis.
