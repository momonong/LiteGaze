# GECO Cross-Subject Generalization Protocol v1.1

Parent protocol: `docs/GECO_GENERALIZATION_PROTOCOL_2026-08-03.md`

Parent protocol lock: `2f5b7aea02a30b0553c2269189b15555fd7b43ae`

Amendment timing: before any model was fitted or any outcome metric was produced

## Why v1 stopped

The first v1 execution stopped on the first participant-trial, as required by its fail-fast rule:

```text
ValueError: duplicate fixation word IDs:
data/geco/population/L1/pp01/trial_10/fixations.csv
```

The source schema contains one `layout.csv` row per word item but multiple chronological fixation rows for words that were revisited. In the first file, 156 fixation rows map to 84 unique word IDs, while the layout contains 111 unique word IDs. For example, word ID 1 has two positive fixation durations (321 ms and 337 ms). The original one-to-one merge assumption was therefore invalid.

No fold model, correlation, confidence interval, or other result was computed before this amendment.

## Sole amendment

Before merging with the layout, group fixation rows by `WORD_ID_WITHIN_TRIAL` and sum all finite positive `reading_time` values. Use `min_count=1`, so a word with no finite duration remains missing rather than becoming zero. A word is considered fixated if and only if the aggregated duration is finite and positive.

This produces total observed word dwell across revisits, applies the same deterministic operation to every participant and trial, and does not use thresholds selected from evaluation results.

All other v1 decisions remain unchanged:

- five balanced subject folds and five balanced trial folds;
- 25 new-reader/new-trial double-holdout cells;
- label-free primary features;
- fixed Ridge alpha of 1.0;
- training-only scaling;
- macro participant Spearman primary endpoint;
- 10,000 participant bootstraps and 100,000 sign flips;
- `cognitive_mass` remains provenance-risk and primary-ineligible;
- CPU-only execution.
