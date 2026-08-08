# Pythia-410M Full GECO L2 Replication — Preregistration v2

- Frozen: 2026-08-08 (Asia/Taipei)
- Status: Pythia GECO outcomes remain unopened
- Parent: v1 outcome-blind technical invalidation
- Candidates, revisions, corpus, outcomes, folds, features, alpha, bootstrap,
  decision tiers, and GPU policy: unchanged from v1

## The only v2 change

GECO v1 stopped before outcome access because GPT-2 emitted a standalone ASCII
space token in `Text_ID 4:33`. The token lay exactly between the display items
`words:` and `erything`, so it overlapped neither word under the v1 contract.

v2 admits a zero-overlap token only when its complete source substring is
Unicode whitespace located exactly in the separator gap between two adjacent
display-word spans. It is assigned to the following word and its surprisal is
included in that word's subtoken sum. This mirrors the leading-space convention
used when a byte-level tokenizer fuses the separator and following text into one
token.

Every other ambiguous alignment still fails:

- non-whitespace zero-overlap offsets;
- leading or trailing whitespace without a following display word;
- offsets extending outside the exact adjacent-word gap;
- tokens overlapping multiple words;
- non-monotonic mappings or uncovered words.

The rule is based only on tokenizer offsets and source text. It cannot read or
depend on gaze or reading-time outcomes.

## Re-run boundary

Both GPT-2 and full Pythia features are recomputed under v2 in a new cache. No
partial or complete v1 feature is reused. All other frozen design and decision
clauses remain exactly as described in the v1 preregistration. OneStop remains
closed, and even strong GECO replication cannot change the production model
without independent Reader Assessment v3 word-review evidence.

The machine-readable v2 contract is
[`protocols/2026-08-08-pythia-geco-l2-replication-v2.json`](protocols/2026-08-08-pythia-geco-l2-replication-v2.json).
