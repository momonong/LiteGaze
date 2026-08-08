# Reader Assessment v3 Fusion Coverage Simulation

- Protocol: `reader-assessment-v3-fusion-coverage-v1`
- Device: `cpu` (GPU used: `False`)
- Seed: `20260808`; replicates per cell: `200`
- Runtime: `17.4355` seconds

This simulation checks assignment coverage only. It does not simulate a human
effect, fit a model, inspect question content, or establish a sample-size or
validity threshold.

| Candidate | N | Passage families | Confirmation families | Min total exposure p05 | Min dev-person x dev-passage p05 | Min joint confirmation/passage p05 | Joint confirmation word labels median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact_18 | 300 | 18 | 3 | 100 | 50 | 14 | 480 |
| compact_18 | 600 | 18 | 3 | 200 | 105 | 30 | 960 |
| compact_18 | 900 | 18 | 3 | 300 | 162 | 49 | 1440 |
| balanced_36 | 300 | 36 | 9 | 50 | 21 | 4 | 720 |
| balanced_36 | 600 | 36 | 9 | 100 | 48 | 11 | 1440 |
| balanced_36 | 900 | 36 | 9 | 150 | 75 | 19 | 2160 |
| diverse_48 | 300 | 48 | 12 | 37 | 14 | 2 | 720 |
| diverse_48 | 600 | 48 | 12 | 75 | 34 | 6 | 1440 |
| diverse_48 | 900 | 48 | 12 | 112 | 53 | 13 | 2160 |

## Decision

Use `diverse_48` as the fusion stimulus-pool target. It is the only tested
candidate with 12 independent confirmation passage families. `compact_18`
has only three, so a passage-resampling interval would be unstable regardless
of how many labels are collected from those same three passages.

This does **not** mean 48 passages or any displayed N is automatically enough.
The final participant count still requires a frozen effect-size/utility model,
attrition assumptions, subgroup cells, and cluster-aware power simulation.
Passage difficulty, genre, and domain balance also require a later frozen
assignment manifest; this run balances only study partitions and exposure.
Reading-item/testlet calibration is a separate study because repeating more
word labels within the fusion study does not create independent psychometric
item evidence.
