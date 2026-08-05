# Gaze capture-independence gate and next experiment preregistration

- Date: 2026-08-05
- Branch: `experiment/gaze-motion-shift-robustness`
- GPU budget: zero for the provenance phase; CPU only for the planned model run
- Participant images opened by the provenance audit: no
- Question/answer datasets used: none

## Goal

Measure robustness to a genuinely different physical capture without treating
two artifacts of one browser recording as independent evidence. The immediate
deliverable is a provenance-safe dataset and split gate. A model improvement is
eligible for promotion only after the data gate passes.

## Exploratory trigger

An aggregate metadata check found 46 adjacent same-label session pairs. Forty-
five were less than one hour apart and 40 changed between a session with a raw
video and one without it. The current browser flow stores direct frames first,
then uploads the same recording into a newly created session. These observations
are diagnostic only; no accuracy comparison was run on those pairs.

## Frozen independence rules

The following rules were fixed before implementing or running the formal audit:

1. Every new physical collection receives a random `capture_run_id`.
2. Direct frames and video-extracted frames from that collection share the same
   ID. The video session additionally records the direct `source_session_id`.
3. Sessions sharing a run ID or a source-session edge form one capture group.
4. Legacy same-label sessions without provenance and separated by less than 24
   hours form one conservative capture group.
5. A cross-capture benchmark is ready only when at least five operational labels
   each have at least two independent groups, providing at least ten groups in
   total. Labels are not asserted to be unique people.
6. Malformed metadata or conflicting provenance is a hard failure.
7. Audit output is aggregate-only and must not expose participant labels or
   session IDs.

## Frozen model-evaluation rule

If the independence gate later passes, outer evaluation holds out a complete
capture group. Direct/video artifacts and motion bursts can never be split.
Hyperparameters must be selected only inside the remaining capture groups. A
challenger is promoted only if its macro held-out mean improves by both at least
5 pixels and 5% over the gaze-polynomial baseline. Synthetic perturbations may
be reported as stress tests, but never as real-world accuracy.

## Planned execution

1. Persist capture provenance through direct and video collection paths.
2. Add deterministic offline audit and leakage regression tests.
3. Run the formal audit on the frozen historical snapshot.
4. If historical data is not ready, collect a new 65-sample motion-diverse run
   and evaluate with leave-one-motion-block-out validation on CPU.
5. Reserve a separate capture run for confirmation before making any general
   cross-session accuracy claim.

## Results

Formal command:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.audit_gaze_session_independence `
  --json-output docs\experiments\results\2026-08-05-capture-independence-baseline.json
```

| Measure | Result |
| --- | ---: |
| Sessions / manifest rows | 83 / 1,019 |
| Explicitly identified capture runs | 0 |
| Conservative capture groups | 38 |
| Sessions collapsed into shared groups | 80 sessions in 35 groups |
| Legacy adjacency links applied | 45 |
| Operational labels | 36 |
| Labels with at least two independent groups | 1 |
| Independent runs belonging to repeat labels | 2 |
| Malformed rows / provenance conflicts | 0 / 0 |
| Source SHA-256 | `533619f51706cfe20f04fc2698daab6ba03a154627472689c2805fe6a88bccbc` |

Status: `not_ready`. The frozen gate requires five repeat labels and ten
independent runs. No model comparison was run, because doing so on the linked
direct/video pairs would overstate generalization. GPU use remained zero.

The next evidence-producing action is a new `motion_robust` capture. It may be
used for leave-one-motion-block-out exploration; a separately identified run
must remain untouched for confirmation before any cross-capture claim.

## Follow-up

One new 65-row capture subsequently passed the motion-coverage gate and was
evaluated only with nested motion-block holdout. See
[`2026-08-05-gaze-motion-run-001.md`](2026-08-05-gaze-motion-run-001.md). It is
one independent capture and therefore does not change this historical
cross-capture gate to `ready`; a separate confirmation run is still required.
