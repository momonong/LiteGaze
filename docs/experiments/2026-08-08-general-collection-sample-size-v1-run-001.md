# General collection sample-size sensitivity — run 001

Date: 2026-08-08
Protocol: `general-collection-sample-size-v1`
Compute: CPU only; 50,000 Monte Carlo iterations per cell; no participant outcomes; GPU not used.

## Question

How many people must be enrolled to have a reasonable chance of ending with all three of the following after two visits: 52 paired behavioral participants, 30 paired word-level gaze candidates, and 20 paired participants in a subgroup whose prevalence is assumed to be 25%? The paired model comparison is calculated at the participant level so that 96 labels from one person are not treated as 96 independent people.

## Frozen assumptions

The preregistered sensitivity grid is in `protocols/2026-08-08-general-collection-sample-size-v1.json`. It varies visit-1 completion, visit-2 retention, behavioral usability, and independent gaze-quality eligibility. It tests enrollment sizes from 48 through 256 and paired standardized effects of 0.30, 0.40, and 0.50.

These are assumptions, not estimates. Blinded aggregate rates from moderated rehearsals must replace them before formal recruitment.

## Result

The smallest enrollment reaching at least 80% joint probability for all three yield targets, while also exceeding 80% expected power for a standardized paired effect of 0.40, was:

| Scenario | Minimum enrolled | Expected paired behavioral | Expected word-gaze candidates | Expected 25% subgroup | Joint target probability |
|---|---:|---:|---:|---:|---:|
| Optimistic | 128 | 101.6 | 71.1 | 25.4 | 90.7% |
| Base | 144 | 97.3 | 53.5 | 24.3 | 86.0% |
| Pessimistic | 208 | 104.8 | 36.7 | 26.2 | 84.2% |

At 96 enrolled under the base assumptions, the expected paired behavioral count was 64.9 and expected word-gaze count was 35.7, but the chance of reaching 20 participants in the assumed 25% subgroup was only 18.5%; joint target probability was 17.5%. The expected paired power was 67.5% for effect 0.30 and 89.5% for effect 0.40.

## Decision

- Do not advertise a single final sample size yet. Use 1–3 research-team sessions only for software dress rehearsal, then a small ethics-covered feasibility wave to replace attrition and gaze-quality assumptions.
- Treat 144 as the current base-scenario planning number for a fully frozen cohort that must satisfy all three yield targets, not as authorization to recruit and not as a number inferred from existing outcomes.
- If any models, thresholds, exclusions, passages, or utility definitions are tuned on a development cohort, that cohort cannot supply the confirmation result. Freeze everything and recruit a new confirmation cohort, or use a separately preregistered honest cross-fitting design.
- If subgroup recruitment is not targeted and the count remains below the declared target, report uncertainty descriptively; do not claim subgroup equivalence.
- Behavioral word-review labels remain available when gaze abstains. Word-level gaze claims require the independent start/end quality band and cannot be rescued by lowering thresholds after outcomes are seen.

Machine-readable results: `results/2026-08-08-general-collection-sample-size-v1.json`.
