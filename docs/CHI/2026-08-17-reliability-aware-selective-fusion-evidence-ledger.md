# Reliability-Aware Selective Fusion: CHI 2027 Evidence Ledger

**Date:** 2026-08-17

**Status:** working research plan; not a validity, accuracy, or deployment claim
**Target:** [CHI 2027 Papers](https://chi2027.acm.org/authors/papers/), Thursday, 10 September 2026, Anywhere on Earth

## Paper decision

The defensible CHI center is a **quality-aware adaptive reading system** that can abstain from unreliable gaze-based personalization and fall back to text/person behavior. Zero-shot Columbia Gaze transfer is enabling negative technical evidence: it reveals a boundary of webcam-gaze generalization and motivates fallback, but it does not evaluate an interactive reading task or a user benefit and therefore is not the paper's core contribution. A label-free Columbia coverage-risk analysis was not preregistered or run and must not be backfilled as confirmatory evidence.

### Candidate contribution sentence

> We contribute a reliability-aware reading architecture that converts session-specific webcam-gaze quality into task-level uncertainty and abstention, preserves a text/person fallback, and evaluates when this strategy improves—or avoids harming—personalized reading assistance across unseen readers, passages, sessions, and capture conditions.

This framing is only earned after the human evidence described below. A system prototype plus public-dataset results is insufficient: CHI evaluates contribution significance, methodological quality, and support for claims ([CHI 2027 submission guide](https://chi2027.acm.org/guide-to-a-successful-submission/); [contribution types](https://chi2027.acm.org/contributions-to-chi/)).

## Research questions

- **RQ1 — Feasibility:** What fraction of participant-session-passage observations remains eligible for word-level gaze after independent calibration, drift, sampling, and tracking-quality checks?
- **RQ2 — Primary utility:** On a joint holdout of unseen participants, passage families, capture sessions, and device groups, does F2 improve three-class word-review probabilities over both F1 text/person evidence and F0 always-on fusion? The primary metric is all-row mean NLL; fixed coverage is secondary.
- **RQ3 — Safe degradation:** Does the gaze branch degrade through exact F1 fallback, and what does the predeclared selective risk-coverage analysis show without outcome-selected thresholds?
- **RQ4 — Repeatability and burden:** How repeatable and burdensome is the two-visit workflow, including completion, calibration, retries, timing, and response-process evidence?

RQ1 is technical enabling evidence; RQ2 and RQ3 define the current frozen
probability-estimation and safety study. Interruption, workload, useful/false
prompt rates, and calibrated trust remain a **future interaction-study extension**
whose outcome, interface exposure, contrasts, and analysis have not yet been frozen.
They cannot be claimed from the current protocol or planning simulation.

## Frozen comparator ladder

| ID | Comparator | Purpose | Promotion status |
|---|---|---|---|
| B0 | Development-partition training prevalence | Minimum calibration reference without confirmation leakage | Required |
| B1 | Lexical controls only | Tests whether simple word properties explain the outcome | Required |
| B2 | Frozen causal text surprisal + lexical controls | Frozen text evidence | Required |
| B3 | B2 + independently collected language/reading anchors | Tests incremental person information without gaze | Required; collection remains blocked until anchors are authorized and frozen |
| G1 | Quality-gated gaze-only features | Isolates gaze signal and abstention behavior | Diagnostic; never rescues low-quality gaze |
| F0 | Always-on text/person/gaze, without quality abstention; development-frozen missing-gaze imputation branch | Establishes the cost of indiscriminate fusion | Required |
| F1 | B2/B3 text/person evidence, without gaze | Deployable fallback | Required |
| F2 | B2/B3 + reliability-aware selective gaze | Proposed system | Candidate |
| O1 | Label-aware/oracle selection | Possible upper bound and error analysis | Future optional diagnostic, not in the current frozen protocol and never deployable |

All model selection must exclude the final participant, passage, session, and capture-condition holdouts. Coverage thresholds must be frozen without using held-out word-review outcomes.

## Claim-to-evidence ledger

| Candidate claim | Evidence currently available | Missing evidence / restriction |
|---|---|---|
| Webcam gaze is vulnerable to person and capture shift. | Prior work establishes substantial accuracy/precision and calibration limits in everyday gaze input ([Feit et al., CHI 2017](https://www.microsoft.com/en-us/research/publication/toward-everyday-gaze-input-accuracy-precision-eye-tracking-implications-design/); [Beyond Basic Tuning, ETRA 2024](https://doi.org/10.1145/3649902.3653346)). LexiGaze has a software-complete measurement runner but no accepted physical measurement run. | Complete frozen real capture; report uncertainty and failure rate, not only mean point error. |
| The current gaze candidate improves subject-held-out performance within its development dataset. | On MPIIGaze, the recorded candidate improved macro angular error from `9.1250°` to `6.9450°` (`23.89%` relative; bootstrap difference CI `[-2.7186°, -1.5944°]`; 14/15 subjects improved). | This is within-dataset research evidence, not proof of webcam deployment accuracy. The Columbia failure below blocks a cross-domain generalization claim. |
| Quality-aware fusion is intrinsically safer than static fusion. | Synthetic aggregate MAE improved, and exact missing-gaze fallback worked. | Frozen synthetic drift failed; no real-user outcome supports safety or benefit. Claim remains unearned. |
| Label-free abstention identifies reliable cases across datasets. | No Columbia coverage-risk result exists. The zero-shot experiment is only a stress test and failure-boundary probe. | The frozen cross-domain gate failed, and coverage-risk was not preregistered or run. A newly frozen score must be calibrated without target labels and evaluated on untouched data before it can guide user-facing behavior. |
| Selective fusion improves reading assistance. | No independent human result yet. | Compare F1, F0, and F2 on frozen post-reading word-review outcomes; report effect sizes, uncertainty, calibration, coverage, and failure cases. |
| Abstention/fallback improves user experience. | No user evidence yet. | Measure false prompts, useful prompts, interruption, workload, and calibrated trust in an interactive study. Do not infer experience from gaze error alone. |
| LexiGaze measures English proficiency, general reading ability, or cognition. | Current design correctly keeps these fields `not_estimated`. | Any such construct needs an independent validated anchor, construct-specific protocol, empirical linking/validation, and appropriately powered sample. Do not derive it from WPM, regressions, gaze quality, or model fit. |

## Current negative findings that must remain visible

### Synthetic drift failure

The frozen quality-aware fusion run reduced aggregate MAE from `0.092053` to `0.089963` (`-0.002089`, about `2.27%`) but worsened the drift condition from `0.113635` to `0.118118` (`+0.004483`). Its decision is `record_failure_without_parameter_changes`; the candidate remains shadow-only. Aggregate improvement must not be presented without this condition-level reversal.

### Columbia zero-shot cross-domain failure

The prior frozen Columbia Gaze transfer result at immutable commit
`d54b731ff07459cc5673e1e5b17a0447e236dc7c` failed its cross-domain
promotion gate. This is evidence that within-dataset MPIIGaze gains do not
establish cross-domain webcam robustness. It is also a useful negative result
for motivating abstention and F1 text/person fallback.

- **Permitted use:** RQ1 diagnostic, failure-boundary figure, and motivation for safe fallback. Any Columbia risk/coverage analysis must be a future, newly frozen analysis and cannot retrofit the prior outcome as confirmation.
- **Not permitted:** a claim of improved reading, user benefit, production webcam accuracy, or successful domain generalization.
- **Artifact path:** `d54b731ff07459cc5673e1e5b17a0447e236dc7c:docs/experiments/2026-08-08-columbia-cross-domain-gaze-v2-run-001.md`
- **Frozen protocol path/hash:** `d54b731ff07459cc5673e1e5b17a0447e236dc7c:docs/experiments/protocols/2026-08-08-columbia-cross-domain-gaze-v2.json`; canonical SHA-256 `6f0f03c60365ac5e7d735cecff24064d8c52a6a28c20a83c257dc94319633fee`.
- **Exact gate and failed metric(s):** zero-gaze reference `12.0455°`; EyePoseTinyCNN-v1 `14.2841°`, delta `+2.2386°`, participant-bootstrap 95% CI `[+2.0703°, +2.4115°]`, `0/56` participants improved; uncalibrated UniGaze `17.6857°`, delta `+5.6402°`, 95% CI `[+5.4423°, +5.8572°]`, `0/56` improved. Both effectiveness gates failed.
- **Label-free risk/coverage result:** `not_preregistered_not_run`. Do not tune or select a score on the same Columbia outcomes and present it as confirmation.

Columbia is a high-resolution, fixed-task gaze-estimation dataset rather than an interactive reading study ([Columbia Gaze dataset](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/)). Even a favorable result would remain enabling technical evidence.

### Current self-development export smoke

The current export contains one completed self-development participant/session,
six passages from six passage families, and 48 post-reading word-review labels:
`39 no_review`, `7 unsure`, and `2 review_needed`. Reading gaze, fixed-target
validation gaze, and uncertainty evidence are eligible for formal export in
`0` sessions; `649` client-roundtrip reading rows remain in the explicitly
unverified diagnostic table. The required branch is therefore F1 text/person
fallback.

This result proves only that behavioral labels remain available while ineligible
gaze stays outside the evidence tables. The single participant and two positive
labels prohibit fitting, effect estimation, accuracy, benefit, or promotion
claims. The deterministic smoke artifacts are
`docs/CHI/experiments/2026-08-17-current-data-fusion-smoke-v1.json`
(SHA-256 `e4fd5d6e90e7234e5b12449670544d4818b5a9ca118b0061a62292a3425d6657`)
and the paired Markdown report (SHA-256
`2f68cdb27a0fe831f8e1c199ccecff0343f8bb51cb3765340b9729e0f97bb444`).

### Assumption-sensitive planning simulation

The frozen CPU-only generator ran 200 replications for each of 54 cells spanning
enrollment `20/40/144/300/600/900`, optimistic/base/pessimistic yield and ICC
bundles, and null/weak/moderate gaze-signal anchors. F2 had to beat both F1 and
F0 in the same replication. Under the moderate synthetic anchor, the first
tested enrollment meeting the diagnostic thresholds was `600` for optimistic
and base assumptions; none of the tested enrollments met them under pessimistic
assumptions. The weak anchor met no joint diagnostic threshold in any scenario.

These are sensitivity results, not recruitment recommendations: no human effect,
practical threshold, nuisance estimate, or independent device-transfer effect
was available. The deterministic artifacts are
`docs/CHI/experiments/2026-08-17-reliability-aware-selective-fusion-power-v1.json`
(SHA-256 `7c806ce162c3070b50f0a0103eff8e9c011292eca5e5c4a2d1c9487b3f038b36`)
and its Markdown report (SHA-256
`7a9f07efc37486175122ef7181714661152d47cc87a63492228f3f0f7d8f6d9d`).

### Novelty and validity risks

- Webcam gaze plus text is not itself novel. GazeReader already combined webcam gaze and language modeling for ESL word-difficulty prediction; its reported full-model F1 (`75.73`) was nearly unchanged without gaze (`75.59`), while cross-document F1 was substantially lower (`56.31`) ([GazeReader, CHI EA 2023](https://doi.org/10.1145/3544549.3585790)). LexiGaze must demonstrate the incremental value and failure-aware use of gaze.
- Prior gaze-contingent reading interfaces already exist, including gaze-driven LLM prompting ([GazePrompt, CHI 2024](https://doi.org/10.1145/3613904.3642878)). Novelty must come from reliability-aware interaction and evidence of when the system should not act.
- Uncertainty estimation and adaptation are active gaze-estimation topics ([UnReGA, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Cai_Source-Free_Adaptive_Gaze_Estimation_by_Uncertainty_Reduction_CVPR_2023_paper.html); [CPE, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_Enhancing_Accuracy_of_Uncertainty_Estimation_in_Appearance-based_Gaze_Tracking_with_CVPR_2026_paper.html)). A new confidence score alone is unlikely to carry a CHI paper; the interaction consequences and human evidence matter.
- Cognitive or English labels inferred from interaction traces risk construct overreach. They stay exploratory covariates or `not_estimated` unless independently validated.

## Explicitly excluded evidence runner

`scripts/run_chi_experiments.py` is **excluded** from this plan and from CHI evidence. Do not run it for the present four-hour work block, and do not use its outputs for claims, model selection, figures, or tables.

Reasons:

- it uses simulated gaze/text targets and simulated multi-feature fusion labels;
- it derives proficiency/fatigue-like variables from WPM and regressions without independent construct validation;
- it includes a trainable PyTorch cross-attention path without frozen participant/passage/capture holdouts;
- its cursor-ground-truth assumptions and hard-coded subject summaries do not supply an independent reading outcome;
- it was not frozen or preregistered for the present RQs.

Legacy outputs, if any, are historical engineering artifacts only.

## Human-study evidence required

Before any recruitment or claim-bearing collection:

1. Obtain the applicable ethics approval/exemption determination and document consent, withdrawal, compensation, privacy, video handling, retention, and researcher contact. CHI requires human-participant work to comply with the researchers' applicable ethical review and asks authors to provide review context ([CHI 2027 Papers](https://chi2027.acm.org/authors/papers/)). Security hardening is not the current optimization target, but research ethics and truthful consent cannot be postponed until after collection.
2. Freeze stimuli, participant/passage/session/device holdouts, exclusion rules, quality bands, coverage thresholds, primary outcome, and analysis before opening the claim-bearing holdout.
3. Use the independent three-level post-reading word-review judgment as the primary outcome. QA/comprehension, gaze, cursor, WPM, regressions, and model confidence must not become training or selection labels for that outcome.
4. Include natural capture shifts: repeat visits, lighting/posture changes, ordinary webcam placement, and at least one device shift when feasible. If gaze quality fails, preserve behavioral labels and invoke the F1 text/person fallback.
5. Compare F1 text/person fallback, F0 always-on fusion, and F2 selective fusion. Report log loss/NLL, normalized ranked probability score, Brier score, calibration, risk-coverage/AURC, and uncertainty intervals. Report comprehension as secondary unless separately powered.
6. Treat a `5–8` participant dress rehearsal and a `20–40` participant feasibility pilot as process checks, not validity confirmation. Determine the confirmatory sample by simulation/power using the frozen estimand and clustered participant/passage structure.
7. Before making interruption, workload, trust, or assistance-timing claims, freeze a separate interaction-outcome contract: the exposed interface behavior, prompt policy, contrasts, instruments, timing, burden and harm outcomes, stopping/exclusion rules, and analysis. The current machine-readable protocol does not authorize those claims.

The process-only dress-rehearsal contract and practical-effect elicitation
procedure are frozen for internal dry-run and review; the participant
invitation/debrief and moderator runbook are implemented as review drafts, not
finalized or authorized participant materials. None of these artifacts
authorizes external recruitment: the
applicable ethics determination, exact approved contacts and participant
materials, compensation-policy binding, independent human cost judgments,
operational budgets, signoff, and actual 5–8-person rehearsal evidence remain
pending. Until those inputs are present, both readiness validators fail closed.

## CHI 2027 go/no-go

The official Papers deadline is **Thursday, 10 September 2026, Anywhere on Earth**. CHI 2027 has no separate abstract deadline; short papers receive the same review process, and an incomplete paper or grossly insufficient methods/data can be desk rejected ([CHI 2027 Papers](https://chi2027.acm.org/authors/papers/)).

**Internal decision date:** **2026-08-24**. If the human-evidence gates below are not met, do not force a CHI 2027 Papers claim from synthetic, public-dataset, or one-person evidence.

**Current Papers status (2026-08-17): `NO-GO`.** The implementation and planning
artifacts are ready for internal dry-run and review, but participant-facing
templates are not finalized or authorized, and ethics context, authorized
anchors, interaction outcomes, a human-approved practical effect threshold, and
claim-bearing evidence are not complete.

### GO only if all are true

- applicable ethics status and participant materials are complete before claim-bearing collection;
- the contribution, RQs, comparator ladder, primary outcome, holdouts, and analysis are frozen;
- real participant evidence supports or meaningfully falsifies RQ2 and RQ3, with uncertainty and failure cases;
- the Columbia and synthetic negative results are reported rather than hidden;
- every main claim maps to a reproducible artifact and a result produced by an eligible frozen runner;
- the paper can be complete and internally reviewed before the official deadline.

### NO-GO for a CHI Papers claim if any are true

- evidence remains synthetic/public-dataset-only;
- the main result is selected after opening the final holdout;
- user benefit is inferred from gaze error or model confidence;
- cognitive/English constructs lack independent validation;
- the paper depends on excluded `scripts/run_chi_experiments.py` outputs;
- ethics, consent, or data-handling requirements are unresolved.

On no-go, preserve the full-paper protocol and target a later completed study. If a bounded work-in-progress is appropriate, evaluate the separate [CHI 2027 Posters deadline (21 January 2027)](https://chi2027.acm.org/) without relabeling incomplete validation as a full contribution.

## Four-hour artifact ledger

| ID | Artifact | Status / decision | Path | Hash or immutable ID |
|---|---|---|---|---|
| A1 | CHI positioning and evidence ledger | Establishes the core/enabling split; current Papers status is NO-GO | `docs/CHI/2026-08-17-reliability-aware-selective-fusion-evidence-ledger.md` | Self-referential file; bound by the delivery commit rather than an embedded self-hash |
| A2 | Frozen Columbia cross-domain protocol | Complete on the prior immutable experiment commit | `d54b731ff07459cc5673e1e5b17a0447e236dc7c:docs/experiments/protocols/2026-08-08-columbia-cross-domain-gaze-v2.json` | canonical SHA-256 `6f0f03c60365ac5e7d735cecff24064d8c52a6a28c20a83c257dc94319633fee` |
| A3 | Columbia zero-shot result | Negative cross-domain gate; coverage-risk not preregistered or run | `d54b731ff07459cc5673e1e5b17a0447e236dc7c:docs/experiments/2026-08-08-columbia-cross-domain-gaze-v2-run-001.md` | aggregate result SHA-256 `05b3540ab5c6d0f3fad6d5de7a28e0ab9df46e393a40606a0fa45c7d7a33aa15` |
| A4 | Synthetic drift evidence | Frozen failure; shadow-only | `docs/experiments/2026-08-07-quality-aware-text-fusion-v2-run-001.md` | file SHA-256 `4aed1be3653438359dcb2631bfe2f8955e0a8e041ecf42cc8d07f867f917dc7f` |
| A5 | Frozen selective-fusion planning protocol | Planning-only; no collection, recruitment, or promotion authorization | `docs/CHI/protocols/2026-08-17-reliability-aware-selective-fusion-v1.json` | SHA-256 `f9fad8c5ee32ddb6d7b532a0231aeb5eed33a3d9c25185ec62c43bb115489b23` |
| A6 | Leakage-safe evaluator | F0/F1/F2, exact fallback, all-row hybrid risk, normalized RPS, label-shuffle sentinel, and diagnostic crossed-cluster bootstrap; statistical-method review still required | `core/cognition/selective_fusion_evaluation.py` | file SHA-256 `7ea88b00ffa55a696e7a1c7d22418c1c0a4940ec1d1d00e5c56c0a5d6c072450` |
| A7 | Synthetic yield/power sensitivity | 54 cells x 200 replications; no human outcomes; not a recruitment recommendation | `docs/CHI/experiments/2026-08-17-reliability-aware-selective-fusion-power-v1.json` | SHA-256 `7c806ce162c3070b50f0a0103eff8e9c011292eca5e5c4a2d1c9487b3f038b36` |
| A8 | Current-data export/schema smoke | 1 self-development participant, 48 reviews, 0 gaze-eligible sessions; no fit or effect estimate | `docs/CHI/experiments/2026-08-17-current-data-fusion-smoke-v1.json` | SHA-256 `e4fd5d6e90e7234e5b12449670544d4818b5a9ca118b0061a62292a3425d6657` |
| A9 | Human-study readiness delta | Software contract implemented and participant-facing templates drafted for internal review only; external collection remains NO-GO pending ethics/contact/material authorization, human practical-effect judgments and budgets, signoff, the 5–8-person process rehearsal, a separate interaction-outcome contract, and claim-bearing evidence | This ledger, `Human-study evidence required` | Owner: research team before 2026-08-24 internal decision |
| A10 | CPU-only full repository quality gate | `549/549` passed in 223.762 s, 0 failures/errors/skips; CUDA hidden, Torch not imported, network/process probes blocked, artifact changes empty; GPU pre/post snapshots both 0% and 191 MiB (peak not continuously measured) | `docs/CHI/experiments/2026-08-17-chi-selective-fusion-quality-gate.json` | SHA-256 `751fbb3e81947b9a983345f2f0f89b8946ad60db8e8e878f6cf5d32a1bf394aa` |
| A11 | Frozen process-only dress-rehearsal contract | Exact outcome-blind `DR01`–`DR08` slots; 5–8 participants; no extension to chase a gate; rehearsal rows excluded from confirmation; unfinished and no-capture opportunities retained in process denominators | `docs/CHI/protocols/2026-08-17-dress-rehearsal-process-measurement-v1.json` | SHA-256 `6f6264a58e820e47c414f1e86fd499dccb4930a587258cabc694ba80e7c610bd` |
| A12 | Practical-effect and operational-cost elicitation worksheet | Frozen NLL grid, three cost contexts, both F1/F0 comparisons, independent-role unanimity rule, and non-compensating operational co-gates; human ratings, budgets, and signoff remain pending, so no threshold is selected | `docs/CHI/protocols/2026-08-17-practical-effect-cost-elicitation-v1.json` | SHA-256 `0e59975d1b1aa4d3359fd8169d9eb5096098495c55710ccad7ab22c728db3459` |
| A13 | Participant-facing rehearsal readiness bundle | Invitation, debrief, moderator runbook, and fail-closed approval/readiness audit are implemented; current verdict is `NO_GO_external_participants` and no external participant is authorized | `docs/participant_study/`, `scripts/audit_dress_rehearsal_readiness.py` | Bound by the delivery commit; the audit script SHA-256 is `ff4c95188255e2ca57752f1df5abd58c41d9834c8938db9032b37dacaef71120` |

## Primary-source anchors

- [CHI 2027 Papers](https://chi2027.acm.org/authors/papers/)
- [CHI 2027 Guide to a Successful Submission](https://chi2027.acm.org/guide-to-a-successful-submission/)
- [Contributions to CHI](https://chi2027.acm.org/contributions-to-chi/)
- [GazeReader, CHI EA 2023](https://doi.org/10.1145/3544549.3585790)
- [GazePrompt, CHI 2024](https://doi.org/10.1145/3613904.3642878)
- [Toward Everyday Gaze Input, CHI 2017](https://www.microsoft.com/en-us/research/publication/toward-everyday-gaze-input-accuracy-precision-eye-tracking-implications-design/)
- [Beyond Basic Tuning, ETRA 2024](https://doi.org/10.1145/3649902.3653346)
- [UnReGA, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Cai_Source-Free_Adaptive_Gaze_Estimation_by_Uncertainty_Reduction_CVPR_2023_paper.html)
- [CPE uncertainty estimation, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_Enhancing_Accuracy_of_Uncertainty_Estimation_in_Appearance-based_Gaze_Tracking_with_CVPR_2026_paper.html)
- [Columbia Gaze dataset](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/)
