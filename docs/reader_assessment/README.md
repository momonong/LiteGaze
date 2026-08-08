# LexiGaze Reader Assessment v2

> 2026-08-08 update: v3 的研究與測量設計已凍結，但尚未接上 live
> collection。請先讀 [`MEASUREMENT_DESIGN_V3.md`](MEASUREMENT_DESIGN_V3.md)
> 與機器可讀的
> [`reader_assessment_validity_v3.json`](reader_assessment_validity_v3.json)。
> 目前 production/dry-run 行為仍是本文件所述的 v2；v3 不會繞過倫理、內容審查或外部 anchor 授權。

## Outcome

The old "Cognitive Inspector" must not be treated as a validated ability test. Version 2 replaces its arbitrary 0–100 scores with an evidence-bounded measurement pipeline:

1. describe observable behaviour;
2. assess measurement quality;
3. report uncertainty where it is defensible;
4. abstain from latent-trait claims that have not been validated;
5. keep reading assessment separate from typography optimization.

The current pilot can support software and data-collection experiments. It is **not operationally ready** to classify cognitive ability, general reading ability, attention, fatigue, English proficiency, or CEFR level.

## Why v1 was retired

The audit found that v1:

- mapped WPM, regression rate, and mean fixation duration to a 0–100 "reading ability" score using hand-written linear breakpoints;
- mapped the Zipf frequency of "struggle words" to English proficiency and returned `88` when it found no such words;
- subtracted fixed points for regressions and rereads to create an "attention index";
- labelled a second-half/first-half fixation-duration ratio as fatigue;
- changed text difficulty and typography at the same time, then selected a supposedly optimal layout from the confounded result;
- used three passages with two questions each and presented the output as a capability profile;
- sent answer keys to the browser and allowed client-side score construction;
- generated stronger claims through an LLM report even though the underlying evidence did not support them.

Those rules were test-covered, but the tests only protected the formulas. They did not establish construct validity, reliability, precision, fairness, or external validity.

## Measurement contract

### Directly observable in one session

- fixation-group count and robust fixation-duration summaries;
- observed word indices, rereads, and backward transitions;
- explicit full-text reading rate, **only** when full text length, completion, and start/end time are supplied;
- tracking confidence, timestamp quality, estimated sample cadence, and usable event coverage;
- passage-specific associations such as lexical rarity versus dwell, clearly labelled as exploratory session signals.

### Provisional session evidence

- reading fluency for the current passage when explicit elapsed time, full text length, comprehension items, and usable gaze data all exist;
- comprehension performance on the current uncalibrated pilot item bank;
- an experimental theta on the pilot-bank scale, always returned with posterior SD and a 95% credible interval.

### Explicitly not estimated

- general cognitive ability;
- intelligence or clinical status;
- attention;
- fatigue;
- cognitive load without a validated person- and text-conditioned reference model;
- general reading ability;
- English proficiency or CEFR;
- optimal typography during the ability-assessment phase.

Compatibility fields under `user_profile` remain in the API, but unsupported scores are now `null` and marked deprecated.

## Adaptive pilot protocol

- 4–6 passages, three questions per passage.
- Three question constructs: explicit information, inference, and lexical/cohesive interpretation.
- Multiple topics and two forms at each provisional difficulty tier.
- Fixed `16 px / 650 px / 1.7` layout throughout ability evidence collection.
- Answer keys never appear in the public passage payload.
- The server scores responses and signs each round result; later routing and reports reject modified results.
- Routing uses an EAP estimate and a standard-error stopping rule on an explicitly **uncalibrated** pilot scale.
- Gemini/LLM passage generation and LLM ability reporting are disabled in the live v2 routes.

The pilot item parameters are expert seeds. They exist so the software can be simulated and tested; they are not empirical calibration results.

## Evidence behind the design

- The [Standards for Educational and Psychological Testing](https://www.testingstandards.net/) make score interpretation and intended use dependent on validity evidence and require attention to reliability/precision, standard errors, fairness, and replication. A formula that looks plausible is not validation.
- The [Council of Europe CEFR Companion Volume](https://rm.coe.int/cefr-companion-volume-with-new-descriptors-2020/16809ea0d4) describes multiple reading activities and proficiency descriptors. A word-frequency pause heuristic cannot be linked to CEFR without a formal linking study.
- White's controlled experiment found that [word frequency and orthographic familiarity affect fixation behaviour](https://pubmed.ncbi.nlm.nih.gov/18248149/). Fixation duration is therefore partly stimulus-dependent, not a pure person trait.
- Eskenazi and Folk experimentally separated [comprehension-driven and oculomotor-driven regressions](https://pubmed.ncbi.nlm.nih.gov/27873185/). Regressions cannot be converted directly into an attention penalty.
- The EyeScore study predicted external proficiency scores only after collecting [145 ESL participants, standardized-test outcomes, controlled high-frequency eye tracking, and text-aware models](https://aclanthology.org/N18-1180/). Its any-text performance was materially weaker than fixed-text performance, underscoring the need for text and participant holdouts.
- Tywoniw's eye-tracking assessment study found that [different reading tasks required different predictor mixtures and interacted with proficiency, reasoning, and motivation](https://doi.org/10.3389/fcomm.2023.1176986). Gaze metrics alone do not isolate one ability construct.
- A naturalistic comprehension study used [participant-level cross-validation across three studies](https://pubmed.ncbi.nlm.nih.gov/33029808/) before claiming generalization. LexiGaze v2 adopts participant and item/text holdout as a future external-validation requirement.
- Research on computerized adaptive testing shows that [standard-error stopping depends on item-bank information](https://doi.org/10.7333/jcat.v1i1.16). v2 therefore exposes posterior uncertainty and does not treat a fixed three-round path as precise.

## Anti-overfit policy

No model or threshold may be selected from a fixed comprehension question set and then evaluated on that same set.

The validation ladder is:

1. software invariants and synthetic assumptions;
2. participant-group holdout;
3. unseen passage/item-form holdout;
4. joint participant + item holdout;
5. preregistered external confirmation against an independent proficiency/reading measure;
6. subgroup fairness/DIF and test-retest analysis.

Simulation output is labelled software evidence only. It cannot promote the claim status to validated ability.

## Compute policy

- All v2 assessment tests and simulations are CPU-only.
- `CUDA_VISIBLE_DEVICES` is cleared in documented commands.
- The assessment route does not import Torch or call a hosted LLM.
- Larger runs are justified by Monte Carlo precision, not by unused GPU capacity.

## Next empirical milestone

Collect consented pilot responses with passage ID, item responses, explicit reading duration, tracker quality, device/layout metadata, and an independent external reading/proficiency criterion. Freeze a participant split and an item-form split before fitting item parameters. Only then evaluate IRT calibration, DIF, reliability, and criterion validity.
