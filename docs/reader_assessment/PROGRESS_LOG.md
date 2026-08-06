# Reader Assessment v2 Progress Log

## 2026-08-06 — branch `feat/evidence-based-reader-assessment`

### Audit

- Confirmed that v1 ability, proficiency, attention, load, and fatigue outputs were hand-written transforms rather than validated scales.
- Confirmed an arbitrary `88` English score fallback when no struggle words were found.
- Confirmed that all visited words were effectively treated as struggle words in one code path.
- Confirmed that WPM used observed/fixated words divided by dwell time and omitted explicit full-text session time.
- Confirmed that adaptive testing changed passage difficulty and typography together, preventing causal interpretation.
- Confirmed that only six questions across three rounds supported the old capability report.
- Confirmed that the browser received answer keys and that the LLM report could amplify unsupported conclusions.

### Baseline

- Corrected the test command to use `.venv` Python 3.11 rather than the system Python 3.14, which is outside the project constraint and lacked dependencies.
- Baseline v1 inspector/adaptive tests: 13 passed. This established behavioural compatibility but not validity.

### Decisions

- Retire all unsupported 0–100 ability scores instead of recalibrating arbitrary constants.
- Preserve legacy keys as deprecated `null/not_estimated` values to make downstream migration explicit.
- Separate observables, data quality, session evidence, experimental model output, and latent-trait claims.
- Require explicit text length, completion, and elapsed time before calling a rate WPM.
- Treat lexical dwell and early/late change as exploratory session signals only.
- Fix typography during ability evidence collection.
- Move answer scoring to the backend and sign each result.
- Disable live LLM-generated items and reports for measurement.
- Use an uncalibrated adaptive pilot only to build a future item-calibration dataset.
- Keep all validation CPU-only and disallow QA-content fitting.

### Implementation

- Rebuilt `CognitiveInspector` as v2 with robust summaries, Wilson intervals, quality checks, abstention claims, and context-aware WPM.
- Rebuilt Markdown reports around evidence and limitations.
- Added a six-passage, eighteen-item, three-construct English pilot bank with two forms per provisional difficulty tier.
- Added EAP/posterior-SD routing, 4–6 round stopping, fixed layout, server scoring, and signed round provenance.
- Updated the browser UI to display evidence, quality, and abstentions rather than fake scores.
- Added API and metamorphic regression tests.

### Verification

- Focused v2 suite passed before repository-wide verification.
- Static bank audit: 6 unique passages, 18 unique items, all three constructs represented six times, answer-key positions balanced 5/4/5/4, no public answer leakage.
- Quick CPU Monte Carlo: 2,000 simulated participants across matched and shifted-parameter regimes; all software gates passed.
- Quick-run finding: almost every simulated participant used all six rounds. The current bank is suitable for pilot data collection but too small/information-limited to claim efficient variable-length CAT operation.
- Quick-run finding: posterior estimates shrink at the extreme ends of the provisional scale. This reinforces the ban on ability labels before real calibration and bank expansion.

### Deterministic long CPU run

- Command: `.venv\\Scripts\\python.exe -X utf8 -m scripts.run_reader_assessment_validation --participants 20000 --seed 20260806 --output-dir docs/reader_assessment/experiments --name validation_v2` with `CUDA_VISIBLE_DEVICES=''`.
- Total simulated participants: 40,000 across two preregistered software regimes.
- Runtime: 290.418 seconds; no GPU requested; no parameter fitting; no QA content used for fitting.
- Matched assumptions: Spearman 0.8408, RMSE 0.6297, MAE 0.4963, 95% interval coverage 0.9365.
- Hidden shifted item parameters: Spearman 0.8448, RMSE 0.6173, MAE 0.4888, 95% interval coverage 0.9401.
- All frozen engineering gates passed.
- Operational measurement readiness remains `false`.
- Max-round fractions were 0.9893 and 0.9913, confirming that the pilot bank is not yet an efficient variable-length CAT.
- Extreme-bin shrinkage remains material: matched RMSE was 1.0083 below theta -1.5 and 0.8842 above theta 1.5.
- JSON SHA-256: `43FD78668FFC8D0016BFAC34301E2B3D4CE5E31B73DF0881169FECA1C88B4D11`.
- Markdown SHA-256: `06ED869D3BAA3BA1E10AF49B4E3868121623BFDE6302F2C39D2180ABB0FE2A32`.

### Final quality gate and browser verification

- Repository offline quality gate: 88 tests passed, 0 failures, 0 errors, and 0 skipped.
- The quality-gate worker ran with `CUDA_VISIBLE_DEVICES=-1`; PyTorch was never imported, no network or subprocess probes escaped the guard, and the GPU snapshot was unchanged before and after the run.
- Final quality-gate runtime after formatting and copy-boundary cleanup: 2.003 seconds in the worker and 2.287 seconds in the supervisor, below the 240-second timeout.
- Critical Ruff checks passed for the active assessment routes; all six new or rebuilt Python modules pass full Ruff checks and formatting.
- `git diff --check` passed. The remaining Git messages are line-ending notices, not whitespace errors.
- Flask route-map inspection confirmed that only the four v2 adaptive endpoints are active: `start`, `score`, `next`, and `report`.
- An in-browser walkthrough confirmed the first assessment round loads, answer keys and explanations are absent from the public payload, the layout remains fixed at 16 px / 650 px / 1.7, and no browser warnings or errors appear.
- The browser walkthrough exposed two stale UI defects, both fixed before this gate: an unsupported cognitive-load label and an instruction that said two questions although the bank serves three.
- Final diff review also removed a stale “cognitive and typography assessment complete” claim and relabelled coordinate-direction markers as descriptive rather than cognitive events; UI regression tests now freeze those boundaries.
- Offline quality-gate JSON SHA-256: `1F64804A934090CEE57620106A97BBA29A982B1BA0D34B9DF47D17B0783D8BAB`.

### Milestone boundary

- The software, abstention policy, provenance controls, and offline validation harness are ready for a controlled pilot.
- The module is deliberately **not** ready to report general cognitive ability, CEFR/English proficiency, attention, fatigue, or efficient CAT scores.
- The next evidence milestone requires consented participant data, an external English anchor measure, held-out participants/items, item calibration, differential-item-functioning checks, and bank expansion. No question-answer dataset should be used to tune the estimator.
