# PROVO Cross-Corpus Zero-Shot Protocol v1

Protocol date: 2026-08-03

Protocol lock: the Git commit that first adds this file, before the complete PROVO
eye-tracking file is downloaded or any aggregate outcome is computed.

Official command (after acquisition):

```powershell
$env:LEXIGAZE_DEVICE = "cpu"
$env:CUDA_VISIBLE_DEVICES = ""
uv run python scripts/evaluate_provo_zero_shot.py
```

## Objective

Test whether a fixed lexical difficulty model trained only on GECO native-English
readers transfers, without fitting or calibration, to the independent PROVO corpus.
Scientific completion does not require a positive result. The protocol is complete
when the frozen model, prespecified comparators, participant-level uncertainty, data
fingerprint, and leakage audit are reported unchanged.

This experiment does not use a question-answer dataset. No PROVO outcome, cloze
norm, LSA score, or predictability field is eligible for feature selection, fitting,
threshold selection, calibration, stopping, or exclusion.

## Independent test data

- Corpus: PROVO (Luke & Christianson, 2018), 84 native-English participants
  reading 55 short English texts.
- Official project: <https://osf.io/sjefs/>
- Official OSF file GUID: `a32be`
- File: `Provo_Corpus-Eyetracking_Data.csv`
- Recorded OSF size: `69,662,713` bytes
- Recorded OSF SHA-256:
  `38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7`
- Local path: `data/provo/raw/Provo_Corpus-Eyetracking_Data.csv`

Before this protocol was locked, inspection was limited to the paper, OSF metadata,
and an HTTP range containing the CSV header and first three rows. No complete file,
aggregate duration, participant metric, correlation, model fit, or test result was
available.

Required PROVO columns are:

- identity: `Participant_ID`, `Word_Unique_ID`, `Text_ID`, `Word_Number`;
- text: `Word`, `Word_Cleaned`, `Word_Length`;
- outcomes: `IA_DWELL_TIME`, `IA_SKIP`.

The run must find exactly 84 participants and 55 text IDs. It records, but does not
precondition on, the complete row count and unique word-item count. Duplicate
`Participant_ID` + `Text_ID` + `Word_Unique_ID` rows, inconsistent text for an item,
or a source hash/size mismatch stop the run.

Total dwell time is parsed as milliseconds. A word is fixated if and only if
`IA_SKIP == 0` and `IA_DWELL_TIME` is finite and strictly positive. A skipped word
must have `IA_SKIP == 1` and no positive dwell time. Contradictions stop the run;
they are not resolved by inspecting downstream metrics. The conditional-duration
target is `log1p(IA_DWELL_TIME)` for fixated rows. No observation is removed based
on duration magnitude.

## GECO training data

- Local source: `data/geco/population/L1/`
- Population: all 18 native-English GECO L1 participants discovered by the frozen
  loader; L2 readers are excluded to match PROVO's native-English population.
- Inputs: every paired `layout.csv` and `fixations.csv` under the L1 tree.
- Repeated positive fixation durations for a word are summed with `min_count=1`, as
  locked by GECO protocol v1.1.
- Target: `log1p` of positive total word dwell in milliseconds.

The run fingerprints every used GECO file. No GECO row is removed based on outcome
magnitude. GECO supplies all fitted parameters; PROVO supplies none.

## Frozen cross-corpus features

Features are computed by exactly the same function from the displayed `Word` text
in both corpora. They do not use corpus-specific norms or reading outcomes.

Token normalization is Unicode NFKC followed by whitespace stripping. Leading and
trailing Unicode punctuation are removed for the lexical-frequency lookup while
internal apostrophes and hyphens are retained. The lookup token is case-folded.

The five prespecified Ridge features are:

1. `log1p` of the count of Unicode letters and numbers in the normalized word;
2. English Zipf frequency from locked `wordfreq` using `zipf_frequency(token, "en")`,
   clipped to `[0, 8]`;
3. relative word position `(zero_based_position / max(item_count - 1, 1))` within
   the displayed GECO trial or PROVO text;
4. whether the first cased character is uppercase;
5. whether the final non-whitespace character is Unicode punctuation.

Rows with empty display text, non-finite position, or non-finite features fail the
run. `Word_Cleaned`, PROVO cloze values, POS tags, LSA scores, GECO surprisal,
GECO attention, and `cognitive_mass` are not model inputs.

## Frozen models

### Conditional duration (primary)

- Model: NumPy Ridge regression with unpenalized intercept.
- Training rows: every fixated GECO L1 row.
- Test rows: every fixated PROVO row.
- Alpha: fixed at `1.0`.
- Feature means and scales: fitted on GECO only.
- No hyperparameter search, early stopping, coefficient clipping, corpus offset, or
  PROVO intercept recalibration.

Prespecified zero-fit comparators are:

- `word_length_only`: Unicode letter/number count, with larger values predicting
  longer duration;
- `lexical_rarity_only`: negative English Zipf frequency, with rarer values
  predicting longer duration.

### Fixation occurrence (secondary)

A fixed L2-penalized logistic regression uses the same standardized features and all
GECO L1 rows. The penalty is `1.0`, the intercept is unpenalized, and deterministic
Newton/IRLS optimization uses at most 100 iterations with tolerance `1e-8`. No
threshold is selected. It is evaluated on all PROVO rows using ROC AUC and Brier
score; this endpoint cannot change the primary conclusion.

## Endpoints and inference

The sole primary endpoint is the unweighted macro-average of per-participant
Spearman correlations between the frozen Ridge score and observed conditional log
dwell time. Participant is the unit of inference; word rows are never treated as
independent replicates.

Prespecified uncertainty and tests use seed `20260804`:

- 10,000 participant bootstrap samples for percentile 95% confidence intervals;
- 100,000 participant-level sign flips for Ridge against zero;
- 100,000 paired participant-level sign flips for Ridge minus word length.

The macro correlations for word length and lexical rarity, participant-level MAE,
and secondary fixation ROC AUC/Brier are reported. They are not tuning objectives.

The frozen decision labels are:

1. **incremental cross-corpus evidence** only if Ridge minus word length is positive
   and its paired two-sided sign-flip `p < 0.05`;
2. **basic lexical transfer only** if Ridge is positive with a bootstrap interval
   excluding zero but criterion 1 is not met;
3. **no confirmed transfer** otherwise.

These labels summarize evidence and do not determine whether the engineering goal
is complete.

## Stopping and reporting rules

1. Commit this protocol before downloading the complete official test file.
2. Download only the identified OSF object and require its recorded size and SHA-256.
3. Execute the complete dataset once with frozen defaults.
4. Integrity or implementation failures stop the run and are recorded. A mechanical
   amendment may be committed only before any aggregate model result is produced.
5. After any primary result is visible, do not change tokens, features, alpha,
   exclusions, outcome mapping, comparators, inference, or decision rules.
6. Report negative and null results as prominently as positive results.
7. Retain the PROVO v1 outputs and use a new corpus/protocol for future development.
8. Run CPU-only. The evaluator must not import or initialize Torch/CUDA.

## Source citation

Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large
eye-tracking corpus with predictability norms. *Behavior Research Methods, 50*,
826-833. <https://doi.org/10.3758/s13428-017-0908-4>
