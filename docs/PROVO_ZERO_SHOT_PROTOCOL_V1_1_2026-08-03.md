# PROVO Cross-Corpus Zero-Shot Protocol v1.1

Parent protocol: `docs/PROVO_ZERO_SHOT_PROTOCOL_2026-08-03.md`

Parent protocol lock: `8356a44a8dd9df06c3eba41a96275f68feb0ad63`

Frozen v1 implementation: `37322880dcdfd96dd466fa58eff4714182944e3b`

Amendment timing: after the first full-file schema load failed, but before any
GECO model was fitted and before any PROVO prediction, correlation, confidence
interval, p-value, coefficient, or aggregate model result was produced.

## Why v1 stopped

The first v1 execution loaded and fingerprinted GECO L1, then stopped while
validating PROVO identity/text fields:

```text
ValueError: PROVO text IDs, word numbers, and word lengths must be finite
```

The official CSV is an EyeLink Data Viewer Interest Area report joined to PROVO
predictability annotations. The annotation-side `Word_Unique_ID`, `Word_Number`,
`Word`, and `Word_Cleaned` fields contain `NA` for 57 displayed interest areas per
participant. `Word_Length` is additionally `NA` for 27 distinct annotated word
items. These missing annotation values do not mean that the displayed EyeLink
interest areas are absent:

- `IA_ID`, `IA_LABEL`, `Text_ID`, and `TRIAL_INDEX` are finite for every row;
- `Participant_ID` + `Text_ID` + `IA_ID` is unique for every row;
- each `Text_ID` + `IA_ID` has one consistent `IA_LABEL` across participants;
- there are still exactly 84 participants and 55 text IDs.

No rows or participants were selected from an outcome metric during this schema
diagnosis.

The v1 fixation mapping was also semantically incorrect. The official SR Research
reading-measures guidance identifies `IA_DWELL_TIME` as total duration and uses
`IA_SKIP` for the first-pass fixation ratio. Consequently, `IA_SKIP == 1` can coexist
with positive total dwell when an initially skipped word is visited later. This occurs
in the official file and is not a contradiction.

Official field-semantics references:

- <https://www.sr-research.com/support/thread-336.html>
- <https://www.sr-research.com/data-viewer/>

## Mechanical schema amendments

### Displayed item identity and text

For every official CSV row, use the complete EyeLink fields:

- participant: `Participant_ID`;
- passage: `Text_ID`;
- word item and within-passage position: integer `IA_ID`;
- displayed text: non-empty `IA_LABEL`.

`Word_Unique_ID`, `Word_Number`, `Word`, `Word_Cleaned`, and `Word_Length` remain
available for source auditing but are not required to be non-missing and are not
used as model inputs. The frozen Unicode feature function is applied to `IA_LABEL`
in PROVO and to the displayed `WORD` field in GECO.

No official CSV row is dropped by this amendment.

### Total fixation occurrence

A PROVO word is considered fixated if and only if `IA_DWELL_TIME` is finite and
strictly positive. This matches the GECO definition based on positive summed total
dwell. Zero or missing total dwell is non-fixated.

`IA_SKIP` is retained only as a source audit count for first-pass skipping. It does
not define the secondary any-fixation outcome and it cannot trigger a contradiction
with positive total dwell.

The conditional-duration target remains `log1p(IA_DWELL_TIME)` for positive total
dwell. No row is removed based on duration magnitude.

## Decisions that remain frozen

All non-schema decisions from v1 are unchanged:

- all 18 GECO L1 readers train the models; all 84 PROVO readers are test-only;
- the five lexical features and identical Unicode normalization rules;
- fixed Ridge alpha `1.0` and fixed logistic penalty `1.0`;
- GECO-only feature scaling and no PROVO calibration or corpus offset;
- word length and lexical rarity comparators;
- macro participant Spearman as the sole primary endpoint;
- 10,000 participant bootstraps and 100,000 participant sign flips;
- the three prespecified evidence labels and their decision thresholds;
- no QA data, PROVO cloze/LSA/predictability norms, GECO `cognitive_mass`, GPU,
  Torch, hyperparameter search, or test-result-driven exclusions.

The next complete execution is PROVO protocol v1.1. After its first primary result
is visible, it is frozen regardless of outcome.
