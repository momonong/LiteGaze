# OneStop Outcome-Blind Confirmation Preregistration (v1)

- Frozen: 2026-08-05 (Asia/Taipei)
- Branch: `research/text-model-confirmation-corpus`
- Parent evidence: Provo + GECO L2 generalization Run 001
- Status at freeze: **no OneStop data row or reading-time value inspected**
- Compute: CPU only; GPU forbidden for v1

## Why this is the next bounded milestone

The previous run found a small, consistent held-out contribution from frozen
GPT-2 causal surprisal on Provo and GECO L2. Both corpora had historical project
exposure, so they cannot support a final untouched confirmation claim. A
repository-wide search at the parent commit found no OneStop implementation,
result, or cached source. This protocol freezes the third-corpus test before any
OneStop outcome is read.

## Corpus selection

| Candidate | Decision | Reason fixed before outcome access |
| --- | --- | --- |
| OneStop ordinary reading | Selected | 360 English L1 readers, 30 articles/162 paragraphs, a separately downloadable ordinary-reading subcorpus, direct OSF identity, and enough complete articles for five untouched folds. |
| CELER | Rejected for this run | The eye data can be reconstructed only after obtaining licensed PTB-WSJ and BLLIP source texts from LDC, which prevents a self-contained reproducible acquisition path. |
| MECO English | Deferred | It is open and naturalistic, but its 12 texts provide a materially weaker complete-text split for this confirmation. |
| Dundee | Deferred | Access and redistribution are less reproducible than the selected public release. |

Only public corpus descriptions, the official variable documentation, the
official download script, and HTTP/OSF file metadata were consulted. Published
model effects and OneStop outcome values were not used to choose features or
thresholds.

## Frozen source identity

- Dataset/release: OneStop Eye Movements 1.0
- File: `ia_Paragraph_ordinary.csv.zip`
- Official OSF download ID: `xkgfz`
- Expected bytes: `177291322`
- Expected SHA-256: `8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9`
- Eye-tracking data/code license: CC BY 4.0
- Underlying text/annotations license: CC BY-SA 4.0

The source archive stays under ignored `data/` storage. A hash mismatch is an
identity failure, not permission to silently accept a newer file.

## QA-overfit boundary

OneStop's materials include comprehension questions, but v1 is deliberately not
a QA experiment:

1. use only the ordinary regime (`question_preview == false`), so the question
   was not shown before paragraph reading;
2. use only first reading of the paragraph, never the question, answer, QA, or
   feedback interest periods;
3. never read question identity/text, answers, selected answer, correctness,
   critical/distractor spans, comprehension score, or any other QA label;
4. never use the dataset's precomputed GPT-2 surprisal, frequency, word length,
   syntax, or semantic annotations;
5. compute all permitted lexical and GPT-2 fields independently from
   `IA_LABEL` only.

The remaining scope is still *reading for comprehension*, because participants
answered a question after each paragraph. Conclusions will not be generalized
to task-free browsing without a separate corpus.

## Inclusion and identity rules

- File/subcorpus: ordinary `ia_Paragraph` only.
- `practice_trial == false` and `article_id != 0`.
- `question_preview == false`.
- `repeated_reading_trial == false`.
- `difficulty_level == "Adv"` so only original Guardian text is analyzed; the
  simplified paired version is excluded rather than treated as independent.
- Read exactly the columns listed in the machine-readable protocol. Extra
  columns may exist in the archive but are never loaded by the analysis.
- Hold out complete global articles using `(article_batch, article_id)`, not
  paragraphs or random rows. The expected design has 30 article groups.
- Reset causal context at every displayed paragraph sequence. Known layout and
  hyphenation variants receive separate sequence hashes; they never cross an
  article fold.
- Exclude the first word of every displayed sequence because it has no left
  context.
- For each outcome, retain only finite positive durations and apply `log1p`.

If the source schema cannot implement these rules exactly, v1 is invalidated.
A corrected v2 must be frozen before reading outcomes.

## Frozen models

No feature or hyperparameter search is permitted.

- `word_length_only`: current log character length.
- `M0 lexical`: current/previous length and independently computed Zipf
  frequency, relative position, capitalization, punctuation, and GPT-2 subtoken
  count.
- `M1 causal surprisal`: M0 plus current and previous frozen GPT-2-small causal
  surprisal.
- Ridge alpha `1.0`; training-fold-only standardization.
- Five deterministic complete-article folds; seed `20260805`.
- A shuffled-training-target sentinel uses the M1 feature set.
- Entropy is excluded because it failed the prior preregistered gate. Re-testing
  it here would spend the untouched corpus on a rejected hypothesis.

## Outcomes and fixed decision

Primary: positive `IA_DWELL_TIME` (total reading time).

Secondary, descriptive only:

- positive `IA_FIRST_RUN_DWELL_TIME` (gaze duration);
- positive `IA_FIRST_FIXATION_DURATION`.

M1 confirms only if, on the primary outcome:

1. the M1-M0 paired 95% bootstrap lower bound is above zero across participants;
2. the paired 95% bootstrap lower bound is above zero across complete articles;
3. at least four of five outer-fold participant-macro differences are positive;
4. the shuffled-target sentinel does not have a positive 95% lower bound across
   either participants or articles.

All four clauses are conjunctive. Secondary outcomes cannot rescue a failed
primary gate. An unfavorable result is reported unchanged and does not trigger
OneStop-specific tuning.

## Leakage, compute, and stop rules

- All scalers and Ridge coefficients fit on training articles only.
- No gaze coordinate or reading-time field is a predictor.
- Label-free feature extraction is cached before outcome evaluation.
- No LM fine-tuning, alternate model size, prompt, feature, alpha, fold, filter,
  or threshold search.
- Explicit CPU execution only. v1 must stop if asked to use CUDA.
- Raw outcomes and cross-fitted predictions remain ignored local artifacts;
  compact summaries, checksums, decisions, and process records are tracked.

The complete executable contract is
[`protocols/2026-08-05-onestop-confirmation-v1.json`](protocols/2026-08-05-onestop-confirmation-v1.json).

## Primary sources consulted before freeze

- OneStop project and download documentation: https://lacclab.github.io/OneStop-Eye-Movements/
- OneStop variables: https://lacclab.github.io/OneStop-Eye-Movements/variables
- OneStop paper/data availability: https://doi.org/10.1038/s41597-025-06272-2
- Official downloader: https://github.com/lacclab/OneStop-Eye-Movements/blob/main/download_data_files.py
- CELER access conditions: https://github.com/berzak/celer
- MECO corpus description: https://doi.org/10.3758/s13428-021-01772-6
