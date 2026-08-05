# Text Modeling Validity Review and Experiment Protocol

- Date: 2026-08-05
- Branch: `research/text-modeling-validity`
- Status: research review and preregistered experiment plan; no new benchmark result yet
- Compute used for this review: CPU/file inspection only; no model inference and no GPU workload

## 1. Decision

The next high-value objective is not to collect more samples immediately or replace GPT-2 with a larger model. It is to establish which text-derived signals add reproducible predictive value for human reading behavior after lexical, document, participant, and gaze effects are controlled.

The primary claim to test is:

> A frozen text model provides incremental, out-of-sample information about human reading time beyond a lexical/oculomotor baseline, and the effect transfers across passages and corpora.

This protocol explicitly excludes question-answer accuracy as a model-selection target. No QA dataset may be used to tune features, weights, thresholds, or model choice.

## 2. Current-system audit

### 2.1 What is already useful

- English uses causal GPT-2 next-token probabilities. Summing token negative log-probabilities into a word score is a defensible incremental-surprisal construction.
- The archived GECO evaluation reports a 2,000-sentence training range, 100 validation sentences, and 1,000 held-out sentences. Its reported held-out results are useful prior evidence, not final confirmation.
- The archived model comparison found GPT-2 small stronger than several larger causal models on the examined GECO subset. This agrees with recent inverse-scaling results in the literature and gives us a concrete hypothesis to replicate.
- The archived GECO-to-Provo report is promising cross-corpus evidence. Because those results have already been inspected, Provo must now be treated as a replication set rather than a never-seen final test set.

### 2.2 Validity risks to correct before the next model claim

1. **Two different quantities share the name `surprisal`.**
   - English GPT-2 uses left context and estimates incremental surprisal.
   - Chinese BERT masks a word while retaining both left and right context. This is masked pseudo-surprisal/cloze difficulty, not the online surprisal available to a reader at that word.
   - These values must have different names and may not be directly compared on one scale.

2. **Long-context handling changes the model at chunk boundaries.**
   - `_compute_gpt_chunked` evaluates independent chunks, so context resets at every boundary.
   - The chunked return also drops `renyi_entropies`.
   - A sliding prefix/overlap policy and boundary tests are required before document-level claims.

3. **Current load scores are relative to the current text.**
   - Surprisal, entropy, Rényi entropy, XGBoost predictions, and Ridge predictions are min-max or maximum normalized within the processed unit.
   - This helps visualization but prevents calibrated comparison across articles and lets future words in the same unit determine an earlier word's displayed score.
   - Scientific evaluation must retain raw features and fit any scaler on training folds only.

4. **Model provenance is not machine-verifiable.**
   - `core/cognition/xgb_model.json` is byte-identical to the archived artifact, but it does not carry a dataset manifest, split hash, feature-code hash, or training command.
   - Archived reports describe the larger 2,000/100/1,000 GECO experiment, while the standalone training script defaults to 120/30 sentences.
   - `ridge_model.json` records `train_n=100`, `val_n=20`, and `n_samples=663`, which mix sentence and word counts without naming the units.
   - Every future artifact needs a sidecar manifest with corpus version, group splits, feature schema, code commit, tokenizer/model revision, seed, and metrics.

5. **One fusion benchmark is circular and cannot support a text-model claim.**
   - `scripts/experiment_fusion.py` takes GECO total reading time as the target, constructs simulated gaze dwell/fixation directly from that same target, fuses it with text features, and correlates the result back with total reading time.
   - High correlation is expected even if the text feature is random. This experiment can remain a UI/simulation demo, but it must be excluded from empirical model evidence.

6. **The neural cross-attention fusion path is untrained.**
   - `scripts/fusion_module.py::fuse_cross_attention` initializes random weights with a fixed seed and immediately performs inference.
   - Reproducible randomness is not a learned model. The method should be marked experimental/invalid until trained and tested on group-held-out data.

7. **Attention is a candidate predictor, not automatically an explanation.**
   - Attention-derived entropy or distance may predict reading time, but raw attention weights do not establish causal importance.
   - Interpretability language must be limited to validated associations or intervention-based tests.

### 2.3 Data availability

- The expected ignored directories `archive/weichi/GECO_data/` and `archive/weichi/PROVO_data/` are not currently present.
- A large GECO L1 worksheet exists under `archive/data/geco/`, but the archived benchmark scripts expect normalized CSV exports and Provo is absent.
- Therefore the first reproducible run is blocked on restoring or regenerating the public corpus inputs. This is a data-packaging dependency, not a reason to collect new LexiGaze webcam samples.

## 3. Literature synthesis and resulting choices

1. **Larger/lower-perplexity language models are not necessarily more human-like.** Oh and Schuler found that larger GPT-Neo and OPT variants often fit reading time worse, and a controlled training study placed the best fit near two billion training tokens. We should compare compact frozen models before considering a large model or fine-tuning.

2. **Surprisal is still a strong core predictor.** Multilingual work supports incremental surprisal and contextual entropy across languages, while large-scale out-of-sample analyses support an approximately linear surprisal effect on reading time.

3. **Different eye measures may require different text representations.** Recent layer-probing work reports that early-layer representations can outperform scalar surprisal for first fixation and gaze duration, while surprisal remains stronger for total reading time. We must not collapse all eye measures into one target.

4. **Tokenization is a real experimental factor.** BPE aggregation is acceptable in aggregate in some studies, but vocabulary choice changes word-level surprisal and downstream reading-time fits, especially for split words. Tokenizer revision and split status must be logged, and results should be stratified by single-token versus multi-token words.

5. **Masked and causal scores answer different questions.** A bidirectional masked score uses right context and is useful as a cloze/semantic-coherence diagnostic; a causal score models information available during incremental reading. Both may be tested, but they need separate hypotheses and labels.

## 4. Preregistered hypotheses

The hypotheses below must be frozen before restoring the final confirmation corpus.

- **H1 — Incremental surprisal:** causal word surprisal improves passage-held-out prediction of reading time beyond lexical controls.
- **H2 — Time-course specificity:** early-layer representations add more to first-fixation duration and gaze duration; scalar surprisal adds more to total reading time.
- **H3 — Uncertainty:** contextual Shannon/Rényi entropy or entropy reduction adds held-out value beyond current- and previous-word surprisal.
- **H4 — Architecture semantics:** causal surprisal is more appropriate for online effects than masked pseudo-surprisal; masked scores may retain value for later integration effects.
- **H5 — Compact-model advantage:** GPT-2 small matches or outperforms larger frozen causal models after identical token alignment and evaluation.
- **H6 — Cross-language non-equivalence:** Chinese and English require language-specific calibration; raw scores and thresholds will not be pooled across languages.

## 5. Benchmark design

### 5.1 Outcomes

Keep the measures separate:

- Early pass: first-fixation duration, gaze duration, skip probability.
- Late pass: total reading time, rereading/regression indicators.
- LexiGaze deployment outcome: calibrated word difficulty and article-level ranking, evaluated only after the psycholinguistic benchmark passes.

### 5.2 Candidate models

All language models remain frozen.

| ID | Feature set | Purpose |
|---|---|---|
| M0 | word length/characters, log frequency, position, punctuation/POS, previous-word controls | lexical baseline |
| M1 | M0 + causal surprisal at word `t`, `t-1`, and optionally `t-2` | primary incremental hypothesis |
| M2 | M1 + Shannon/Rényi entropy or entropy reduction | uncertainty ablation |
| M3 | M1 + regularized probe on cached early-layer states | early-pass hypothesis |
| M4 | M1 + explicitly named masked pseudo-surprisal | diagnostic comparison, not causal surprisal |
| M5 | M1 + validated syntactic integration features | optional memory/integration hypothesis |

XGBoost may be retained as an engineering comparator, but the primary scientific analysis should use regularized linear/generalized mixed models so that incremental contributions and uncertainty are auditable.

### 5.3 Splits

- Outer evaluation holds out complete passages/documents (`Text_ID` or equivalent); never random words.
- A secondary analysis holds out participants or uses participant random effects.
- Hyperparameters, scalers, feature selection, and early stopping are fit only inside training folds.
- Repeated occurrences of the same text cannot cross train/test boundaries.
- GECO is development evidence; Provo is replication evidence because its historical result is known.
- A new untouched corpus is required for the final confirmatory claim. Corpus choice and inclusion rules must be recorded before its labels are inspected.
- LexiGaze's collected webcam data is reserved for sensor/fusion validation and may not select the text model unless it contains enough independent passages and participants for grouped evaluation.

### 5.4 Metrics

Primary comparison:

- held-out log likelihood or deviance improvement over M0;
- passage-level paired improvement with block bootstrap confidence intervals.

Secondary metrics:

- held-out MAE and out-of-sample `R²` for continuous durations;
- held-out log loss/AUROC for skip or regression outcomes;
- calibration slope/intercept across passages;
- effect stability by corpus, participant, word frequency, POS, and tokenizer split status.

Raw correlation alone is descriptive and cannot select the winning model.

### 5.5 Leakage sentinels

The benchmark must fail loudly when any sentinel fails:

1. Shuffling the target within held-out groups must remove incremental performance.
2. Replacing the text feature with random noise must not reproduce the claimed gain.
3. Predictors must be computable without reading test labels.
4. A feature may not contain a gaze measure derived from the evaluation target.
5. Any target-derived normalization must be fitted on training data only.
6. Split manifests are immutable and hashed.

## 6. GPU budget

The benchmark is intentionally feature-extraction-heavy rather than training-heavy.

1. Phase A is CPU-only: dataset validation, split manifests, lexical baseline, leakage tests, and cached GPT-2-small surprisal where practical.
2. Phase B may use one bounded GPU extraction run for hidden states only after Phase A passes:
   - no language-model fine-tuning;
   - maximum 8 GiB VRAM allocation target;
   - maximum 30 minutes per registered model/corpus extraction job;
   - telemetry recorded at start/end and every minute;
   - stop on out-of-memory, non-finite features, or sustained laptop thermal throttling.
3. Ridge/elastic-net, mixed models, bootstrap, and all ablations run on CPU from cached features.
4. Each cache records model and tokenizer revisions, input hash, context policy, dtype, device, runtime, and peak memory.

No overnight GPU run is justified until a cached CPU baseline and leakage-free split pass their gates.

## 7. Acceptance gates

- **Gate 0 — semantics:** causal surprisal and masked pseudo-surprisal are separately named and tested.
- **Gate 1 — reproducibility:** corpus/split/model manifests reproduce byte-identical feature rows for a fixed seed and revision.
- **Gate 2 — leakage:** all leakage sentinels pass; the circular fusion demo is excluded from evidence tables.
- **Gate 3 — development:** the candidate improves grouped held-out performance over M0 with a passage-block confidence interval that does not favor the baseline.
- **Gate 4 — replication:** the effect has the same direction on Provo without retuning.
- **Gate 5 — confirmation:** a preregistered untouched corpus confirms the primary effect.
- **Gate 6 — product:** only then may the calibrated score replace or augment the current document-relative heuristic.

## 8. Implementation sequence

1. Add a metric contract: `causal_surprisal`, `masked_pseudo_surprisal`, entropy definition, token aggregation, context policy, and units.
2. Fix GPT chunk boundaries and preserve Rényi outputs; add boundary-equivalence tests.
3. Add artifact and dataset manifests plus group-split hashes.
4. Build a CPU-first benchmark with M0/M1 and leakage sentinels.
5. Restore/regenerate GECO and Provo inputs, then reproduce archived results without changing the frozen protocol.
6. Add M2 entropy ablations.
7. Only if justified, run bounded cached early-layer extraction for M3.
8. Select an untouched confirmation corpus and freeze its analysis before label inspection.
9. Integrate the winning calibrated feature into gaze fusion and evaluate text-only, gaze-only, and combined models on the same grouped folds.

## 9. Immediate deliverables for the implementation branch

- `docs/experiments/text-modeling-hypotheses.yaml`: frozen hypotheses, outcomes, splits, and gates.
- `configs/text_modeling/*.yaml`: models, revisions, context, dtype, and compute limits.
- `scripts/text_modeling/build_manifest.py`: corpus and split manifests.
- `scripts/text_modeling/extract_features.py`: deterministic cached feature extraction.
- `scripts/text_modeling/benchmark.py`: grouped, nested evaluation and ablations.
- `tests/text_modeling/`: chunk-boundary, token-alignment, provenance, and leakage-sentinel tests.
- `artifacts/text_modeling/<run-id>/`: metrics, fold predictions, environment, telemetry, and report.

## 10. Primary references

- Oh, B.-D., & Schuler, W. (2023). [Why Does Surprisal From Larger Transformer-Based Language Models Provide a Poorer Fit to Human Reading Times?](https://aclanthology.org/2023.tacl-1.20/)
- Oh, B.-D., & Schuler, W. (2023). [Transformer-Based Language Model Surprisal Predicts Human Reading Times Best with About Two Billion Training Tokens](https://aclanthology.org/2023.findings-emnlp.128/)
- Wilcox, E. G., et al. (2023). [Testing the Predictions of Surprisal Theory in 11 Languages](https://aclanthology.org/2023.tacl-1.82/)
- Oh, B.-D., Schuler, W. (2022). [Entropy- and Distance-Based Predictors From GPT-2 Attention Patterns Predict Reading Times Over and Above GPT-2 Surprisal](https://aclanthology.org/2022.emnlp-main.632/)
- Nair, S., & Resnik, P. (2023). [Words, Subwords, and Morphemes: What Really Matters in the Surprisal-Reading Time Relationship?](https://aclanthology.org/2023.findings-emnlp.752/)
- Tsipidi, E., et al. (2026). [Probing for Reading Times](https://aclanthology.org/2026.acl-long.575/)
- Jain, S., & Wallace, B. C. (2019). [Attention is not Explanation](https://aclanthology.org/N19-1357/)
- Nguyen, K., & Arehalli, S. (2026). [Word predictability estimates from language models are not robust to tokenizer vocabulary](https://aclanthology.org/2026.conll-main.3/)
