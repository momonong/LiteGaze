# Reader Assessment v2 Progress Log

## 2026-08-08 — branch `research/non-cn-text-backbone-benchmark`

### Text-backbone development screen

- 將 GPT-2 固定為歷史 baseline，而非直接以大型 chat LLM 取代；凍結非中國來源的 exact allowlist、immutable revisions、禁止 remote code 與 base-only 規則。
- 在任何新 backbone outcome 開啟前，固定 Provo development-only protocol、完整 passage holdout、shared lexical / tokenization / causal-surprisal feature sets、shortlist gate 與 1.5 GPU-hour 上限。
- GPT-2、Pythia-410M-deduped step1000 與 full checkpoint 均完成 2,743 items 的 label-free extraction；所有 outcome 欄位直到五個候選的技術檢查結束後才開啟。
- SmolLM2-360M 與 OLMo 2 1B 都在相同 display 位置產生 separator-only offset，依 v1 frozen alignment contract 原樣記為技術失敗，未查看 outcome、未放寬規則、未重跑。
- 三個成功 backbone 的 causal surprisal 相對 lexical + tokenizer controls 都有小型正增益且 5/5 folds 為正；GPT-2 仍是穩定基準。
- Pythia full 相對 GPT-2 的 primary M1 participant delta `+0.0008`、text delta `+0.0039`，但 text CI 跨 0 且只有 3/5 folds 較高；依預註冊 point-rule 成為唯一 shortlist，但不構成 replacement evidence。
- Pythia step1000 相對 GPT-2 為 participant `-0.0004`、text `-0.0024`，未入選；預先指定的 early-minus-full contrast 方向偏向 full，但 participant/text CI 均跨 0，不能宣稱 training-token effect。
- 成功 forward 合計 `31.04` 秒／`0.00862` GPU-hours，只使用 1.5-hour budget 的 0.57%；peak reserved 為 GPT-2 `0.371 GiB`、Pythia `0.924 GiB`，全程一次只載入一個模型，沒有 training/fine-tuning。
- Product model 維持 GPT-2 frozen；下一步只能先凍結 GECO L2 replication，再開啟 full Pythia 的 GECO outcome。OneStop 不再用於候選選模。

## 2026-08-08 — branch `research/reader-assessment-validity-v3`

### 問題重構

- 將「文字模型很差」拆成兩件事：目前 frozen causal surprisal 在三個 corpus 有小而一致的 population-level 增益；真正缺少的是 independent personalized outcome。
- 確認 6 passages / 18 items 與 expert-seeded 3PL 只能作工程 pilot，且 passage-linked items 的 local dependence 會讓一般 IRT precision 過度樂觀。
- 停止把 reading/gaze observables 當作 cognitive profile；working memory 等構念只能由獨立、授權、經驗證的作業支持。

### 研究與決策

- 依 testing Standards 與 Evidence-Centered Design，先固定 intended use、claim、evidence、task，再選模型。
- 依 PIAAC 將 reading components 擴為 sentence meaning、explicit、inference、lexical/cohesion 與 source evaluation，但先保留 task-specific scores。
- LexTALE/LEAP-Q 類工具只能分別作 lexical anchor 與 language-background covariates，不自動產生 CEFR。
- CEFR 需另行 specification、standard setting 與 empirical linking；v3 保持 `not_estimated`。
- 初次真實校準改用 balanced fixed forms；只有 empirical item/testlet parameters、information、fairness 與 confirmation 通過後才能重新啟用 CAT。
- Primary product target 改為 post-reading sampled-word `no_review / unsure / review_needed`，QA、gaze duration、text score、calibration coordinates 與 synthetic fusion labels 全部禁止作 training/selection target。
- 凍結 participant、passage family、item family、capture session 與 device class 五個 holdout axes；confirmation 只開一次。
- Cognitive add-on 改為 separate optional session 且禁止 composite；本階段完全 CPU-only。
- v3 凍結前已收集的資料保留作 workflow/quality exploratory evidence，但禁止進入 calibration、model/threshold selection、validation 或 confirmation。

### 實作

- 新增 `MEASUREMENT_DESIGN_V3.md` 與 machine-readable `reader_assessment_validity_v3.json`。
- 新增 CPU/network-free design audit，對 latent-claim abstention、獨立 outcome、anti-QA-overfit、固定 typography、holdout、ablation、gaze fallback、cognition separation 與 compute policy 做 fail-closed 檢查。
- 新增 mutation tests，確保 CEFR 提前晉級、未校準 adaptive routing、QA target、缺 passage holdout、同 session cognition 與 text fine-tuning 都會失敗。
- 將 v3 design test 納入 repository offline quality gate。
- 以 200 replicates、`n=300/600/900` 做 CPU-only matrix coverage simulation；18-passage 方案雖有較多重複 labels，卻只有 3 個 confirmation passage clusters，因此否決。
- 將 reading measurement calibration 與 personalized fusion 拆成不重疊 cohorts；前者規劃獨立 36-family measurement bank，後者採 48-family pool（24/12/12）以保留 12 個 confirmation content clusters。
- Coverage run 001 在 48-family 方案、`n=600` 時產生中位 1,440 個 joint-confirmation word labels，但第五百分位的每篇 joint cell 仍只有 6 人；這證明最終 n 必須用 cluster-aware power/utility simulation 決定，不能把 600 當成自動足夠。

### 邊界

- v3 是已凍結的研究設計，不取代 live v2 protocol，也不授權收案。
- 下一階段是選定合法 external anchors、分別建立並雙人審查 36-family measurement bank 與 48-family fusion pool、做 effect/precision-based sample-size simulation，再更新 ethics/consent 文件。

### 驗證

- v3 design contract 的 19 個 safeguards 全數通過，`collection_ready=false` 如預期保留。
- 11 個 v3 focused tests 通過，涵蓋 protocol mutation、legacy-data isolation 與 coverage determinism。
- Repository offline CPU quality gate：151 tests passed，0 failures/errors/skips；Torch 未 import、network/process probes 均被阻擋、artifact 無變動。
- Gate 前後 GPU telemetry 完全相同：0% utilization、729 MiB / 24,463 MiB；本輪研究、simulation 與驗證沒有使用 GPU compute。
- Python compileall 與 `git diff --check` 通過；專案 `.venv` 未安裝 Ruff，因此沒有把 Ruff 誤報為已執行。
- v3 protocol SHA-256：`52B40498D2ECD2C0D35A45328E6F5AB0DA78C3C7C3F2CC34D99B656D9DB71D93`。
- Coverage JSON SHA-256：`37F4251831CA77C3710AE7245E20A6F4EC2F21ABED47C01B669636931A1695C3`；Markdown SHA-256：`2D15B6776831AEA69B492CE4BDB42A33671CE257D8FBDB0263DA76CF8EF411A7`。

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
