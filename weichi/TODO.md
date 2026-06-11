# Research TODO — Cognitive Load Pipeline Validation

## 現況摘要

Pipeline 已整合以下特徵（符合文獻建議）：
- Surprisal（GPT-2 contextual predictability）
- Zipf frequency（word frequency）
- Age-of-Acquisition / AoA（Kuperman 2012）
- Syntactic dependency load（spaCy parse tree）

**下一步核心問題：如何在不手動標記 GT 的情況下驗證系統有效？**
答案：用眼動數據當作 ground truth（GD / TRT），跑 mixed-effects regression，
看 `load_score` 在控制 frequency、length 等 baseline 後是否仍能顯著預測閱讀時間。

---

## Phase 1：Ground Truth 定義（不需要人工標記）

- [x] **不用 binary hard label**：改用 pipeline 輸出的 `load_score` 連續值作為預測變數
  - 文獻建議：以文本內的上尾分布（70th percentile）定義「難詞」，而非固定門檻
  - 目前 pipeline 已做 70th percentile → 維持現狀即可

- [x] 確認每個詞的 `load_score` 是否合理反映心理語言學難度
  - **v8 結果**：Spearman ρ = 0.42（TRT）/ 0.38（GD），p < 0.001，n = 1044 詞 ✅
  - **v9 結果（三項優化後）**：ρ = 0.434（TRT）/ 0.367（GD），OLS & LMM 均顯著 ✅✅

---

## Phase 2：眼動資料驗證（主要 evaluation）

### 目標
驗證 `load_score` 能否在排除 baseline 變數後，獨立預測受試者在該詞上的閱讀時間。

### 眼動指標（優先順序）

| 指標 | 英文全稱 | 說明 | 優先級 |
|------|---------|------|--------|
| GD   | Gaze Duration | 初次閱讀在該詞的總注視時間，反映詞彙辨識 | ★★★ 最重要 |
| TRT  | Total Reading Time | 含回視的所有注視，反映整合困難 | ★★★ |
| FFD  | First Fixation Duration | 第一次注視持續時間，較噪 | ★★ |
| RR   | Regression Rate | 回視機率，與語法整合負荷有關 | ★★ |

### 已完成（validate_geco.py，150 句）

#### v8 基線（GPT-2 + Ridge，1044 content words）
| 分析 | 結果 | 說明 |
|------|------|------|
| Spearman ρ (TRT) | 0.420, p < 0.001 | 顯著正相關 |
| Spearman ρ (GD) | 0.375, p < 0.001 | 同上 |
| OLS β(load_score) | n.s. | freq 共線性導致 |
| LMM β(load_score) | n.s. | 同上 |

#### ✅ v9（三項優化）— 開發集驗證（150 句，1039 content words）
根據 Deep Research 文獻：(1) POS-gate dep_load, (2) Rényi entropy, (3) XGBoost 後端

| 分析 | 結果 | 變化 |
|------|------|------|
| Spearman ρ (TRT) | 0.434 ****** | +0.014 ↑ |
| OLS β(load) = 0.194 | p < 0.001 ****** | 🔥 從 n.s. 變顯著 |
| LMM β(load_z) = 0.045 | p < 0.001 ****** | 🔥 從 n.s. 變顯著 |
| LMM ΔAIC | +30.5 | 舊：-0.6 |

#### ✅ 論文等級完整驗證 v1（舊，`full_validation.py`，600 句訓練）
- 訓練：600 句，測試：1000 句（sentences 701-1700），n = 4,571 content words
- ρ(TRT)=0.393, ρ(GD)=0.362, R²=0.107, OLS β=0.766 ***, ΔAIC=+127.6
- ⚠️ word_length 特徵當時全為 0（欄位對應 bug）

#### ✅ 論文等級完整驗證 v3（最新，+ LMM + Bootstrap 95% CI）
- XGBoost 訓練：2000 句（n = 9,793 content words）
- **Held-out 測試：1000 句（sentences 2101-3100），n = 4,882 content words（完全 unseen）**
- 新增 OLS 控制：prev_surprisal + prev_word_length（spillover effect）
- word_length 特徵 bug 已修正（WORD_LENGTH 正確對應）

| 分析 | 結果 | 備註 |
|------|------|------|
| Spearman ρ (TRT) | **0.437 *** **| 95% CI [0.412, 0.458] |
| Spearman ρ (GD) | **0.386 *** **| 95% CI [0.359, 0.409] |
| Held-out R² | **0.188** | — |
| OLS β(load) = 0.639 | **p < 0.001 *** **| ΔAIC = +104.6 |
| **LMM β(load_z) = 0.049** | **p < 0.001 *** **| per-reader 隨機截距 |
| **LRT χ²(1) = 184.61** | **p < 0.001 *** **| 最嚴格統計 ✅ |
| LMM ΔAIC | **+182.6** | — |

> **Paper-ready quote (完整版)**：
> "The pipeline predicted mean TRT with Spearman ρ = 0.437 (95% CI [0.412, 0.458])
> and GD ρ = 0.386 (95% CI [0.359, 0.409]) on 4,883 content words from 1,000
> held-out GECO sentences. After controlling for word frequency, length, sentence
> position, and preceding-word spillover, the load score independently predicted TRT
> (OLS β = 0.639, p < .001, ΔAIC = +104.6; LMM β = 0.049, LRT χ²(1) = 184.61,
> p < .001, ΔAIC = +182.6, n = 49,154 reader × word observations)."

#### ✅ 全語料跨章節穩定性驗證（`validate_remaining_geco.py`）
- 測試：sentences 2101–5284（3,183 句，16,318 content words，涵蓋小說後半部）
- 訓練範圍（sentences 1–2100）之外的所有資料

| 指標 | 2101–3100（1000句）| **2101–5284（3183句）** |
|------|--------------------|------------------------|
| Spearman ρ (TRT) | 0.437 *** | **0.440 *** **|
| Spearman ρ (GD) | 0.388 *** | **0.400 *** **|
| Held-out R² | 0.189 | **0.203** |
| OLS β(load) | 0.662 *** | **0.707 *** **|
| OLS ΔAIC | +104.1 | **+342.8** |

> **結論**：ρ 在 16,318 words 上仍為 0.440，pipeline 跨章節完全穩定。

#### ✅ 外部語料 Zero-Shot 驗證（`validate_provo.py`，PROVO）
- 語料：PROVO（Luke & Christianson 2018），55 passages，84 L1 讀者，混合文類（新聞/Wikipedia/敘事）
- **完全 zero-shot**：xgb_model.json 在 GECO 訓練，從未見過任何 PROVO 資料
- n = 1,592 content words

| 指標 | GECO held-out | **PROVO（zero-shot）** |
|------|--------------|----------------------|
| 讀者數 | 14 L1 | **84 L1** |
| ρ (TRT) | 0.437 *** | **0.619 *** **|
| ρ (GD) | 0.388 *** | **0.611 *** **|
| OLS β(load) | 0.662 *** | 0.652 *** |
| OLS ΔAIC | +104.1 | +63.2 |

> **結論**：Zero-shot 跨語料 ρ = 0.619，超越 GECO 結果（0.437），接近人際上限（0.50–0.60）。
> PROVO 的更高 ρ 反映更多讀者（84 vs 14）帶來的更穩定 mean TRT，與更廣的難度分布。
>
> **Paper-ready quote**：
> "Zero-shot transfer to PROVO (Luke & Christianson, 2018; 55 passages, 84 L1 participants,
> mixed genres) yielded Spearman ρ = 0.619 (GD: ρ = 0.611, both p < .001) on 1,592 content
> words, confirming cross-corpus generalization (OLS β = 0.652, p < .001, ΔAIC = +63.2).
> The pipeline was trained exclusively on GECO and had no exposure to PROVO data."

> ⚠️ OLS/LMM 不顯著是正常的：`load_score` 本身含頻率成分，和控制變數 `zipf_score` 共線。
> 需要做**成分分解回歸**（surprisal、AoA、dep_load 分開測）才能看到各成分的獨立貢獻。

### 統計模型設定（依文獻最佳實務）

#### 依變數
- 使用 **log(GD)** 或 **log(TRT)**（必須 log transform，原始值為右偏分布）

#### 預測變數
- 主要：pipeline `load_score`（連續值 0–1）

#### Baseline 控制變數（必須加，否則會高估效果）

| 控制變數 | 原因 |
|---------|------|
| word length（字元數） | surprisal 和 length 有相關 |
| Zipf frequency | 最強的單詞難度 baseline |
| sentence position | 句尾有 wrap-up effect |
| word position in sentence | 位置越後越受前文影響 |
| previous-word spillover | 前一詞的難度會「溢出」到當前詞的注視 |
| punctuation / sentence-final flag | 句末注視時間特別長 |

#### 隨機效應結構
- 最低要求：random intercepts for **subjects** + **items**
- 建議：如果設計允許，加入 key predictor 的 random slopes（within-subject）
- 若模型不收斂：先用最大結構，再逐步簡化（不要一開始就只用 intercepts）

#### 報告指標
- [ ] Fixed-effect coefficient（β）+ SE + p-value
- [x] **Nested model comparison**：ΔAIC = −1.9（freq 共線導致 n.s.，符合預期）
- [x] **Incremental R²**：ΔR² = 0.0001（同理，需成分分解才能看到各成分貢獻）
- [x] **Fixed-effect coefficient（β）+ SE + p-value**：見 full_validation_report.md
- [x] **成分分解回歸** ✅ 已完成（`validate_components.py`，見 `component_report.md`）：
  - Surprisal：ρ = 0.424 ***，R²(solo) = 0.191，joint β = 0.011 ***
  - AoA score：ρ = 0.276 ***，R²(solo) = 0.102，joint β = 0.116 **（頻率之外的獨立貢獻）
  - Dependency load：ρ = −0.021 n.s.（GECO fiction 句法單純，dep_load 效果不顯著）
  - Zipf freq：ρ = −0.429 ***，R²(solo) = 0.227，joint β = −0.035 ***
  - Word length：ρ = 0.456 ***（最強邊際預測子），joint β = 0.031 ***
  - **Joint model R² = 0.308，AIC = −174.9（優於 composite R² = 0.289，ΔAIC = +25.2）**
  - **結論：surprisal 和 AoA 有獨立貢獻，dep_load 在 fiction 語料不顯著，joint model 解釋力優於 composite score**
- [x] **Cross-validated prediction gain** ✅：Bootstrap 95% CI 已完成（見下方）

> ⚠️ 文獻不建議用單一 β 係數絕對值判斷效果是否有意義，應看 **模型 fit 的改善程度**（AIC/BIC 下降、incremental R² 提升）。

### 小樣本處理策略（~20–40 受試者）

- [x] **Bootstrap resampling** ✅（`robustness_analysis.py`）
  - ρ(TRT) = 0.437，95% CI [0.413, 0.459]
  - ρ(GD)  = 0.386，95% CI [0.360, 0.410]
  - 輸出：`bootstrap_ci_plot.png`
- [x] **Leave-one-subject-out (LOSO)** ✅（`robustness_analysis.py`）
  - **14/14 讀者全部 p < .001**
  - Mean ρ = 0.215 ± 0.044  [0.135, 0.291]
  - 輸出：`loso_plot.png`，報告：`robustness_report.md`
- [ ] **Leave-one-item-out**：關心預測能否推廣到新詞彙（可選）
- 注意：surprisal 效果對 **item 數量** 比 subject 數量更敏感；item 要夠多

### GECO 語料使用流程

- [x] **Tokenization 對齊**：string matching 完成，1998 詞中 1044 content words 有效對齊
- [x] **合併 prediction table**：`validate_geco.py` 已實作完整 join 流程
- [x] **Preprocessing（標準步驟）**：
  - 移除極端注視時間（通常 < 80ms 或 > 1200ms 視為 outlier）
  - 排除 track-loss trial
  - 處理 skipped word（未被注視的詞）
  - 標點符號的 token 對齊要特別確認
- [ ] First-pass（GD）和 later regression（TRT）分開計算，分別 run 模型

---

## Phase 3：下游任務驗證（Phase 2 有顯著結果後才做）

- [ ] **閱讀理解預測**：pipeline 難詞分布能否預測受試者的理解測驗分數
- [ ] **Complex Word Identification (CWI)**：
  - 用 SemEval-2016 Task 11 資料集
  - 看 pipeline score 的 F1 能否超越 frequency-only baseline
- [ ] **L2 學習者難度**：比較 L1 和 L2 受試者在 pipeline 高負荷詞上的眼動差異是否顯著（CELER 語料有 L2 資料）

---

## 論文方法段落撰寫 checklist（最終投稿前）

- [ ] 說明 `load_score` 的特徵組成與計算方式
- [ ] 說明為何使用 log(TRT) 而非原始值
- [ ] 說明 random effects 的完整結構與決策理由
- [ ] 報告 baseline model vs full model 的 LRT 或 AIC 比較表
- [ ] 報告 LOSO 或 bootstrap 的穩健性結果

---

## 必讀論文清單（已更新）

| 論文 | 重點 | 優先 |
|------|------|------|
| Demberg & Keller (2008) | Surprisal 在眼動語料的驗證，mixed-effects 標準方法論 | ★★★ |
| Oh & Schuler (2022) EMNLP | GPT-2 predictors → reading time，最直接的方法論模板 | ★★★ |
| Cop et al. (2017) — GECO | GECO 語料結構與使用方式，必引 | ★★★ |
| Smith & Levy — surprisal 理論 | LM surprisal 和 reading time 線性關係的理論基礎 | ★★★ |
| Baayen et al. (2012) Gaze4NLP workshop | Random effects 結構選擇的建議 | ★★ |
| Alves et al. (2025) Gaze4NLP | 最新 LM surprisal → FFD/GD/TRT benchmark | ★★ |
| Gruteke Klein et al. (2025) | Surprisal-based readability evaluation | ★★ |
| Kajiwara & Komachi (2018) | CWI 任務，frequency-based baseline 對照 | ★ |

---

## 可用語料

| 語料 | 特點 | 優先 |
|------|------|------|
| **GECO** | Christie 小說，詞級 TRT/GD/FFD，我們已有部分資料 | ★★★ |
| CELER | 365 受試者，含 L1/L2，適合小樣本外部驗證 | ★★ |
| Dundee Corpus | 新聞文本，經典基準 | ★★ |

---

## Phase B：語言模型比較（Surprisal 來源對預測力的影響）

**目標：比較 GPT-2（現有）vs 更大/不同架構的模型，看 surprisal 品質對 ρ 的影響。**

### 計畫比較模型

| 模型 | 大小 | 架構 | 說明 |
|------|------|------|------|
### ✅ Phase B 已完成（`compare_models.py`，100 句，SurprisalCalc 方法，sum of BPE token NLLs）

| 模型 | 大小 | 架構 | Spearman ρ (TRT) | ΔR² |
|------|------|------|------------------|-----|
| `gpt2` | 117M | GPT-2 | **0.398 ****** | 0.040 |
| `gpt2-large` | 774M | GPT-2 | 0.355 ****** | 0.027 |
| `gpt2-xl` | 1.5B | GPT-2 | 0.345 ****** | 0.026 |
| `EleutherAI/gpt-neo-1.3B` | 1.3B | GPT-Neo | 0.347 ****** | 0.033 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | **LLaMA** | **0.362 ****** | 0.033 |

> ⚠️ GPT-3 無法本機推論（OpenAI API only）。TinyLlama 代表 LLaMA 架構驗證。

**結論：**
- **GPT-2（117M）最佳**：ρ = 0.398，最小模型反而最強
- **TinyLlama（LLaMA 架構）排第二**：ρ = 0.362，優於同級 GPT-2-XL（1.5B）
- **模型越大 ρ 越低**（GPT-2 family）：大模型 surprisal 分布太平滑，難詞/易詞分辨力下降
- **LLaMA 架構優於 GPT-2 大模型**：TinyLlama 1.1B > GPT-2-XL 1.5B，架構比參數量更重要
- 與 base pipeline 結果（ρ = 0.424，150 句，不同計算方式）一致，差異來自方法（sum vs context）

**建議：維持 GPT-2 作為 surprisal 來源**（最佳 ρ，最快，最省記憶體）

### 腳本
- [x] `compare_models.py` ✅
- 輸出：`model_comparison.png`、`model_comparison_report.md`

---

## Phase C：Deep Research Prompt

已生成（見下方），可貼入 Perplexity Deep Research 或 ChatGPT Advanced Research 使用。

### 找相關 paper 的 prompt

```
I am building a cognitive load pipeline for eye-tracking reading research.
The pipeline combines GPT-2 surprisal, Zipf word frequency, Age-of-Acquisition (Kuperman 2012),
and syntactic dependency load (spaCy) to assign per-word cognitive load scores.
We validated it on GECO (Cop et al. 2017), obtaining Spearman ρ = 0.42 (TRT) and ρ = 0.38 (GD),
both p < .001, n = 1,044 content words.

Please find academic papers (2015–2025) on the following topics:

1. **LM surprisal as a predictor of eye-tracking reading times**
   - Papers that use GPT-2, GPT-3, LLaMA, or similar causal LMs to predict GD/TRT/FFD in corpora
     like GECO, Dundee, Natural Reading, or CELER.
   - Key comparison: does model size (117M → 7B) monotonically improve ρ with reading times?
   - Especially relevant: Oh & Schuler (2022) EMNLP; Pimentel et al. (2023); Goodkind & Bicknell (2018)

2. **Multi-feature cognitive load / readability models for eye-tracking**
   - Papers combining surprisal + frequency + AoA + syntactic complexity as joint predictors of fixation time.
   - Any evidence for AoA's *independent* contribution after controlling for frequency.
   - Relevant: Kuperman et al. (2012) on AoA; Brysbaert & New (2009) on word frequency.

3. **Syntactic complexity and reading time (dependency distance/depth)**
   - Papers testing dependency distance or integration cost (Gibson's DLT) as predictor of GD/TRT.
   - Especially on naturalistic/fiction corpora (where effects tend to be weaker than lab stimuli).

4. **SOTA benchmarks for surprisal → reading-time prediction**
   - What is the current best Spearman ρ / R² for predicting word-level TRT from any LM?
   - Any papers comparing LM architectures (GPT-2 vs GPT-Neo vs LLaMA) head-to-head on the same corpus.

5. **L2 reader cognitive load and eye-tracking**
   - Papers studying how cognitive load / surprisal correlates with eye movements in L2 (non-native) readers.
   - Especially relevant if they use CELER or similar corpora with L1/L2 participants.

For each paper found, please provide:
- Full citation (APA format)
- Main method (what model/features used, what corpus)
- Key result (ρ or R² value if available)
- Why it's relevant to our pipeline
```

---
