請根據以下研究資料，製作一份視覺化的研究進展報告（用 React component 呈現，要好看、可以直接截圖）。

主題：**眼動認知負荷 Pipeline v8 → v9 改進報告**

---

## 一、背景說明（給設計參考）

這是一個用於閱讀研究的認知負荷預測 Pipeline，輸入是一段英文文章，輸出是每個詞的「認知負荷分數 (0–1)」。核心原理：讀者在難以預測的詞上會花更長的眼動注視時間（TRT/GD），所以可以用眼動資料反向驗證 pipeline 是否準確。

---

## 二、v8 vs v9 改進對照

### Pipeline 架構變化

| 元件 | v8（舊） | v9（新） | 改進理由 |
|------|---------|---------|---------|
| 語言模型 | GPT-2 surprisal | GPT-2 surprisal（不變） | Phase B 實驗確認 GPT-2 最佳 |
| 資訊熵 | Shannon entropy | **Rényi entropy（α=0.5）** | 預測詞彙前段注視更準確（Pimentel et al. 2023） |
| 詞彙特徵 | AoA + Zipf（不變） | AoA + Zipf（不變） | 驗證有獨立貢獻 |
| 句法複雜度 | dep_load（所有詞類） | **dep_load（POS-gate：僅 NOUN/VERB/PROPN）** | 非內容詞 dep_load 為負向預測子（Rathi 2021） |
| 後端模型 | Ridge Regression（線性） | **XGBoost（非線性）** | 解決 zipf 共線性問題（Salicchi et al. 2022） |

### 三個關鍵改進的原理（各用一段文字解釋）

**改進 1：Rényi entropy（α=0.5）**
Shannon entropy 衡量下一個詞的「不確定性」，但讀者其實在讀到詞的前半段就開始預測。Rényi entropy（α=0.5）對高概率詞給予更大權重，公式為 H₀.₅ = 2·log(Σ√P_i)，更能捕捉讀者在詞頭產生的「預期性認知負荷」。

**改進 2：POS-gate dep_load**
GECO 語料是偵探小說（Christie），句法相對簡單，句法依存負荷（dep_load）在全詞類套用時為負向預測子（ρ = −0.021 n.s.）。根據 Rathi 2021，句法整合負荷只在內容詞（名詞、動詞、專有名詞）上有理論意義。POS-gating 後非內容詞 dep_load 設為 0.0，解決了錯誤方向的干擾。

**改進 3：XGBoost 後端**
v8 的 Ridge Regression 將 load_score 線性合成後，因為包含 zipf 成分，導致在 OLS 迴歸裡與控制變數 zipf_score 完全共線，β 係數 n.s.。XGBoost 非線性地從 7 個原始特徵生成 load_score，打破了共線性，使 load_score 有了獨立的顯著貢獻。

---

## 三、量化結果對照

### 3.1 開發集對照（150 句，約 1,044 content words）

| 指標 | v8（Ridge） | v9（XGBoost） | 變化 |
|------|------------|--------------|------|
| Spearman ρ（TRT） | 0.420 *** | 0.434 *** | **+0.014 ↑** |
| Spearman ρ（GD） | 0.375 *** | 0.367 *** | −0.008（穩定） |
| OLS β(load_score) | n.s. | **p < 0.001 ****** | **🔥 突破顯著** |
| LMM β(load_z) | n.s. | **p < 0.001 ****** | **🔥 突破顯著** |
| LMM ΔAIC | −0.6（負值＝無改善） | **+30.5** | **大幅改善** |

### 3.2 論文等級完整驗證（1,000 句完全 held-out，4,571 content words）

| 指標 | 數值 | 意義 |
|------|------|------|
| Spearman ρ（TRT） | **0.393 *** **| SOTA 文獻範圍 0.35–0.45，前段 ✅ |
| Spearman ρ（GD） | **0.362 *** **| 穩定泛化 |
| Held-out R²（log TRT） | 0.107 | 泛化能力 |
| OLS β(load_score) | **0.766 *** **| 控制頻率/字長/位置後仍顯著 |
| OLS ΔAIC | **+127.6** | 極大模型改善（越正越好） |

> 訓練：600 句 → 完全 unseen 的 1,000 句測試，證明模型能泛化。

---

## 四、各成分獨立貢獻（Phase A 元件分解）

在控制其他變數前，各特徵對 TRT 的邊際相關：

| 特徵 | Spearman ρ（TRT） | 顯著性 | 說明 |
|------|-----------------|--------|------|
| Word length（字長） | **0.456** | *** | 最強單特徵 |
| Zipf frequency（詞頻） | −0.429 | *** | 越常見越短 |
| Surprisal（GPT-2） | 0.424 | *** | 語言模型貢獻 |
| AoA score（習得年齡） | 0.276 | *** | 獨立於詞頻之外 |
| Dependency load | −0.021 | n.s. | GECO fiction 句法單純 |

Joint model（5特徵）R² = 0.308，優於 composite score（R² = 0.289，ΔAIC +25.2）

---

## 五、語言模型比較（Phase B）

比較不同 LM 的 surprisal 對 TRT 預測力（100 句，公平比較）：

| 模型 | 參數量 | 架構 | ρ（TRT） |
|------|--------|------|---------|
| GPT-2 | 117M | Transformer | **0.398 *** **（最佳）|
| TinyLlama | 1.1B | LLaMA | 0.362 *** |
| GPT-Neo | 1.3B | GPT-Neo | 0.347 *** |
| GPT-2-XL | 1.5B | Transformer | 0.345 *** |
| GPT-2-Large | 774M | Transformer | 0.355 *** |

**結論：模型越大 surprisal 品質反而下降**（scaling paradox，Oh & Schuler 2023）。GPT-2（117M）保持最佳，且最快、最省記憶體。

---

## 六、指標解釋（給讀者的說明欄位）

| 指標 | 全名 | 怎麼解讀 |
|------|------|---------|
| **Spearman ρ** | Spearman rank correlation | Pipeline 預測的困難程度排名 vs 眼動注視時間排名的一致性。ρ=1 完美，ρ=0 無相關。文獻 SOTA ≈ 0.35–0.45 |
| **TRT** | Total Reading Time | 讀者在某個詞上的所有注視時間總和（含回視）。反映整體理解困難 |
| **GD** | Gaze Duration | 第一次讀到該詞時的注視時間。反映即時詞彙辨識難度 |
| **OLS β** | OLS regression coefficient | 在 OLS 模型中，控制詞頻/字長/句子位置後，load_score 每增加 1 個單位，log(TRT) 增加的量。顯著代表 load_score 有獨立貢獻 |
| **ΔAIC** | AIC difference（full − base） | 加入 load_score 後模型改善量。正值越大越好（>2 = significant improvement，>10 = strong） |
| **Held-out R²** | Out-of-sample R² | 在完全沒看過的測試資料上，模型能解釋多少 TRT 變異量。這才是真正的泛化能力 |

---

## 設計要求

請設計成一頁式的研究報告（Report Page），風格：
- 學術但現代，類似 Notion Research Page 或 Papers with Code 的卡片風格
- 主色調：深藍 + 白色背景，重點標示用琥珀色/綠色
- 每個「突破」用醒目的 badge 或高亮框標示（例如 "OLS n.s. → p<.001" 這個突破）
- 統計顯著性用星號顯示：*** = p<.001，** = p<.01，* = p<.05
- 結尾加一個 "Paper-ready summary" 段落，顯示論文可直接引用的英文句子

Paper-ready quote（請直接放到報告最後）：
"The cognitive load pipeline (GPT-2 surprisal, Rényi entropy α=0.5, AoA, POS-gated syntactic dependency load, XGBoost) predicted mean TRT with Spearman ρ = 0.393 (GD: ρ = 0.362, both p < .001) on 4,571 content words from 1,000 completely held-out GECO sentences. After controlling for word frequency, length, and sentence position, the pipeline load score independently predicted TRT (OLS β = 0.766, p < .001, ΔAIC = +127.6)."
