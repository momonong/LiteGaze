請根據以下研究資料，製作一份視覺化的研究進展報告（用 React component 呈現，要好看、可以直接截圖）。

主題：**眼動認知負荷 Pipeline — 完整驗證報告（v9）**

---

## 一、背景說明（給設計參考）

這是一個用於閱讀研究的認知負荷預測 Pipeline，輸入是一段英文文章，輸出是每個詞的「認知負荷分數 (0–1)」。核心原理：讀者在難以預測的詞上會花更長的眼動注視時間（TRT/GD），所以可以用眼動資料反向驗證 pipeline 是否準確。

---

## 二、Pipeline v9 架構

| 元件 | 實作 | 說明 |
|------|------|------|
| 語言模型 | GPT-2 surprisal | Phase B 實驗確認 GPT-2（117M）優於更大模型 |
| 資訊熵 | Rényi entropy（α=0.5） | 比 Shannon entropy 更準確捕捉詞頭預測負荷 |
| 詞彙特徵 | AoA + Zipf | AoA 在控制詞頻後仍有獨立貢獻（β *** ） |
| 句法複雜度 | POS-gate dep_load（NOUN/VERB/PROPN only） | 非內容詞 dep_load 設為 0.0 |
| 後端模型 | XGBoost | 非線性合成 7 個特徵，解決 zipf 共線性 |

---

## 三、量化驗證結果（全部 held-out / zero-shot，無 data leakage）

### 3.1 主驗證：GECO Held-out（1,000 句，完全 unseen）

訓練：GECO 前 2,000 句 → 測試：sentences 2,101–3,100（4,883 content words，14 readers）

| 指標 | 數值 | 95% CI | 說明 |
|------|------|--------|------|
| Spearman ρ（TRT） | **0.437 ***** | [0.412, 0.458] | 文獻 SOTA 範圍 0.35–0.45 ✅ |
| Spearman ρ（GD）  | **0.386 ***** | [0.359, 0.409] | First-pass 閱讀 |
| Held-out R²（log TRT） | 0.188 | — | 泛化能力 |
| OLS β(TRT) = 0.639 | **p < .001 ****** | ΔAIC = +104.6 | 控制頻率/字長/spillover 後仍顯著 |
| OLS β(GD) = 0.406  | **p < .001 ****** | ΔAIC = +62.5 | GD 獨立效果 |
| LMM β(TRT) = 0.049 | **p < .001 ****** | LRT χ²(1) = 184.61, ΔAIC = +182.6 | per-reader random intercepts |
| LMM β(GD) = 0.029  | **p < .001 ****** | LRT χ²(1) = 84.77,  ΔAIC = +82.8  | 同上，GD 模型 |

### 3.2 穩健性驗證：Bootstrap + LOSO

| 分析 | 結果 | 意義 |
|------|------|------|
| Bootstrap 95% CI（2,000 次） | ρ(TRT) = [0.413, 0.459] | 估計值穩定，非運氣 |
| LOSO（Leave-One-Subject-Out） | **14/14 readers p < .001** | 個體差異小，全部顯著 |
| LOSO mean ρ | 0.215 ± 0.044 [0.135, 0.291] | 每位讀者個別預測均顯著 |

### 3.3 大規模跨章節驗證（GECO sentences 2,101–5,284）

| 測試範圍 | n words | ρ(TRT) | ρ(GD) | ΔAIC |
|---------|---------|--------|-------|------|
| 1,000 句（標準） | 4,883 | 0.437 | 0.386 | +104.6 |
| **3,183 句（完整後段）** | **16,318** | **0.440** | **0.400** | **+342.8** |

ρ 在 16,318 詞上完全穩定 → Pipeline 跨章節一致，無過擬合。

### 3.4 跨語料 Zero-Shot 泛化（PROVO）

在 GECO 訓練的模型，**從未見過任何 PROVO 資料**，直接預測：
- **PROVO corpus**（Luke & Christianson 2018）：55 passages，84 L1 readers，混合文類（新聞/Wikipedia/敘事）

| 語料 | 讀者數 | ρ(TRT) | ρ(GD) | OLS β | ΔAIC |
|------|--------|--------|-------|-------|------|
| GECO held-out | 14 L1 | 0.437 *** | 0.386 *** | 0.639 | +104.6 |
| **PROVO（zero-shot）** | **84 L1** | **0.619 ****** | **0.611 ****** | 0.652 | +63.2 |

PROVO 的更高 ρ 反映更多讀者（84 vs 14）帶來的更穩定 mean TRT，以及更廣的文類難度分布。

---

## 四、各成分獨立貢獻（Phase A 元件分解）

| 特徵 | 邊際 ρ（TRT） | Joint β | 說明 |
|------|-------------|---------|------|
| Word length | **0.456 *** **| 0.031 *** | 最強單特徵 |
| Zipf frequency | −0.429 *** | −0.035 *** | 越常見越短 TRT |
| Surprisal（GPT-2） | 0.424 *** | 0.011 *** | 語言模型貢獻 |
| AoA score | 0.276 *** | 0.116 ** | 控制詞頻後仍顯著 |
| Dependency load | −0.021 n.s. | n.s. | GECO fiction 句法單純 |

Joint model（5 特徵）R² = 0.308，優於 composite score（R² = 0.289，ΔAIC = +25.2）

---

## 五、語言模型比較（Phase B）

| 模型 | 參數量 | ρ（TRT） |
|------|--------|---------|
| **GPT-2** | **117M** | **0.398 ***（最佳）** |
| TinyLlama | 1.1B | 0.362 *** |
| GPT-Neo | 1.3B | 0.347 *** |
| GPT-2-Large | 774M | 0.355 *** |
| GPT-2-XL | 1.5B | 0.345 *** |

**Scaling paradox**：模型越大 surprisal 品質反而下降（Oh & Schuler 2023）。GPT-2（117M）保持最佳。

---

## 六、指標速查表

| 指標 | 怎麼解讀 | 好的標準 |
|------|---------|---------|
| **Spearman ρ** | Pipeline 難度排名 vs 眼動時間排名的一致性 | 文獻 SOTA ≈ 0.35–0.45 |
| **TRT** | 所有注視時間（含回視），反映整合困難 | — |
| **GD** | 初次閱讀注視時間，反映詞彙辨識 | GD < TRT 效果正常 |
| **OLS β** | 控制 baseline 後 load_score 的獨立貢獻 | 顯著（p < .05） |
| **ΔAIC** | 加入 load_score 後模型改善量 | >10 = strong |
| **LRT χ²** | 最嚴格的 LMM 比較（含 per-reader 效應） | p < .001 |
| **Bootstrap CI** | 若重抽樣 ρ 的穩定性 | 窄 CI = 估計穩定 |
| **LOSO** | 每位讀者個別預測是否顯著 | 全部顯著 = 跨個體 |
| **Zero-shot** | 跨語料不用 fine-tune 的泛化 | ρ 維持高水準 |

---

## 設計要求

請設計成一頁式的研究報告（Report Page），風格：
- 學術但現代，類似 Notion Research Page 或 Papers with Code 的卡片風格
- 主色調：深藍 + 白色背景，重點標示用琥珀色/綠色
- 每個「突破」用醒目的 badge 或高亮框標示
- 統計顯著性用星號顯示：*** = p<.001，** = p<.01，* = p<.05
- Bootstrap CI 和 LOSO 結果用視覺化方式呈現（小圖示或數字卡片）
- PROVO zero-shot 結果要特別強調（這是最有說服力的泛化證據）
- 結尾加一個 "Paper-ready summary" 段落，顯示論文可直接引用的英文句子

Paper-ready quotes（請直接放到報告最後）：

**主要結果：**
"The pipeline predicted mean TRT with Spearman ρ = 0.437 (95% CI [0.412, 0.458]) and GD ρ = 0.386 (95% CI [0.359, 0.409]) on 4,883 content words from 1,000 held-out GECO sentences. After controlling for word frequency, length, sentence position, and preceding-word spillover, the load score independently predicted TRT (OLS β = 0.639, p < .001, ΔAIC = +104.6; LMM β = 0.049, LRT χ²(1) = 184.61, p < .001, ΔAIC = +182.6, n = 49,154 reader×word observations) and GD (OLS β = 0.406, p < .001, ΔAIC = +62.5; LMM β = 0.029, LRT χ²(1) = 84.77, p < .001, ΔAIC = +82.8)."

**跨語料泛化：**
"Zero-shot transfer to PROVO (Luke & Christianson, 2018; 55 passages, 84 L1 participants, mixed genres) yielded Spearman ρ = 0.619 (GD: ρ = 0.611, both p < .001) on 1,592 content words, confirming cross-corpus generalization (OLS β = 0.652, p < .001, ΔAIC = +63.2). The pipeline was trained exclusively on GECO and had no exposure to PROVO data."

**穩健性：**
"Leave-one-subject-out analysis confirmed significant positive predictions for all 14 individual GECO readers (mean ρ = 0.215 ± 0.044, range [0.135, 0.291], all p < .001)."
