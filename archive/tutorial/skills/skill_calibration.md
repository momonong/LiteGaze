# Agent Skill: IntelligentGaze - {module_name} 模組開發指南

## Metadata
- **領域**：邊緣運算與神經符號 AI
- **適用對象**：Cursor Composer / Vibe Coding 教學助理

## Overview (模組核心目標)
作為 Calibration（校準）與眼動追蹤領域的專家，我將針對 **EyeLingo** 如何在演算法層面結合眼動軌跡與文本佈局進行深度解析。

EyeLingo 的核心突破在於**放棄了「精確座標對齊」的傳統校準思維**，轉而使用「**特徵化區域映射 (Feature-based Region Mapping)**」與「**變形金剛架構 (Transformer-based Encoding)**」來處理雜訊。

---

### 1. EyeLingo 的多模態結合邏輯

EyeLingo 透過一個改編自 **T5 (Text-to-Text Transfer Transformer)** 的編碼器-解碼器架構，實現了眼動與佈局的深度融合：

#### A. 輸入特徵工程 (Input Features)
它不直接輸入原始座標，而是將軌跡與佈局轉化為以下具體特徵：

1.  **Gaze Trajectory (眼動軌跡)**:
    *   輸入原始座標 $(g_{x\_raw}, g_{y\_raw})$ 與滑動平均後的座標 $(g_x, g_y)$。
    *   目的：捕捉細微的注視特徵（Fine-grained）與宏觀的閱讀流向（General patterns）。

2.  **Text Layout (文本佈局)**:
    *   對於每個 Token $w$，提取其邊界框 (Bounding Box) 座標：$w_{x_s}$ (左), $w_{x_t}$ (右), $w_{s_y}$ (上), $w_{t_y}$ (下)。

#### B. 距離與時間特徵公式 (Alignment Metrics)
為了讓模型理解「注視與單字的空間相關性」，EyeLingo 定義了兩個關鍵的映射參數，這也是 Vibe Coding 中最核心的邏輯：

*   **平均注視-Token 距離 $d(g, w)$**:
    $$d(g, w) = \sqrt{\left( \frac{1}{N_g} \sum_{i=1}^{N_g} g_{x_i} - \frac{w_{x_s} + w_{x_t}}{2} \right)^2 + \left( \frac{1}{N_g} \sum_{i=1}^{N_g} g_{y_i} - \frac{w_{s_y} + w_{t_y}}{2} \right)^2}$$
    *(註：$N_g$ 為滑動窗口內的採樣點數。)*

*   **注視持續時間 $t(g, w)$**:
    $$t(g, w) = |\{(g_{x_i}, g_{y_i}) \mid 1 \le i \le N_g \wedge w_{x_s} \le g_{x_i} \le w_{x_t} \wedge w_{s_y} \le g_{y_i} \le w_{t_y}\}|$$
    *(註：計算注視點落在該單字邊界框內的次數。)*

---

### 2. 與傳統座標偏移校準 (Offset-based Calibration) 的區別

傳統方法（如 9 點校準或線性偏移補正）通常假設眼動誤差是線性或可預測的空間位移，但 EyeLingo 採用了完全不同的範式：

| 特性 | 傳統座標偏移校準 | EyeLingo 結合方式 |
| :--- | :--- | :--- |
| **校準目標** | 減少 $Gaze(x, y)$ 與 $Target(x, y)$ 的 Euclidean Error。 | 學習 $Gaze\ Trajectory$ 與 $Text\ Layout$ 的**潛在關聯 (Latent Correlation)**。 |
| **容錯機制** | 依賴頻繁的重新校準 (Re-calibration) 來應對姿勢變動。 | **注意力機制 (Cross-Attention)**：模型能自動忽略偏離文本行的噪訊，識別注視模式。 |
| **處理邏輯** | 硬性映射 (Hard Mapping)：$Gaze \to Point$。 | 軟性特徵 (Soft Features)：將注視距離與持續時間作為權重輸入模型。 |
| **環境需求** | 需要高精度硬體（如 Tobii Pro Nano）。 | 支援低精度硬體（如 Webcam），因為 PLM 的語義資訊能補償空間定位的不足。 |

---

### 3. Vibe Coding 實作建議：演算法流程邏輯

如果你要開發類似的系統，應遵循以下偽代碼邏輯進行校準融合：

```python
# 1. 建立滑動窗口 (Sliding Window, e.g., 1s)
window_gaze = get_current_gaze_samples() # 60Hz -> 60 points

# 2. 定義感興趣區域 (Region of Interest, ROI)
# 利用 Gaze 點的邊界框擴展，篩選出候選單字群
roi_words = fetch_words_in_bounding_box(min(window_gaze.x), max(window_gaze.x), ...)

# 3. 特徵提取 (Feature Extraction for Transformer Decoder)
token_features = []
for word in roi_words:
    dist = calculate_euclidean_distance(window_gaze, word.bbox_center)
    duration = count_points_inside(window_gaze, word.bbox)
    
    # 將空間特徵與 PLM (RoBERTa) 的語義向量拼接
    combined_vec = concatenate([
        word.roberta_embedding, 
        positional_encoding(word.x, word.y),
        normalized(dist), 
        normalized(duration)
    ])
    token_features.append(combined_vec)

# 4. 變形金剛推理 (Inference)
# 透過 Cross-Attention 讓模型自行判斷哪一個注視點對應哪一個單字
is_unknown = model.predict(token_features)
```

### 專家總結
EyeLingo 的精髓在於它**承認眼動數據是不精確的**。它利用 **Transformer 的 Cross-Attention 模組**，將不穩定的眼動點與結構化的文本佈局進行「模糊匹配」。這種方法讓系統即使在用戶姿勢改變導致注視點偏移 1-2 行的情況下，仍能透過注讀時間與語言模型的難度預測，準確鎖定未知的單字。

## Core Concepts (關鍵演算法與理論)
你好，我是 Calibration 領域專家。針對 EyeLingo 論文中所提出的 **Positional Data Encoding** 機制，該架構是基於 **T5 (Text-to-Text Transfer Transformer)** 的 Encoder-Decoder 結構進行改良，旨在解決眼動儀與文本座標精準對齊（Mapping）的魯棒性問題。

以下是該機制的技術細節與演算法邏輯，可直接用於 Vibe Coding 或系統實作參考：

---

### 1. Encoder：從 Gaze Trace 提取行為特徵
Encoder 的核心任務是將隨時間變化的凝視點軌跡（Gaze Trajectory）轉化為高維度的行為特徵向量 $H_g$。

*   **輸入數據 (Input Modalities)：**
    為了同時捕捉「精細運動」與「總體趨勢」，輸入為四通道的序列：
    1.  **Raw Gaze ($g_{x\_raw}, g_{y\_raw}$):** 未經處理的原始座標，保留震顫與微跳視（Microsaccades）信息。
    2.  **Moving Averaged Gaze ($g_x, g_y$):** 經滑動平均濾波後的座標，代表穩定的凝視路徑。
*   **參數設定：**
    *   **採樣率 (Sampling Rate):** 60Hz。
    *   **時間窗口 (Sliding Window):** 1 秒核心窗口，前後各擴充 1 秒，總計 **3 秒** 的軌跡數據（共 180 個樣本點）。
*   **提取邏輯：**
    $$H_g = \text{Encoder}(g_x, g_y, g_{x\_raw}, g_{y\_raw})$$
    利用 Transformer 的 Self-Attention 機制，Encoder 會學習軌跡中的時間相關性，識別出如「回視 (Regression)」、「長時凝視 (Long Fixation)」等與生詞閱讀障礙高度相關的模式。

---

### 2. Decoder：利用特徵進行生詞預測
Decoder 並非生成文本，而是執行一個「序列標註」任務。它結合了 Encoder 輸出的行為編碼 $H_g$ 與目標 Token 的空間屬性。

#### A. Token 級別特徵輸入
對於每個候選 Token $w$，Decoder 接收以下空間特徵：
1.  **座標位置 ($w_x, w_y$):** Token 在頁面上的定界框（Bounding Box）中心。
2.  **平均注視距離 ($d(g, w)$):** 計算該窗口內所有凝視點中心與 Token 中心的歐幾里得距離。
    $$\text{Target}_x = \frac{w_{x\_start} + w_{x\_end}}{2}, \quad \text{Target}_y = \frac{w_{y\_start} + w_{y\_end}}{2}$$
    $$d(g, w) = \sqrt{ (\frac{1}{N_g}\sum g_{x_i} - \text{Target}_x)^2 + (\frac{1}{N_g}\sum g_{y_i} - \text{Target}_y)^2 }$$
3.  **注視時長 ($t(g, w)$):** 落在 Token 邊界框內的凝視點總數。
    $$t(g, w) = |\{ (g_{xi}, g_{yi}) \mid w_{x\_s} \le g_{xi} \le w_{x\_t} \land w_{y\_s} \le g_{yi} \le w_{y\_t} \}|$$

#### B. 跨模態注意力機制 (Cross-Attention)
Decoder 透過 Cross-Attention 模塊，將 Token 的空間屬性作為 **Query**，去檢索 Encoder 產生的行為特徵 $H_g$ (**Keys & Values**)。
$$P = \text{Decoder}(H_g, w_x, w_y, d(g, w), t(g, w))$$

*   **邏輯意涵：** 即使眼動座標因為設備誤差（如校準偏移）沒有精確落在 Token 上，Decoder 也能透過 $d(g, w)$ 的相對變化趨勢，判斷當前 3 秒內的軌跡特徵（如反覆確認）是否指向該特定的 Token。

---

### 3. Vibe Coding 實作要點邏輯

若要實作此邏輯，請參考以下擬虛擬代碼結構：

```python
# 1. 數據預處理 (60Hz, 3s window)
gaze_input = stack([raw_x, raw_y, smooth_x, smooth_y]) # Shape: [Batch, 180, 4]

# 2. Encoder 提取行為特徵
gaze_memory = T5Encoder(gaze_input) # 輸出 H_g

# 3. 準備 Decoder 的 Token 空間特徵
# tokens_pos: [Batch, SeqLen, 2] (x, y coordinates)
# tokens_dist: [Batch, SeqLen, 1] (avg distance to gaze cluster)
# tokens_duration: [Batch, SeqLen, 1] (fixation count)
token_features = concat([tokens_pos, tokens_dist, tokens_duration], dim=-1)

# 4. Decoder 融合行為與空間信息
# 使用 Cross-Attention 讓 token_features 去 query gaze_memory
positional_encoding = T5Decoder(token_features, memory=gaze_memory)

# 5. 最終預測 (結合 RoBERTa 的文本特徵)
# combined_features = concat([positional_encoding, roberta_embeddings, knowledge_embeddings])
# prediction = Classifier(combined_features)
```

### 專家點評 (Calibration Insight)：
EyeLingo 的精髓在於**「容錯校準」**。傳統方法依賴絕對座標對齊（Mapping），但本模型透過 **Encoder-Decoder** 結構將「注視行為」與「注視位置」解耦。即使 Webcam 產生的 Gaze Trace 存在 67px 以上的誤差（如文中 4.2 節所述），Decoder 仍能透過全域軌跡特徵與 Token 的相對空間關係捕捉到生詞信號，這是其 F1-score 能達到 71.1% 的關鍵原因。

## Methodology (最佳實作路徑與架構)
你好！我是 Calibration 領域的專家。針對你的需求，我們將結合 EyeLingo 論文中提到的**語言學特徵（Linguistic Characteristics）**，優化你現有的 **Gravity Snap（重力吸附）** 演算法。

在 Calibration 的實務中，單純依靠歐幾里得距離的吸附會忽略使用者的「認知意圖」。根據文獻 2502.10378v1 的發現：**生字（Unknown Words）會導致更長的注視時間（Fixation Duration）**，而功能詞（如 the, and）則常被跳過。

以下是為你設計的動態引力加權參數 $\alpha$ 的計算邏輯：

---

### 🧠 基於語言學先驗的 Gravity Snap 優化方案

#### 1. 核心邏輯
我們將傳統的引力係數 $\alpha$ 改為動態權重 $\alpha_{dynamic}$。其核心思想是：**當單字越可能是「生字」或「關鍵意義詞」時，該單字的吸引力（重力）越強**，以補償眼動儀在小行距下的垂直偏移誤差。

#### 2. 公式定義
$$ \alpha_{dynamic} = \alpha_{base} \times (1 + \text{DifficultyScore}) \times W_{pos} $$

其中：
*   **$\alpha_{base}$**: 基礎重力係數（基於 Gaze 與 Word 中心點的物理距離）。
*   **$\text{DifficultyScore}$**: 基於詞頻的難度分數（0.0 ~ 1.0）。
*   **$W_{pos}$**: 基於詞性的加權權限（0.2 ~ 1.2）。

---

### 3. 參數計算細節 (Vibe Coding 邏輯)

#### A. 詞頻難度分數 ($Score_{freq}$)
根據文獻，詞頻（Term Frequency）是判斷生字的關鍵。我們使用 Zipf Scale 或對數逆頻數：
*   **邏輯**：單字在語料庫中出現頻率越低，分數越高。
*   **實作參考**：
    ```python
    # 假設 freq 為 0.0~1.0 的歸一化頻率 (1.0 代表極常見詞如 "the")
    difficulty_score = 1.0 - normalized_frequency 
    ```

#### B. 詞性加權係數 ($W_{pos}$)
文獻指出內容詞（Content Words）比功能詞更值得注視。
*   **權重表設計**：
    *   **1.2 (極強引力)**: 名詞 (NN), 動詞 (VB), 形容詞 (JJ) —— 這些通常是生字的候選。
    *   **0.8 (中等引力)**: 副詞 (RB)。
    *   **0.2 (極弱引力)**: 限定詞 (DT), 連接詞 (CC), 介係詞 (IN) —— 讀者通常會掃視跳過，不應產生強吸附。

---

### 4. 演算法偽代碼 (直接輔助開發)

```markdown
### Gravity Snap Alpha Calculation Logic

**Input:** 
- `dist`: Gaze point to Word center distance
- `term_freq`: Frequency of the word (0-1, high means common)
- `pos_tag`: Part of speech (from Spacy/NLTK)

**Algorithm:**

1. **Calculate Physical Alpha (Base):**
   `alpha_base = exp(-dist^2 / (2 * sigma^2))`  // 經典高斯引力

2. **Calculate Linguistic Weight:**
   - **Difficulty Component:** `score_diff = clamp(1.0 - term_freq, 0, 1)`
   - **POS Component:**
     ```
     switch (pos_tag):
       case 'NOUN', 'VERB', 'ADJ': w_pos = 1.2
       case 'ADV': w_pos = 0.8
       case 'DET', 'CONJ', 'PREP': w_pos = 0.2
       default: w_pos = 1.0
     ```

3. **Final Dynamic Alpha:**
   `alpha_final = alpha_base * (1.0 + 0.5 * score_diff) * w_pos`

4. **Apply Snap:**
   `gaze_snapped = gaze_raw + alpha_final * (word_center - gaze_raw)`
```

---

### 專家觀點：為什麼這樣做有效？

1.  **補償行間距誤差**：論文 2.2 提到，小行距 (1.0-2.0) 下眼動儀極易產生垂直漂移。透過降低功能詞（如 "of", "a"）的引力，並增加實質單字（如 "ignominious"）的引力，可以有效防止注視點被錯誤地吸附到無意義的虛詞上。
2.  **模擬「Eye-Mind Hypothesis」**：根據文獻引用 [25]，人類注視點留在一個詞上的時間與處理難度成正比。動態引力機制在演算法層面模擬了這種心理現象，讓 Calibration 具備「語意感知」能力。
3.  **效能平衡**：此方法不需要像 EyeLingo 運行大型 RoBERTa 模型，只需預查一張詞頻表與簡單的 POS Tagging，非常適合你要求的輕量化 Snap 公式。

這套邏輯能顯著提升在閱讀情境下的 Gaze-to-Word Mapping 精確度。

## Constraints & Edge Cases (軟硬體限制與極端狀況)
你好，我是 Calibration 領域的專家。針對你提到的真實閱讀場景中「行距狹窄（1.0-2.0）」導致的垂直位移雜訊與「座標吸附錯誤（Line Misassignment）」問題，EyeLingo 文獻提出了一套從**空間濾波、時間窗口到多模態特徵解耦**的處理方案。

這套方法的核心邏輯在於：**不再強求「點對點」的硬性映射（Hard Mapping），而是透過「區域編碼（Region Encoding）」與「語言模型先驗（PLM Priors）」來抵消物理層面的校準誤差。**

以下是具體的技術實作細節，可直接用於 Vibe Coding 的邏輯構建：

---

### 1. 空間去噪算法：動態邊界框 (Bounding Box) 剪枝
為了防止因眨眼或垂直漂移導致的座標「跳行」，系統在定位感興趣區域 (Region of Interest, RoI) 時執行以下邏輯：

*   **垂直離群點剔除規則：**
    *   **輸入：** 1 秒滑動窗口內的所有 $y$ 座標集合 $Y$。
    *   **邏輯：** 分析 $y$ 座標分佈。若發現一小組數據點與其餘主體數據點的垂直距離超過 **3 個行高度 (3 line widths)**，則將該小組視為由眨眼或大幅度噪訊引起的離群點並予以刪除。
    *   **失效保護 (Fail-safe)：** 若窗口內所有數據點的 $y$ 值波動均過大（無法形成穩定的行聚集），則直接捨棄該 1 秒窗口。

### 2. 特徵提取：軟性距離度量 (Soft Distance Metrics)
與其將 gaze 點硬性指定給某一行或某個字，EyeLingo 計算了每個 Token 與 Gaze 軌跡之間的「概率相關性」：

*   **平均注視距離 ($d(g, w)$):**
    $$d(g,w) = \sqrt{ \left( \frac{1}{N_g} \sum g_{x,i} - \frac{w_{x,s} + w_{x,t}}{2} \right)^2 + \left( \frac{1}{N_g} \sum g_{y,i} - \frac{w_{s,y} + w_{t,y}}{2} \right)^2 }$$
    *   *邏輯：* 計算 Token 中心與注視點中心間的歐幾里得距離，而非布林值判定。
*   **注視時長 ($t(g, w)$):**
    統計在該 Token 矩形邊界框（Bounding Box）內的注視點總量。

### 3. 軌跡編碼：Transformer-based Encoder-Decoder
這是應對垂直誤差的「殺手鐧」。系統不使用手工規則校正行數，而是將原始數據餵入模型自動學習偏移規律：

*   **雙流輸入：**
    1.  **Raw Gaze:** 保留原始的高頻抖動特徵。
    2.  **Moving Averaged Gaze:** 使用移動平均濾波後的平滑軌跡，過濾微小的物理震顫。
*   **模型架構：** 基於 **T5 (Transformer)** 的編碼器。
    *   **Encoder:** 輸入 $g_x, g_y, g_{x,raw}, g_{y,raw}$，學習注視模式。
    *   **Decoder:** 輸入 Token 的絕對座標 $(w_x, w_y)$ 與上述的距離/時長特徵。
    *   **效果：** Cross-attention 機制會自動學會：當垂直座標處於兩行之間時，結合上下文（Context）來判斷該軌跡更可能屬於哪一行。

### 4. 語言模型補償 (Linguistic Prior Compensation)
當校準物理失效（垂直漂移無法挽回）時，系統會調高 **RoBERTa** 權重的影響力：

*   **原理：** 難詞檢測不只看「看多久」，也看「這字難不難」。
*   **實作：** 即使 Gaze 座標因為漂移「吸附」到了錯誤的行，RoBERTa 提取的上下文語意特徵（如 Term Frequency, POS Tags, NER）仍能提供強力的分類依據。
*   **消融實驗證明：** 移除 Textual Encoding 後 F1-score 從 **71.1% 暴跌至 16.2%**，證明了在雜訊環境下，語言模型的「文字重力」比注視點座標更穩定。

---

### 具體參數總結表 (For Implementation)

| 參數 (Parameter) | 設定值 (Value) | 說明 (Description) |
| :--- | :--- | :--- |
| **Window Length** | 1.0 second | 用於定位候選詞的滑動窗口 |
| **Context Extension** | $\pm$ 1.0 second | 總計取 3 秒 gaze 數據來提取行為特徵 (前後各加1s) |
| **Sampling Rate** | 60 Hz | 穩定採樣率，webcam 數據需透過 Spline 插值對齊 |
| **Outlier Threshold** | 3.0 * Line Height | 判定 Y 軸座標跳行的硬性閾值 |
| **Font / Line Size** | 10 pt / 1.0-2.0 spacing | 真實環境場景，對應 y 軸約 3.3mm - 4.1mm |
| **Pre-trained Model** | RoBERTa-base | 提供 768 維的語意向量，用於糾正座標偏差帶來的誤判 |

### 專家建議 (Expert Tips for Vibe Coding):
如果你在實作時發現注視點依然在兩行之間「左右橫跳」，請導入 **Focal Loss**（如論文式 8 所示），這能強迫模型專注於處理那些「座標模糊但語意明確」的難樣本，防止模型過度依賴不穩定的 $y$ 座標。

## Example Q&A (Vibe Coding 指引與範例)
你好，我是 calibration 領域的專家。針對 **EyeLingo** 文獻中所提到的挑戰——即「即便使用專業眼動儀，注視點（Gaze point）與文字座標（Word Bounding Box）之間仍存在物理偏移與垂直誤差」——我們需要一套動態校準機制。

在 Vibe Coding（語境開發）的範式下，我們不再撰寫硬編碼的規則，而是透過「引力場（Gravity Field）」的概念，讓語言模型（PLM）預測的機率作為引力，將偏移的注視點「吸附」回正確的文字行上。

以下是為你準備的兩個 Vibe Coding 提問引導（Prompts），請直接輸入給你的 AI 編碼助理：

---

### 提問一：建構基於語言權重的「認知引力場」映射器

**目標：** 讀取原始數據，並計算每個單字對注視點產生的「拉力」。

> **Vibe Coding 提示詞：**
> 「請寫一個 Python 類別 `GazeGravityMapper`。這個工具需要讀取一個 `raw.csv`（包含 `timestamp`, `gaze_x`, `gaze_y`）以及一個 `cognitive_weights.json`（包含單字座標 `bbox` 與來自 RoBERTa 的機率權重 `alpha`）。
>
> **邏輯要求：**
> 1. **引力公式：** 請實作一個函數計算單字 $w$ 對注視點 $g$ 的引力 $F$。公式為：$F(g, w) = \frac{\alpha_w}{d(g, w)^2}$，其中 $d$ 是歐幾里得距離，$\alpha_w$ 是該單字的認知重要性權重。
> 2. **座標加權：** 針對每一影格的注視點，計算其週邊 50px 內所有單字的引力總合，輸出一個『期望座標』 $Gaze_{target} = \frac{\sum (F_i \cdot Center_i)}{\sum F_i}$。
> 3. **輸出：** 將處理後的結果儲存為 `refined_gaze.json`。」

---

### 提問二：實作「動態偏移補償（Dynamic Drift Correction）」函式

**目標：** 根據第一步產生的引力落差，自動修正眼動儀的系統性偏移（如使用者姿勢改變造成的垂直位移）。

> **Vibe Coding 提示詞：**
> 「請實作一個 Python 函式 `calibrate_dynamic_offset(raw_gaze, target_gaze)`，用於修正眼動儀的長期漂移（Drift）。
>
> **演算法邏輯：**
> 1. **位移向量：** 計算原始注視點與引力目標點之間的向量差 $\Delta = Gaze_{target} - Gaze_{raw}$。
> 2. **滑動窗口平滑：** 為了避免眼動跳躍（Saccade）造成的干擾，請使用長度為 60 影格（約 1 秒）的滑動平均（Moving Average）來計算『當前系統偏移量』 $Offset_{bias}$。
> 3. **座標修正：** 最終修正後的座標為 $Gaze_{calibrated} = Gaze_{raw} + Offset_{bias}$。
> 4. **異常處理：** 若 $\Delta$ 超過 3 行文字的高度（約 60-80px），視為使用者正在切換閱讀區域，此時應降低校準強度（權重設為 0.1），避免過度拉伸座標。
>
> 請確保程式碼具備高效能，能處理 `raw.csv` 中數萬行的數據。」

---

### 專家技術補充（供你理解背後邏輯）：

在實作這兩個函式時，你需要提供給 AI 的核心邏輯如下：

*   **資料結構參考：**
    *   `raw.csv`: `[t, x, y]`
    *   `cognitive_weights.json`: `{ "word": "ignominious", "bbox": [x1, y1, x2, y2], "alpha": 0.85 }` (Alpha 越高，代表 PLM 認為它是難詞，使用者更有可能在此停頓)。
*   **關鍵參數：**
    *   **垂直敏感度：** 由於 EyeLingo 提到行距（Line Spacing）是最大的誤差來源，你的 `Offset_{bias}` 應該給予 Y 軸更高的平滑權重。
    *   **引力閾值：** 只有當 $\alpha > 0.5$ 的單字才具備強大的引力，這符合論文中「難詞會吸引更多 Fixation」的認知科學假設。

這兩段 Prompt 將引導 AI 幫你完成一個**具備認知感知能力的動態校準系統**，而非單純的幾何修正。