# Agent Skill: IntelligentGaze - {module_name} 模組開發指南

## Metadata
- **領域**：邊緣運算與神經符號 AI
- **適用對象**：Cursor Composer / Vibe Coding 教學助理

## Overview (模組核心目標)
EyeLingo 系統的核心概念在於**實時、高準確度地檢測 ESL 學習者在閱讀過程中遇到的生詞，透過整合眼動軌跡 (Gaze) 和預訓練語言模型 (PLM) 的優勢，並藉由 PLM 提供的豐富語言特徵來彌補傳統眼動追蹤在精確度上的不足。**

以下是 EyeLingo 如何結合這兩種模態來解決傳統眼動追蹤準確度問題的具體機制：

### EyeLingo 系統的核心概念及結合機制

EyeLingo 將生詞檢測視為一個二元分類問題，利用 transformer-based 模型處理來自兩種模態的資訊：

1.  **眼動軌跡 (Gaze) 的作用 - 定位閱讀區域與提供使用者行為資訊：**
    *   **解決傳統精準度問題的策略：** 傳統眼動追蹤方法常試圖將每個凝視點精確地映射到單個詞彙上，但在實際閱讀環境中（例如：行距小、頭部移動），眼動追蹤的精度（約 $0.3^\circ - 1.1^\circ$）往往不足以實現詞級的精確映射，導致大量誤差。EyeLingo 透過以下方式應對：
        *   **定位「興趣區域」(Region of Interest, ROI)：** EyeLingo 不追求將單個凝視點精確匹配到詞，而是從一個 **1 秒的滑動時間窗**內的凝視序列中推導出一個 **文字 bounding box**。這個 bounding box 定義了使用者在該時間段內最可能閱讀的文字區域，從而更穩健地處理凝視數據的固有不準確性。
        *   **自動學習凝視模式：** 系統採用一個基於 **T5 的編碼器-解碼器模型**來處理凝視數據。
            *   **編碼器 (Encoder)** 接收原始凝視軌跡 $(g_x^{raw}, g_y^{raw})$ 和平滑後的凝視軌跡 $(g_x, g_y)$，學習捕捉使用者行為的細粒度和通用位置資訊，輸出凝視位置編碼 $H_g$：
                $$ H_g = \text{Encoder}(g_x, g_y, g_x^{raw}, g_y^{raw}) $$
            *   **解碼器 (Decoder)** 則結合 $H_g$ 以及每個候選詞彙的坐標 $(w_x, w_y)$、凝視-詞彙距離 $d(g,w)$ 和凝視時長 $t(g,w)$ 來預測詞彙的位置相關資訊 $P$。其中 $d(g,w)$ 和 $t(g,w)$ 反映了使用者對該詞彙的認知處理強度。
                *   凝視-詞彙距離：
                    $$ d(g,w) = \sqrt{\left(\frac{\sum_i^{N_g} g_{x_i}}{N_g} - \frac{w_{x_s} + w_{x_t}}{2}\right)^2 + \left(\frac{\sum_i^{N_g} g_{y_i}}{N_g} - \frac{w_{y_s} + w_{y_t}}{2}\right)^2} $$
                *   凝視時長：
                    $$ t(g,w) = |\{(g_{x_i}, g_{y_i}) | 1 \le i \le N_g \land w_{x_s} \le g_{x_i} \le w_{x_t} \land w_{y_s} \le g_{y_i} \le w_{y_t}\}| $$
                *   解碼器輸出：
                    $$ P = \text{Decoder}(H_g, w_x, w_y, d(g,w), t(g,w)) $$

2.  **預訓練語言模型 (PLM, 如 RoBERTa) 的作用 - 提供強大語言線索以彌補凝視不足：**
    *   **捕捉豐富的語言特徵：** 針對在 ROI 內識別出的候選詞彙，EyeLingo 採用 **RoBERTa** 模型來編碼其文字資訊。RoBERTa 能夠捕捉詞彙的上下文語義、句法結構和詞彙之間的複雜關係 $Z$：
        $$ Z = \text{RoBERTa}(s) $$
        其中 $s$ 是 ROI 內的文本內容。PLM 的強大語言理解能力為判斷詞彙難度提供了核心依據。
    *   **詞級知識增強：** 由於 PLM 會將詞彙切分為 token，可能損失部分詞級資訊，因此 EyeLingo 引入了可學習的嵌入層 $K$ 來補充詞彙層面的先驗知識，例如：
        *   詞頻 (term frequency)
        *   詞性 (part-of-speech, POS)
        *   命名實體識別 (named entity recognition, NER)
        這些特徵進一步強化了模型對詞彙固有難度的判斷，減少了對凝視精確度的依賴。

3.  **兩者協同工作 - 整合與分類：**
    *   最終，EyeLingo 將從凝視模態得到的**位置編碼 $P$**、從 PLM 得到的**文字編碼 $Z$** 以及**詞級知識嵌入 $K$** 進行拼接：
        $$ H = [P; Z; K] $$
    *   這個整合後的向量 $H$ 被輸入到一個**二元分類器 (邏輯迴歸模組)** 中，輸出該詞彙為生詞的機率 $p$：
        $$ p = \sigma(W_o \cdot H + b_o) $$
    *   訓練時採用**焦點二元交叉熵損失 (Focal Binary Entropy Loss)**，以應對生詞與已知詞之間的類別不平衡問題：
        $$ \mathcal{L}(p, \hat{y}) = -\alpha \hat{y} (1-p)^\gamma \log(p) - (1-\alpha) (1-\hat{y}) p^\gamma \log(1-p) $$
        其中 $\alpha$ 和 $\gamma$ 是控制權重和模型對難例關注速度的超參數。

**總結來說，EyeLingo 透過以下方式克服傳統眼動追蹤的準確度問題：**

*   **Gaze 提供「粗粒度」的區域定位和「使用者相關」的行為線索**，而非精確的點對點匹配，從而對眼動數據的雜訊具有更高的容忍度。
*   **PLM 提供「細粒度」且「上下文豐富」的語言理解**，作為判斷詞彙難度的主要依據。即使凝視數據不夠精確，PLM 也能從周圍的文字語境中推斷詞彙的難度。
*   **兩者結合形成互補：** Gaze 確保實時性與個性化，PLM 確保高準確度與對凝視雜訊的魯棒性。這使得 EyeLingo 即使在嘈雜的網路攝影機數據上也能表現良好，並實現了遠超傳統基線的性能。

## Core Concepts (關鍵演算法與理論)
根據文獻 "2502.10378v1.pdf"，預訓練語言模型 (PLM)，特別是 RoBERTa，以及額外的詞級知識嵌入，能有效提取出與使用者認知困難度相關的語言學特徵。這些特徵對於判斷 ESL 學習者是否遇到生詞至關重要。

以下是 PLM 提取或與其協同作用的語言學特徵及其對判斷認知困難度的幫助：

### 1. 預訓練語言模型 (PLM) 提取的上下文語言學特徵 (Contextual Linguistic Features)

文中採用 RoBERTa 模型來捕捉文本中的豐富語言學資訊。RoBERTa 作為基於 Transformer 架構的預訓練語言模型，能夠學習複雜的模式和依賴關係，這對理解單詞難度至關重要。

*   **特徵類型**: 上下文嵌入 (Contextual Embeddings)、句法結構 (Syntactic Structures)、語義關係 (Semantic Relationships)。
*   **提取方式**: RoBERTa 模型透過其自注意力 (self-attention) 機制和前饋層，處理感興趣區域 (Region of Interest) 內的文本。它將文本數據 $s \in \mathbb{R}^{n_{txt}}$ 編碼為 $Z \in \mathbb{R}^{n_{txt} \times n_r}$，其中 $n_{txt}$ 是文本中的 tokens 數量，$n_r$ 是 RoBERTa 的隱藏維度。
    $$Z = \text{RoBERTa}(s)$$
*   **如何幫助判斷認知困難度**:
    *   **捕捉上下文相關性**: 單詞的難度不僅取決於其本身，也與其所處的上下文密切相關。RoBERTa 透過理解單詞在句子、段落中的作用和意義，判斷單詞在特定語境下是否難以理解。例如，一個單詞在孤立情況下可能已知，但在特定複雜語境中可能會引起認知困難。
    *   **語義和句法複雜度**: RoBERTa 能夠捕捉複雜的句法結構和語義關係。如果一個單詞的句法角色不明確或其語義與上下文不符，這可能增加學習者的認知負擔。
    *   **上下文獨特性 (Contextual Distinctiveness)**: 文獻中提到，上下文獨特性是與單詞難度強烈相關的語言學先驗知識。RoBERTa 的上下文嵌入自然地反映了單詞在不同語境下的獨特性，幫助模型識別那些在特定語境中意義不尋常或難以推斷的單詞。

### 2. 詞級知識嵌入 (Word-level Knowledge Embeddings)

由於預訓練語言模型在分詞 (tokenization) 過程中可能損失部分詞級資訊，因此論文引入了額外的詞級知識嵌入來彌補這一不足。這些嵌入作為可學習的參數，與 RoBERTa 的輸出結合。

*   **特徵類型**: 詞頻 (Term Frequency)、詞性 (Part-of-Speech, POS)、命名實體識別 (Named Entity Recognition, NER)。
*   **引入方式**: 這些資訊被編碼為可學習的嵌入 $K \in \mathbb{R}^{n_{txt} \times n_k}$，其中 $n_k$ 是知識嵌入的維度。它們直接添加到模型的輸入中，以補充 RoBERTa 的上下文輸出。
    $$H = [P; Z; K]$$
    其中 $P$ 是位置數據編碼，$Z$ 是 RoBERTa 輸出，$K$ 是詞級知識嵌入。
*   **如何幫助判斷認知困難度**:
    *   **詞頻 (Term Frequency)**:
        *   **邏輯**: 單詞的出現頻率與其熟悉度呈正相關。對於 ESL 學習者來說，低頻詞通常比高頻詞更難理解。直接提供詞頻資訊可以作為單詞難度的強力指標。
        *   **幫助**: 模型可以學習到詞頻越低，該詞越可能是一個生詞。
    *   **詞性 (Part-of-Speech, POS)**:
        *   **邏輯**: 單詞的詞性提供了其在語法中的功能資訊。對於學習者來說，一個單詞的詞性如果與其常見用法不同，或者在複雜句法結構中出現，都可能增加理解難度。
        *   **幫助**: 詞性資訊可以幫助模型識別語法上複雜或非常規的單詞用法，進而判斷其認知難度。
    *   **命名實體識別 (Named Entity Recognition, NER)**:
        *   **邏輯**: 命名實體（如人名、地名、組織名）通常是專有名詞，其難度來源於知識背景而非純粹的詞彙量。識別出這些詞有助於模型區分普通詞彙和特定資訊。
        *   **幫助**: NER 標籤可以幫助模型區分普通詞彙和命名實體，因為命名實體的認知困難度可能與詞彙本身的難度評估邏輯不同。例如，一個新地名對所有學習者都可能是生詞，而一個普通動詞的難度則因人而異。

總而言之，該論文透過將 RoBERTa 捕獲的**上下文語義和句法特徵**與**詞頻、詞性、命名實體**等明確的詞級知識相結合，為判斷 ESL 學習者的認知困難度提供了全面且精確的語言學基礎。這種結合使得模型即使在眼動數據存在誤差的情況下，也能保持高準確性。

## Methodology (最佳實作路徑與架構)
作為 `text_model` 領域的專家，根據 `2502.10378v1.pdf` 文獻的啟發，將語言模型（例如 bert-tiny）的預測機率轉換為詞彙驚奇度 (Word Surprisal)，並進一步映射到 UI 的動態引力半徑 (Gravity Radius) 是一個巧妙且符合文獻精神的應用。

文獻中提到：
1.  **語言模型的角色**：PLMs（如 RoBERTa）「展示了捕捉豐富語言資訊的強大能力 [11, 29]，這與詞彙難度密切相關 [18]。」(Section 1)
2.  **Gaze 的角色**：Gaze 提供「使用者依賴的及時資訊來檢測不同使用者不同的未知詞彙。」(Contributions, Section (2))
3.  **模型結合**：EyeLingo「整合了 PLM 提供的語言資訊和凝視軌跡，使用 transformer-based 模型預測未知詞彙。」(Section 1)

詞彙驚奇度 (Word Surprisal) 正是量化語言模型對一個詞彙「預期程度」的核心指標，低驚奇度表示詞彙符合上下文預期，高驚奇度則表示詞彙不符預期，可能更難理解。這與文獻中「詞彙難度」和「未知詞彙檢測」的概念高度契合。動態引力半徑則可以作為 UI 上呈現這種「難度/驚奇度」的視覺或互動強度指標。

以下是 Python 偽程式碼邏輯，旨在輔助 Vibe Coding：

---

## 語言模型預測機率轉換為詞彙驚奇度與 UI 動態引力半徑

### 1. 核心概念與定義

*   **詞彙驚奇度 (Word Surprisal)**: 定義為 `-log2(P(word | context))`。其值越高，表示詞彙在給定上下文中的出現機率越低，即越「驚奇」或越「困難」。
*   **動態引力半徑 (Gravity Radius)**: UI 上詞彙周圍的互動或視覺強調區域，其大小與詞彙驚奇度正相關。驚奇度越高，半徑越大，暗示該詞彙可能需要更多關注或輔助。
*   **`bert-tiny` 的作用**: 假設 `bert-tiny` 作為一個 Masked Language Model (MLM)，能夠預測在給定上下文下，特定位置上每個詞彙的機率分佈。

### 2. 參數設定 (Constants and Hyperparameters)

```python
# UI 相關參數
MIN_GRAVITY_RADIUS_PX = 10  # 最小引力半徑 (像素)
MAX_GRAVITY_RADIUS_PX = 50  # 最大引力半徑 (像素)

# 詞彙驚奇度映射參數 (需根據實際 bert-tiny 輸出分佈進行經驗性校準)
# 這些值應從真實語料庫中 bert-tiny 預測的 surprisal 分佈中得出，
# 例如，取第 5 和第 95 百分位數作為參考。
MIN_EXPECTED_SURPRISAL = 1.0  # 預期詞彙驚奇度下限 (log2 單位)
MAX_EXPECTED_SURPRISAL = 10.0 # 預期詞彙驚奇度上限 (log2 單位)

# 語境窗口大小 (參考文獻中的 64 個 tokens)
CONTEXT_WINDOW_SIZE = 64

# 其他模型相關參數
import math
import torch
from transformers import BertTokenizer, BertForMaskedLM

# 載入 bert-tiny 模型和 Tokenizer
# 注意: 請根據實際使用的 bert-tiny 模型名稱進行調整，例如 'google/bert_uncased_L-2_H-128_A-2'
BERT_MODEL_NAME = 'google/bert_uncased_L-2_H-128_A-2'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
lm_model = BertForMaskedLM.from_pretrained(BERT_MODEL_NAME)
lm_model.eval() # 設置為評估模式
```

### 3. 核心邏輯：從 LM 機率到動態引力半徑

#### 步驟 1: 計算詞彙驚奇度 (Calculate Word Surprisal)

此函數將使用 `bert-tiny` 預測一個詞彙在給定上下文中的機率，並轉換為驚奇度。

```python
def calculate_word_surprisal(word: str, context_sentence: str, lm_model, tokenizer) -> float:
    """
    計算一個詞彙在給定上下文中的驚奇度。
    這通常涉及將目標詞彙 mask 掉，讓 LM 預測其機率。

    Args:
        word (str): 要計算驚奇度的目標詞彙。
        context_sentence (str): 包含目標詞彙的完整句子或上下文段落。
        lm_model: 預載入的 BertForMaskedLM 模型。
        tokenizer: 預載入的 BertTokenizer。

    Returns:
        float: 詞彙的驚奇度值 (-log2(P(word | context)))。
               如果無法計算，返回一個預設高值。
    """
    # 查找目標詞彙在句子中的位置
    # 注意：這裡假設 word 是一個單一的 token，實際應用中可能需要更複雜的 token 對齊。
    # 對於多詞 token (subwords)，可以計算其驚奇度的平均值或總和。
    word_tokens = tokenizer.tokenize(word)
    if not word_tokens:
        # 無法分詞，返回高驚奇度
        return MAX_EXPECTED_SURPRISAL * 2 

    # 構造 masked 句子
    masked_sentence = context_sentence.replace(word, tokenizer.mask_token, 1)

    # Tokenize 輸入
    inputs = tokenizer(masked_sentence, return_tensors="pt", truncation=True, max_length=CONTEXT_WINDOW_SIZE)
    
    # 找到 [MASK] token 的索引
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    if mask_token_index.numel() == 0:
        # 未找到 mask token，返回高驚奇度
        return MAX_EXPECTED_SURPRISAL * 2
    mask_token_index = mask_token_index[0] # 取第一個 mask token

    with torch.no_grad():
        outputs = lm_model(**inputs)
        predictions = outputs.logits # Logits for each token in the vocabulary

    # 獲取 masked 位置的詞彙分佈
    masked_token_logits = predictions[0, mask_token_index, :]
    
    # 將 logits 轉換為機率
    probabilities = torch.softmax(masked_token_logits, dim=-1)

    # 獲取目標詞彙的 token ID (這裡假設目標詞彙能被單一 token 表示)
    # 實際應用中，如果 word_tokens 是多個 subwords，需要處理每個 subword 的機率
    # 這裡簡化處理，取第一個 subword 的 ID
    target_token_id = tokenizer.convert_tokens_to_ids(word_tokens[0])
    if target_token_id == tokenizer.unk_token_id:
        # 目標詞彙是未知 token，其機率會很低
        return MAX_EXPECTED_SURPRISAL * 2 

    # 獲取目標詞彙的機率
    target_word_probability = probabilities[target_token_id].item()

    # 防止 log(0)
    if target_word_probability <= 1e-10: # 使用一個非常小的正數代替0
        target_word_probability = 1e-10

    # 計算驚奇度
    surprisal = -math.log2(target_word_probability)
    return surprisal
```

#### 步驟 2: 將驚奇度映射到引力半徑 (Map Surprisal to Gravity Radius)

這是一個將驚奇度值線性映射到 UI 視覺範圍內的函數。

```python
def map_surprisal_to_gravity_radius(surprisal: float) -> float:
    """
    將詞彙驚奇度映射到 UI 的動態引力半徑。
    採用線性插值，並進行邊界限制。

    Args:
        surprisal (float): 詞彙的驚奇度值。

    Returns:
        float: 計算出的引力半徑 (像素)。
    """
    # 將驚奇度值限制在預期範圍內
    clamped_surprisal = max(MIN_EXPECTED_SURPRISAL, min(surprisal, MAX_EXPECTED_SURPRISAL))

    # 線性插值
    # 歸一化驚奇度到 [0, 1] 範圍
    normalized_surprisal = (clamped_surprisal - MIN_EXPECTED_SURPRISAL) / \
                           (MAX_EXPECTED_SURPRISAL - MIN_EXPECTED_SURPRISAL)
    
    # 反轉歸一化值，讓高驚奇度對應大半徑，低驚奇度對應小半徑
    # 由於驚奇度與難度正相關，我們希望驚奇度高時，引力半徑大。
    # 所以直接使用 normalized_surprisal
    
    gravity_radius = MIN_GRAVITY_RADIUS_PX + \
                     (MAX_GRAVITY_RADIUS_PX - MIN_GRAVITY_RADIUS_PX) * normalized_surprisal

    # 確保半徑在最小和最大範圍內
    return max(MIN_GRAVITY_RADIUS_PX, min(gravity_radius, MAX_GRAVITY_RADIUS_PX))

```

### 4. 整合到 `IntelligentGaze` 專案流程中

結合文獻中 EyeLingo 的流程 (Fig. 1, Fig. 2)，我們的 `IntelligentGaze` 專案可能會有如下的即時處理邏輯：

```python
# 假設這是即時處理循環的一部分

def process_gaze_and_text_for_gravity_radius(gaze_data_1_sec, current_page_text, lm_model, tokenizer):
    """
    根據即時凝視數據和當前頁面文本，計算每個候選詞彙的動態引力半徑。

    Args:
        gaze_data_1_sec: 過去 1 秒的凝視數據。
        current_page_text: 當前顯示在螢幕上的文本內容。
        lm_model: 預載入的 bert-tiny LM 模型。
        tokenizer: 預載入的 bert-tiny Tokenizer。

    Returns:
        dict: 每個候選詞彙及其對應的動態引力半徑。
              例如: {"word_A": radius_A, "word_B": radius_B}
    """
    
    # 1. 定位感興趣區域 (Region of Interest, ROI)
    # 根據文獻 Section 3.3.4，我們從 1 秒凝視數據中提取一個 bounding box。
    # 這裡簡化為獲取該區域內的所有詞彙。
    # (這部分通常需要具體的 gaze-to-text 映射邏輯，例如基於字體大小和坐標)
    
    # 假設我們有一個函數來從凝視數據中獲取 ROI 內的詞彙
    candidate_words_in_roi = get_words_in_gaze_roi(gaze_data_1_sec, current_page_text)
    
    # 獲取 ROI 周圍的上下文 (文獻 Section 3.1 提到 RoBERTa 應用於 ROI 的所有文本)
    # 為了計算 surprisal，我們需要每個詞彙的上下文句子
    word_contexts = get_context_for_words(candidate_words_in_roi, current_page_text, CONTEXT_WINDOW_SIZE)

    word_gravity_radii = {}

    for word, context_sentence in word_contexts.items():
        # 2. 計算詞彙驚奇度
        surprisal = calculate_word_surprisal(word, context_sentence, lm_model, tokenizer)

        # 3. 映射驚奇度到引力半徑
        gravity_radius = map_surprisal_to_gravity_radius(surprisal)
        
        # 4. 考慮文獻中的其他模態 (Gaze Encoding, Knowledge Embeddings)
        # 文獻中 EyeLingo 最終是結合了 PLM、Gaze Encoding 和 Knowledge Embeddings 
        # 輸入到一個二元分類器。這裡的 surprisal 僅是 PLM 的語言特徵部分。
        # 如果需要更精確地反映「未知詞彙」的潛在難度，可以將這部分作為
        # 最終 `unknown_word_probability` (來自 EyeLingo 的二元分類器輸出) 的一個因子。
        #
        # 舉例來說，如果 EyeLingo 的二元分類器給出 `P(unknown)`：
        # final_difficulty_score = P(unknown) * gravity_radius_from_surprisal_mapping
        # 或者直接將 `P(unknown)` 映射到引力半徑。
        #
        # 但按照問題要求，我們專注於 `P(word | context)` 到 Surprisal 的轉換。
        # 我們假設 `gravity_radius` 直接反映了由語言模型判斷的詞彙「內在驚奇度」。
        # 若需要整合 gaze 和 knowledge，可能要調整 `map_surprisal_to_gravity_radius`
        # 函數，使其也考慮這些因素，例如將 `surprisal` 替換為一個綜合難度分數。

        word_gravity_radii[word] = gravity_radius

    return word_gravity_radii

# --- 輔助函數 (需要根據實際 UI 和文本解析邏輯實現) ---
def get_words_in_gaze_roi(gaze_data, page_text):
    """
    根據凝視數據和頁面文本，返回 ROI 中的詞彙列表。
    這是一個佔位符，需要實際的 gaze-to-text 映射實現。
    """
    # 實際實現會解析 gaze_data (x, y 坐標，時間戳) 並將其映射到 page_text 的詞彙 bounding box。
    # 參考文獻 Section 3.3.4 的 bounding box 邏輯。
    # 這裡為了演示，返回一個硬編碼的詞彙列表
    return ["unknown", "word", "detection", "gaze", "model"]

def get_context_for_words(words_in_roi, full_text, window_size):
    """
    為 ROI 中的每個詞彙提取其上下文句子。
    這是一個佔位符，需要實際的文本解析和語境提取實現。
    """
    word_contexts = {}
    for word in words_in_roi:
        # 假設 full_text 是一個長字符串，找到 word 的位置，提取前後 window_size/2 的 tokens。
        # 為了簡化，這裡返回整個 full_text 作為上下文。
        # 實際應用中，應考慮句子邊界。
        word_contexts[word] = full_text 
    return word_contexts
```

### 5. 實現考量與增強

*   **`bert-tiny` 的 tokenization**：`bert-tiny` 和大多數 BERT 變體使用 WordPiece 分詞。一個單詞可能被分成多個 subwords。`calculate_word_surprisal` 函數中簡化地只取第一個 subword 的機率。更精確的做法是計算所有 subwords 的聯合機率或平均機率。
*   **上下文窗口**：文獻中提到 `RoBERTa` 應用於「感興趣區域」(Region of Interest, ROI) 中的文本。在 `calculate_word_surprisal` 中，`context_sentence` 應限制在 `CONTEXT_WINDOW_SIZE` 內，以符合 `bert-tiny` 的輸入限制並聚焦局部上下文。
*   **驚奇度校準**：`MIN_EXPECTED_SURPRISAL` 和 `MAX_EXPECTED_SURPRISAL` 需要通過在代表性語料庫上運行 `bert-tiny` 並分析其驚奇度分佈來進行經驗性校準。例如，可以計算大量詞彙的驚奇度，然後取其 5th 百分位數和 95th 百分位數作為 `MIN_EXPECTED_SURPRISAL` 和 `MAX_EXPECTED_SURPRISAL`。
*   **使用者個性化**：文獻強調「未知詞彙因使用者而異」。引力半徑的映射函數 (例如 `MIN_GRAVITY_RADIUS_PX`, `MAX_GRAVITY_RADIUS_PX` 或驚奇度的上下限) 可以根據每個使用者的語言熟練度、VLT 測驗分數或過去的互動行為進行動態調整，實現個性化輔助。
*   **多模態整合**：本偽程式碼專注於語言模型。文獻中的 EyeLingo 結合了 gaze encoding 和 word-level knowledge embedding。一個更完整的系統可以設計一個綜合難度分數，該分數是 surprisal、gaze 特徵（如凝視時長、回溯次數）和詞彙知識（如詞頻、詞性）的加權組合，然後將這個綜合分數映射到引力半徑。
*   **UI 平滑性**：引力半徑的變化可能會在 UI 上顯得突兀。可以應用一個移動平均濾波器或指數衰減來平滑 `gravity_radius` 的變化，以改善使用者體驗。
*   **即時性**：`bert-tiny` 相比大型模型，推斷速度更快，但即時計算每個 ROI 中所有詞彙的 surprisal 仍可能帶來延遲。需要優化批處理和模型部署策略 (如 ONNX)。文獻中提到 EyeLingo 的延遲在 1 秒內。

---

## Constraints & Edge Cases (軟硬體限制與極端狀況)
作為 text_model 領域的專家，我將根據所提供的文獻，針對您的問題進行詳細解答。

---

### 1. 移除 PLM 後 F1-score 的災難性下降

根據論文第 4.4 節「Contribution of PLM」及表 3「Ablation study with eye tracker collected gaze data」的消融實驗結果，可以觀察到當移除預訓練語言模型（PLM，在此指 RoBERTa）時，模型的 F1-score 會發生「災難性的下降」。

*   **完整模型 (Ours (main model))**：F1-score 為 **71.1%**。
*   **移除文本編碼 (w/o textual encoding)**：這表示移除了預訓練語言模型 (RoBERTa) 對文本信息的捕捉。此時 F1-score 驟降至 **16.2%**。

這是一個從 **71.1% 下降到 16.2%** 的巨大跌幅（下降了約 77.2%）。這清晰地表明了 PLM 在未知詞檢測任務中的核心且不可或缺的作用。如果僅移除預訓練的 RoBERTa 權重（即 `w/o pretrained RoBERTa`，F1-score 為 67.6%），雖然也有所下降，但幅度遠不及完全移除 PLM，這說明即使是從頭開始訓練，RoBERTa 的架構仍能捕捉部分語言特徵，但預訓練的知識才是性能飛躍的關鍵。

### 2. 神經符號 (Neuro-Symbolic) 的認知補償比單純升級鏡頭硬體更有效

論文的發現強烈支持了在邊緣運算 (Edge AI) 設備中，採用「神經符號 (Neuro-Symbolic)」的認知補償方法比單純升級鏡頭硬體更為有效。

#### 2.1 鏡頭硬體的根本限制 (以 Gaze 數據為例)

論文第 6.3 節「Limitation and Future Works」明確指出，即使是專業的眼動追蹤儀，其準確性對於詞彙級別的檢測也存在根本限制：

*   **準確性不足**：
    *   普通閱讀條件下（單行間距，10 號字體），行高約為 3-5 毫米。
    *   在 50-60 厘米的閱讀距離下，這需要約 $0.3^\circ - 0.6^\circ$ 的視線準確度。
    *   然而，大多數眼動追蹤儀的注視準確度範圍在 $0.2^\circ - 1.1^\circ$ 之間。
    *   再加上頭部和上半身移動帶來的額外誤差，使得單純依賴注視坐標難以精確定位用戶正在關注的特定詞彙。
    *   實驗中甚至有參與者反映，校準後誤差仍可達 1-3 行。

*   **噪音的普遍性**：
    *   即使是專業眼動儀數據也受用戶姿態變化的影響（如長時間閱讀導致的注視漂移）。
    *   邊緣設備上通常使用的攝像頭（如論文中提到的 webcam）數據噪音更大、採樣率不穩定，且更容易受環境光線和用戶動作影響。論文在第 4.2 節中指出，儘管 webcam 數據噪音更大（x 方向 MAE 約 116 像素，y 方向 MAE 約 67 像素，遠超一行文本高度 20 像素），但 EyeLingo 在其上仍能達到 65.1% 的 F1-score，這證明了其對噪音的魯棒性。

**結論**：單純升級鏡頭硬體，例如從一個中等水平的攝像頭升級到一個稍微好一點的攝像頭，可能只能帶來微小的、邊際的準確性提升，而無法根本解決上述的物理限制和實際使用中的噪音問題。它仍然無法保證詞彙級別的精確度，尤其是在單行間距等緊湊排版下。

#### 2.2 神經符號的認知補償機制

EyeLingo 模型正是「神經符號」認知補償的典型範例，它通過結合不同模態（gaze 和 text）的信息，並利用兩種不同的智能處理方式（神經網絡的模式識別與符號知識的集成）來克服單一模態或單一方法的局限性：

1.  **「神經」組件 – 預訓練語言模型 (PLM) 的核心作用：**
    *   **捕捉豐富的語言特徵 (Textual Information Capturing)**：PLM（如 RoBERTa）具有強大的能力來捕捉詞彙的上下文語義、句法結構和詞彙頻率等語言學特徵。這些特徵對於判斷一個詞的難度至關重要（第 3.1 節）。
    *   **彌補 Gaze 的不準確性**：正如第 6.1 節所述，「PLM 彌補了這個問題，通過基於文本提供語言信息。」當 gaze 無法精確指向某個詞時，PLM 能夠利用周圍文本的語義信息來「猜測」或「理解」用戶可能遇到的未知詞。這是一種「認知補償」——用語言智能來彌補感知（gaze）數據的不足。
    *   **主要性能貢獻者**：PLM 提供的語言學信息是提升模型性能的「主要角色」（第 4.4 節）。F1-score 從 71.1% 降至 16.2% 的實驗結果是其核心地位的鐵證。

2.  **「符號」組件 – 知識嵌入 (Knowledge-grounded Enhancement)：**
    *   **集成詞彙級知識**：除了 PLM 捕捉的上下文語義外，EyeLingo 還加入了詞彙級的先驗知識嵌入，如詞頻 (term frequency)、詞性 (part-of-speech, POS) 和命名實體識別 (named entity recognition, NER)（第 3.2.3 節）。
    *   **補償詞彙級信息損失**：這些「符號化」的語言學知識彌補了 PLM 在 tokenization 過程中可能丟失的詞彙級信息，進一步提升了模型的準確性（第 3.1 節）。這也是一種明確的認知補償，將結構化的、人類可理解的語言學規則集成到模型中。

3.  **「神經」組件 – Gaze 編碼器 (Positional Data Encoding) 的輔助作用：**
    *   **實時區域定位**：儘管 gaze 準確性有限，但它對於「實時定位感興趣區域 (Region of Interest, RoI)」至關重要，這是實現實時應用（如即時詞彙解釋）的基礎（第 3.1 節、第 6.1 節）。
    *   **用戶依賴性**：Gaze 數據也提供了用戶特定的閱讀行為模式信息，有助於識別不同用戶的未知詞彙，因為未知詞彙因人而異（第 4.3 節）。雖然其對 F1-score 的提升不如 PLM 顯著（移除 gaze 編碼後，F1-score 從 71.1% 降至 68.5%），但它提供了個性化和實時性的關鍵要素。

#### 結論：

在邊緣運算設備中，由於成本、功耗和實際部署環境的限制，難以期望獲得研究級別的傳感器硬件和數據質量。在這種情況下，一個能夠利用多模態信息（低質量的 gaze + 文本），並通過「智能」算法（尤其是強大的 PLM 進行語言理解和符號知識進行補充）進行「認知補償」的 Neuro-Symbolic 方法，遠比盲目追求傳感器硬件的邊際提升更具成本效益和實際效果。

EyeLingo 的成功（在 webcam 數據上仍有 65.1% F1-score）證明了其魯棒性和實用性。它以語言智能（Neuro-Symbolic）來填補感知數據（Gaze）的空缺，使得在低成本、低功耗的 Edge AI 設備上實現高性能的未知詞檢測成為可能。

## Example Q&A (Vibe Coding 指引與範例)
作為 text_model 領域的專家，我將根據您提供的文獻《2502.10378v1.pdf - Unknown Word Detection for English as a Second Language (ESL) Learners Using Gaze and Pre-trained Language Models》，設計兩個 Vibe Coding 提問。這份文獻強調了預訓練語言模型 (PLM) 在捕捉語言特徵和詞彙難度方面的關鍵作用，以及即時、輕量級解決方案的重要性。

我們將引導學生利用 `transformers` 函式庫，以 "Surprisal Score" 作為衡量詞彙難度（文獻中提到的 "linguistic characteristics related to word difficulty"）的指標，並將結果輸出為前端可用的 JSON 格式。

---

### **背景知識回顧**

在文獻中，作者提出使用預訓練語言模型 (PLM) 如 RoBERTa 來捕捉豐富的語言資訊，以輔助未知詞彙檢測。其中一個重要的語言學概念是 **Surprisal (驚訝度)**，它衡量了在給定上下文下，一個詞彙出現的預期程度。Surprisal 越高，表示該詞彙在當前上下文中越不具預期性，也可能對讀者（特別是 ESL 學習者）而言越陌生或困難。

對於因果語言模型 (Causal Language Models)，一個詞彙 $w_i$ 的 Surprisal 通常定義為其在所有先前詞彙 $w_1, ..., w_{i-1}$ 下的負對數機率：
$S(w_i) = -\log_2 P(w_i | w_1, ..., w_{i-1})$

對於像 RoBERTa 這類 Masked Language Models (MLM)，它在訓練時會隨機遮蔽部分 token 並預測這些被遮蔽的 token。我們可以使用類似的方法來估計 Surprisal：逐一遮蔽句子中的每個詞彙（或其構成的 token），讓模型預測被遮蔽的詞彙，然後計算其負對數機率。這可以近似地反映該詞彙在完整上下文中的「不確定性」或「驚訝度」。文獻中也提及了效率考量，因此選擇一個輕量級的模型是符合其精神的。

---

### **Vibe Coding 提問**

#### **提問一：計算單詞的 Surprisal Score**

**目標：** 載入一個輕量級的預訓練語言模型，並計算輸入句中每個單詞的 Surprisal Score。

**任務描述：**

請撰寫一個 Python 函式 `calculate_word_surprisals`，它接受一個英文句子作為輸入，並使用 `transformers` 函式庫載入一個輕量級的 Masked Language Model（例如：`roberta-base` 或 `distilbert-base-uncased`）來計算句子中每個**有意義**單詞的 Surprisal Score。

**具體要求：**

1.  **模型載入：** 使用 `AutoTokenizer` 和 `AutoModelForMaskedLM` 載入 tokenizer 和模型。建議使用 `roberta-base`。
2.  **單詞遮蔽策略：** 對於句子中的每個單詞 $w_i$：
    *   識別該單詞在 tokenizer 後對應的所有 subword tokens。
    *   將這些 subword tokens 在輸入序列中替換為 `tokenizer.mask_token`。
    *   將修改後的序列輸入到模型中進行預測。
    *   提取模型在被遮蔽位置上對原始 subword tokens 的 log-probability。
    *   計算該單詞的 Surprisal Score 為其所有對應 subword tokens 的負對數機率的平均值。
        *   **公式：** 對於單詞 $w_i$ 及其對應的 subword tokens $t_1, t_2, ..., t_k$：
            $S(w_i) = \text{Average}_{j=1}^k (-\log_2 P(t_j | \text{context with } t_j \text{ masked}))$
            其中 $P(t_j | \text{context with } t_j \text{ masked})$ 可以從模型的 `logits` 輸出經過 `softmax` 轉換並取對應 token ID 的機率獲得。
3.  **輸出格式：** 函式應返回一個列表，其中每個元素是一個字典，包含 `{'word': str, 'surprisal': float}`。
4.  **過濾標點符號與功能詞：** 在計算 Surprisal Score 時，請忽略標點符號以及常見的功能詞 (如 "a", "an", "the", "is", "are" 等)。你可以建立一個簡單的功能詞列表進行過濾。

**函式簽名：**

```python
from typing import List, Dict

def calculate_word_surprisals(sentence: str) -> List[Dict[str, float]]:
    """
    Calculates the surprisal score for each meaningful word in a given English sentence
    using a pre-trained Masked Language Model.

    Args:
        sentence: The input English sentence.

    Returns:
        A list of dictionaries, where each dictionary contains a word and its surprisal score.
        Example: [{'word': 'example', 'surprisal': 8.5}, ...]
    """
    pass
```

**提示：**

*   `tokenizer.encode` 可以將句子轉換為 token ID 序列，`tokenizer.convert_ids_to_tokens` 可以將 ID 轉換回 token 字符串。
*   `torch.nn.functional.log_softmax` 可以方便地從 `logits` 計算 log-probabilities。
*   對於 RoBERTa，請注意其 tokenization 可能會在單詞前添加 `Ġ` 符號，表示單詞的開頭。在處理時需要考慮這一點。

---

#### **提問二：整合詞彙資訊並匯出 JSON 格式**

**目標：** 擴充 Surprisal Score 的結果，加入文獻中提及的其他語言學特徵，並將最終結果匯出為 JSON 格式，供前端應用使用。

**任務描述：**

請撰寫一個 Python 函式 `generate_frontend_data`，它接受一個英文句子和 `calculate_word_surprisals` 函式返回的 Surprisal 列表作為輸入。該函式需要進一步處理每個單詞，增加其詞性 (Part-of-Speech, POS) 標籤和一個模擬的詞頻 (Term Frequency) 分數，最終將所有資訊整合並匯出為 JSON 格式的字符串。

**具體要求：**

1.  **輸入：**
    *   `sentence`: 原始輸入的英文句子。
    *   `surprisal_data`: 來自 `calculate_word_surprisals` 函式的輸出（`List[Dict[str, float]]`）。
2.  **詞性標註 (POS Tagging)：**
    *   使用 `spaCy` 函式庫來對原始句子進行詞性標註。
    *   對於 `surprisal_data` 中的每個單詞，查詢其對應的 POS 標籤。
3.  **模擬詞頻 (Simulated Term Frequency)：**
    *   文獻中提到了詞頻 (Term Frequency) 是衡量詞彙難度的一個指標。為了簡化，你可以創建一個簡單的字典來模擬詞頻數據，或者直接為每個單詞分配一個隨機的「頻率分數」（例如 1 到 100 之間）。
    *   為每個單詞增加 `frequency_score` 欄位。
4.  **整合與輸出：**
    *   將 Surprisal Score、POS 標籤和模擬詞頻分數整合到每個單詞的字典中。
    *   最終的輸出應該是一個 JSON 格式的字符串，其根元素是一個列表，包含所有處理後的單詞資訊。

**函式簽名：**

```python
import json
from typing import List, Dict

def generate_frontend_data(sentence: str, surprisal_data: List[Dict[str, float]]) -> str:
    """
    Integrates surprisal scores with additional linguistic features (POS tag, simulated frequency)
    for each word in a sentence and exports the data as a JSON string for frontend use.

    Args:
        sentence: The original input English sentence.
        surprisal_data: A list of dictionaries, where each dictionary contains a word and its surprisal score.
                        Example: [{'word': 'example', 'surprisal': 8.5}, ...]

    Returns:
        A JSON formatted string containing a list of word objects with their features.
        Example: '[{"word": "example", "surprisal": 8.5, "pos_tag": "NOUN", "frequency_score": 50}, ...]'
    """
    pass
```

**提示：**

*   `spaCy` 的基本用法：
    ```python
    import spacy
    nlp = spacy.load("en_core_web_sm") # 需要先安裝：python -m spacy download en_core_web_sm
    doc = nlp(sentence)
    for token in doc:
        print(token.text, token.pos_)
    ```
*   處理單詞匹配：`surprisal_data` 中的單詞可能與 `spaCy` 的 tokenization 略有不同。請確保能正確匹配。一個簡單的方法是，如果 `spaCy` 的 `token.text` 存在於 `surprisal_data` 的 `word` 欄位中，則進行匹配。
*   使用 `json.dumps()` 函式將 Python 列表或字典轉換為 JSON 字符串。

---