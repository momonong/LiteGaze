# 🧠 從文獻到程式碼：Vibe Coding 與 Agent Skill 實作揭秘

在 IntelligentGaze 專案中，我們不僅僅是在寫 Python 程式碼，我們更是在實踐一種全新的軟體開發典範：**Vibe Coding (氛圍編程)** 與 **Agent Skill Implementation (代理人技能實作)**。

這份文件將為您深度解析 `knowledge/` 目錄下的 `/papers` 與 `questions.yaml` 之間的共生關係，以及這個設計背後的強大邏輯與管線 (Pipeline)。

---

## 1. 知識轉譯管線 (The Knowledge Translation Pipeline)

在我們的專案架構中，每個知識模組（例如 `text_model` 或 `calibration`）底下都有兩個核心元素：`/papers` 目錄與 `questions.yaml` 檔案。它們之間的關係，就像是**「原物料」**與**「提煉配方」**。

### 📄 `/papers` (原物料：未加工的領域知識)
這個目錄存放了該模組相關的所有學術論文（PDF）、技術文件或 Markdown 筆記。
- **本質**：它是龐大、無結構、且充滿艱澀學術術語的原始資料庫。
- **痛點**：如果你直接把 10 篇 PDF 塞給一個寫程式的 AI (例如 GitHub Copilot 或 Cursor) 說「幫我寫程式」，AI 會因為資訊量過大、重點失焦，而產生幻覺 (Hallucination) 或寫出不符合學理基礎的程式碼。

### ⚙️ `questions.yaml` (提煉配方：精準的萃取指令)
這個 YAML 檔案定義了我們「希望 AI 從這些龐雜論文中提取出什麼具體的知識」。
- **本質**：它是一系列精心設計的 Prompt (提示詞)，負責對海量文獻進行聚焦。
- **結構**：它以鍵值對 (Key-Value) 的形式存在，定義了輸出的章節。例如：
  - `concepts`: "請列出論文中提到的核心演算法參數與名詞解釋..."
  - `methodology`: "請根據論文詳細解釋 Cursor-Guided Reading 的實作邏輯與數學公式..."

### 🔄 運作邏輯 (The Pipeline)
當我們執行 `skill_builder.py` 時，系統會進行以下「知識提煉」的自動化流程：
1. **聚合 (Aggregation)**：腳本會將 `/papers` 裡的所有 PDF、文字檔全部讀取出來，首尾相連合併成一個巨大的「超級文本」。
2. **詢問 (Interrogation)**：腳本會遍歷 `questions.yaml` 裡的每一個問題。它會將「超級文本」作為背景知識（Context），並將問題作為指令（Instruction），透過非同步的方式發送給 Gemini 模型。
3. **萃取與重組 (Extraction & Assembly)**：Gemini 會在海量文獻中精準尋找答案，過濾掉無關的學術廢話，並將結果輸出為精確的 Markdown 格式。最後，腳本會將這些答案填入 `default_template.md` 的佔位符中，組合出最終的 `skill_<module>.md` 檔案。

---

## 2. 為什麼這被稱為「Skill Implementation (技能實作)」？

在傳統的軟體工程中，「實作一項功能」意味著開發者親手寫下數百行的 IF-ELSE 邏輯。而在 AI 代理人 (AI Agent) 的時代，「實作一項技能」意味著**建構一份高質量的 System Prompt 或知識庫**。

- **賦予 AI 專業靈魂**：預設的 ChatGPT 或 Claude 就像是一個「什麼都懂一點的通才」。但我們的專案需要的是一個「深諳神經符號學與眼動科學的頂尖專家」。
- **Skill.md 就是 AI 的大腦擴充包**：我們透過 `skill_builder.py` 產出的 `skill_text_model.md`，本質上就是一份**「專家技能書 (Agent Skill)」**。
- **技能的裝載**：當我們把這份彙整好的 Markdown 餵給另一個 AI（例如在使用 Cursor 編輯器時作為 Context，或是餵給我們自己寫的 Diagnostic Agent）時，這個 AI 就瞬間「學會」了這項技能。它會深刻理解什麼是 Word Surprisal、知道為何需要 L2 正則化，並且能基於這些「極度專業的領域知識」來幫我們寫出符合科學原理的程式碼，或是分析學生的眼動軌跡。

我們將人類的學術知識，系統化地轉換為了機器可讀、可執行的「技能模組」。這就是 AI 時代的 Skill Implementation。

---

## 3. 為什麼這被稱為「Vibe Coding (語意/氛圍編程)」？

**Vibe Coding** 是一個新興的開發術語，指的是**「透過塑造上下文氛圍與提供自然語言的知識脈絡，引導 AI 自主生成正確的程式碼」**的過程，而不是開發者自己一行一行地敲打 Python 語法。

`/papers` 與 `questions.yaml` 的共生設計，正是 Vibe Coding 最極致的體現：

1. **從「命令機器」到「指導專家」**：我們並沒有在原始碼裡硬寫 `gravity_radius` 應該怎麼算。相反地，我們在 `questions.yaml` 中要求 AI 去論文裡找出計算驚奇度的科學邏輯。
2. **塑造 Vibe (氛圍與脈絡)**：我們把完整的學術論文 (`/papers`) 作為 Context 丟給 AI。我們不是在給 AI 一個空洞的「寫一個迴圈」的任務，而是給了它一整個「認知科學實驗室的學術氛圍」。在這種充滿領域知識的 Vibe 中，AI 寫出來的程式碼（例如我們生成的 `colab_neuro_symbolic.py`）會自帶學術嚴謹性、變數命名會符合論文慣例，邏輯也會深具科學深度。
3. **聲明式知識開發 (Declarative Knowledge Development)**：開發者只需要在 `questions.yaml` 中「聲明」我們需要什麼知識（What），而不需要去管底層的 PDF 解析或 NLP 萃取是怎麼做的（How）。當學術界有新的論文發表時，我們只要把新的 PDF 丟進 `/papers` 並重新跑一次腳本，AI 就會根據新的論文「氛圍」重新整理出技能文件，進而引導 AI 寫出升級版的演算法程式碼。

### 💡 總結

這個由 `papers/` -> `questions.yaml` -> `skill.md` -> `Generated Code` 構成的管線，將**學術研究、提示詞工程 (Prompt Engineering) 與軟體開發**完美融合。這代表著未來工程師的職責正在轉變：我們不再只是「程式碼的打字員」，而是「知識的策展人」與「AI 系統的架構師」。
