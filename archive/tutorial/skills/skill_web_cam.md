# Agent Skill: IntelligentGaze - {module_name} 模組開發指南

## Metadata
- **領域**：邊緣運算與神經符號 AI
- **適用對象**：Cursor Composer / Vibe Coding 教學助理

## Overview (模組核心目標)
您好，作為 web_cam 領域的專家，我將根據您提供的文獻，精確闡述 WebGazer.js 的 baseline GIGO 問題，以及文獻提出的三個核心解決架構。

---

## Baseline WebGazer 的 GIGO (Garbage In, Garbage Out) 問題

在非受限的瀏覽器環境中部署基於網路攝影機的眼動追蹤，由於以下幾個根本性限制，導致 WebGazer.js 的基礎實作極易產生 **"Garbage In, Garbage Out" (GIGO)** 的問題：

1.  **數據污染的來源不精確：** Baseline WebGazer 依賴 2D 臉部特徵提取和連續的數學回歸來映射用戶的視線點 (Point of Regard, POR)。如果這些回歸訓練數據來自與用戶實際 POR 不一致的互動（例如，用戶眼睛看著螢幕右側，但鼠標停留在左側），回歸模型就會被污染，導致收集到的視線坐標無效。
2.  **追蹤器本身的局限性：** 早期或基準實作常使用 `clmtrackr` 或 HAAR cascade 等舊型追蹤模型，這些模型在複雜的頭部姿勢、微表情和次優光照條件下難以維持穩定的臉部特徵鎖定，導致輸入數據從源頭就質量低下。
3.  **歷史數據的污染：** WebGazer 預設會利用 `localforage` (IndexedDB) 跨瀏覽器會話保存訓練數據。如果用戶在不同的環境（椅子、光線、距離）下重新使用應用程式，舊的回歸權重會與當前的物理幾何 actively 衝突，導致災難性的追蹤漂移，這也是典型的 GIGO 問題。
4.  **光學盲點與生理限制：** 網路攝影機固有的鏡頭邊緣桶狀畸變，以及用戶向下看螢幕底部時，上眼瞼遮擋瞳孔和角膜光反射 (Purkinje image) 變形，都會導致該區域的數據被嚴重破壞，若允許內容進入這些區域，則會向模型輸入「垃圾數據」。

這些問題使得 WebGazer.js 在沒有經過優化配置的情況下，難以作為高擬真數據收集工具。

---

## 3 個核心解決架構

文獻提出透過實施以下三個核心架構，從根本上解決 GIGO 問題，將基礎 WebGazer 部署轉化為高精度數據收集工具：

### 1. 引擎架構升級：過渡到 `TFFacemesh` Tensor

此步驟旨在解決 WebGazer 基礎實作中因**追蹤器本身局限性**而導致的 GIGO 問題。

*   **問題診斷：** 舊的追蹤模型 (`clmtrackr`, HAAR cascades) 在複雜頭部姿勢、微表情、次優光照下表現不佳，無法提供高質量輸入。
*   **解決方案：** 整合 `TensorFlow.js (TFJS)` 和 `MediaPipe's Facemesh` 拓撲（在 WebGazer API 中稱為 `TFFacemesh` 追蹤器）。
*   **具體參數與邏輯：**
    *   **核心特性：** 提供一個**密集的 468 點 3D 臉部幾何圖**，即使在輕微的頭部平移或可變環境光照下，也能精確提取高擬真度的眼部區域（鞏膜、虹膜、瞳孔邊界）。
    *   **初始化配置：** 必須顯式將追蹤器設置為 `"TFFacemesh"`。
    *   **回歸模型：** 必須顯式將回歸模型設置為 `"ridge"` 或 `"threadedRidge"`，以高效處理由此產生的高維特徵向量。
        *   **`ridge` 回歸：** 推薦用於穩定、實時預測。它採用 **L2 正則化** (`L2 regularization`) 來防止模型對網路攝影機流中固有的噪聲輸入過度擬合。L2 正則化透過縮小權重係數來防止極端權重值，從而避免因異常光反射導致權重矩陣災難性峰值，確保預測光標穩定。
        *   **`threadedRidge`：** 將 Ridge 回歸的矩陣計算分配給 Web Workers 異步執行，減少主線程阻塞，適用於高頻採樣環境。
    *   **GIGO 預防：** 為了防止來自之前未校準會話的數據污染，**必須**調用 `window.webgazer.clearData()` 來清除本地持久化的訓練數據。這確保 Ridge 模型以一個原始、中性的權重矩陣開始。

```javascript
document.addEventListener("DOMContentLoaded", async () => {
    if (typeof window.webgazer === 'undefined') {
        console.error("Critical Error: WebGazer.js library is not loaded.");
        return;
    }
    try {
        // 防止 GIGO：清除所有持久化的訓練數據，確保模型從零開始學習
        window.webgazer.clearData(); 

        await window.webgazer
           .setTracker("TFFacemesh") // 顯式設定為 TFFacemesh 追蹤器
           .setRegression("ridge")   // 顯式設定為 Ridge 回歸模型，使用 L2 正則化
           .setGazeListener((data, elapsedTime) => {
                if (data == null) return;
                const gazeEvent = new CustomEvent('gazeUpdate', {
                    detail: { x: data.x, y: data.y, time: elapsedTime }
                });
                window.dispatchEvent(gazeEvent);
            })
           .begin(); 

        window.webgazer.showVideoPreview(true)
           .showPredictionPoints(false)
           .showFaceOverlay(true)
           .showFaceFeedbackBox(true);
        
        console.log("WebGazer engine successfully initialized with TFFacemesh and Ridge Regression.");
    } catch (error) {
        console.error("WebGazer Initialization Failed:", error);
    }
});
```

### 2. 隱式校準：實施 9 點網格「解鎖」協議

此步驟旨在解決 WebGazer 中因**模型未經適當初始化校準**而導致的 GIGO 問題。

*   **問題診斷：** 眼動追蹤模型本質上需要個人校準。用戶通常會忽略或錯誤地執行可選的校準步驟，導致數學模型無法穩健建立。如果模型僅在螢幕中心區域校準，則在用戶看向邊緣時會嚴重失準。
*   **解決方案：** 將 9 點校準序列作為強制性的 UI「解鎖」閘道器。這迫使用戶在與主要內容互動之前，提供數學上分佈均勻的基準回歸錨點。
*   **具體參數與邏輯：**
    *   **心理物理學原理：** 9 點網格通常位於螢幕水平和垂直觀看區域的 **10%、50% 和 90%** 處。這迫使用戶注視極端角落和中心，從而建立用戶眼動的絕對最大和最小範圍。Ridge 回歸算法利用這些外圍錨點來內插內部螢幕空間。
    *   **校準參數：**
        *   **網格坐標：** `10vw`, `50vw`, `90vw` (水平) 和 `10vh`, `50vh`, `90vh` (垂直)。這確保校準發生在視窗邊界附近，且不觸發邊緣光學畸變。
        *   **每點點擊次數：** `5` 次。多次點擊要求延長注視時間，捕獲更多網路攝影機幀的變化，為該特定螢幕坐標建立更穩健的統計平均值。
        *   **背景顏色：** 高不透明度 (例如 `#0f172a` 的 `rgba(15, 23, 42, 0.98)`)。消除視覺干擾，防止關鍵校準階段非自主的眼跳。
        *   **點大小：** `30px` 直徑。確保易於點擊 (Fitts's Law)，同時足夠小以強制形成緊湊、高度特定的視覺注視區域。
    *   **核心 API 調用：** 每點擊一次，呼叫 `window.webgazer.recordScreenPosition(e.clientX, e.clientY, 'click')`。這會捕獲當前的網路攝影機幀，提取眼部特徵，並將這些像素排列數學地映射到點擊的 `(e.clientX, e.clientY)` 向量。

```javascript
class CalibrationProtocol {
    constructor() {
        // ... (其他初始化設置)
        this.clicksRequiredPerPoint = 5; 
        this.points = [ // 標準 9 點網格，使用視窗百分比 (vw, vh)
            { x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 },
            { x: 10, y: 50 }, { x: 50, y: 50 }, { x: 90, y: 50 },
            { x: 10, y: 90 }, { x: 50, y: 90 }, { x: 90, y: 90 }
        ];
    }

    renderNextPoint() {
        // ... (生成校準點的 DOM 元素)
        pointElement.addEventListener('click', (e) => {
            clickCount++;
            pointElement.innerText = this.clicksRequiredPerPoint - clickCount;

            // 核心 API 調用：強制 WebGazer 註冊此精確點擊作為訓練事件
            window.webgazer.recordScreenPosition(e.clientX, e.clientY, 'click');

            if (clickCount >= this.clicksRequiredPerPoint) {
                // ... (點擊完成後的狀態更新和進度推進)
            }
        });
        this.container.appendChild(pointElement);
    }

    unlockApplication() {
        // ... (完成校準後隱藏校準門和顯示應用程序)
        console.log("Implicit Calibration Protocol Complete. Regression Model Anchored. App Unlocked.");
    }
}
```

### 3. 連續訓練：「光標引導式閱讀」UX 啟發法

此步驟旨在解決 WebGazer 中因**鼠標運動與視線脫鉤導致模型長期污染和漂移**的 GIGO 問題。

*   **問題診斷：** WebGazer 預設監聽瀏覽器的 `mousemove` 和 `click` 事件進行連續訓練。但在標準閱讀應用中，鼠標通常與用戶視線分離（例如，鼠標靜止在左邊緣，眼睛掃描右側文本）。這種「運動路徑」與「視覺路徑」的錯位會「毒害」Ridge 回歸矩陣，導致長期準確性嚴重下降和模型漂移。
*   **解決方案：** 實施「光標引導式閱讀」介面。透過 UI 限制，強制用戶用光標追蹤文本以進行閱讀。這在用戶的中心凹視線點 (POR) 和鼠標的物理 `(x, y)` 坐標之間建立了**不可避免的生物力學鎖定**。
*   **具體參數與邏輯：**
    *   **原理：** 當用戶視線與光標嚴格綁定時，鼠標的每一次移動都成為一個高度準確、完美對齊的訓練向量，直接輸入 WebGazer 引擎。這將原本的**模型污染源**（閒置鼠標移動）轉變為**高擬真校準數據的連續流**。
    *   **UI 實現：「手電筒」文本遮罩技術。**
        *   主要閱讀文本透過 CSS 濾鏡完全模糊 (`filter: blur(10px); opacity: 0.3;`)。
        *   一個清晰可讀的「鏡頭」（或聚光燈）僅在鼠標光標的即時幾何半徑下生成 (`-webkit-mask-image: radial-gradient(...)`)。
        *   這迫使用戶必須物理拖動「鏡頭」來閱讀文本。
    *   **核心 API 調用：** 在 `readingZone` 元素上監聽 `mousemove` 事件，並**顯式**將當前鼠標坐標 `(e.clientX, e.clientY)` 餵給 WebGazer 進行訓練。
        *   `readingZone.addEventListener('mousemove', (e) => { ... });`：確保僅在閱讀區域內進行訓練。
        *   `window.webgazer.recordScreenPosition(e.clientX, e.clientY, 'move');`：因為用戶必須精確地看向鼠標移動的位置才能閱讀文本，這保證了物理眼部特徵與 `(e.clientX, e.clientY)` 向量的對齊。

```javascript
document.addEventListener("DOMContentLoaded", () => {
    const readingZone = document.getElementById('guided-reading-zone');
    const readingLens = document.getElementById('reading-lens');
    
    if (!readingZone ||!readingLens) return;

    readingZone.addEventListener('mousemove', (e) => {
        const rect = readingZone.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // 更新 CSS 徑向漸變遮罩位置以跟隨光標，創建手電筒效果
        readingLens.style.webkitMaskImage = `radial-gradient(circle 80px at ${x}px ${y}px, black 40%, rgba(0,0,0,0.8) 70%, transparent 100%)`;
        readingLens.style.maskImage = `radial-gradient(circle 80px at ${x}px ${y}px, black 40%, rgba(0,0,0,0.8) 70%, transparent 100%)`;

        // 顯式核心 API 調用：將全局視窗坐標餵給 WebGazer 以持續訓練模型
        // 確保用戶視線與鼠標位置嚴格對齊
        if (window.webgazer && window.webgazer.isReady()) {
            window.webgazer.recordScreenPosition(e.clientX, e.clientY, 'move');
        }
    });

    readingZone.addEventListener('mouseleave', () => {
        readingLens.style.webkitMaskImage = `none`;
        readingLens.style.maskImage = `none`;
    });
});
```

---

這些優化措施從底層引擎、初期校準到持續訓練策略全面解決了 WebGazer.js 在非受限環境下產生的 GIGO 問題，顯著提升了其數據收集的精度和穩定性。

## Core Concepts (關鍵演算法與理論)
作為Web_Cam領域的專家，根據您提供的文獻《WEBGAZER.md》，以下是關於TFFacemesh tracker與ridge regression的必須性，以及不使用L2 regularization的後果的精確解釋：

---

### 1. 為什麼必須明確設定 `TFFacemesh` tracker 與 `ridge` regression？

明確設定 `TFFacemesh` tracker 與 `ridge` regression 是WebGazer.js從「Garbage In, Garbage Out (GIGO)」情境轉變為「穩健、高保真數據採集儀器」的**基礎架構優化步驟**。這兩種設定共同解決了傳統瀏覽器眼動追蹤在數據保真度、空間準確性和時間解析度上的根本限制。

#### 1.1. 必須明確設定 `TFFacemesh` Tracker 的原因：

*   **解決舊追蹤模型的缺陷 (GIGO mitigation)**：文獻指出，WebGazer的早期迭代和基礎實現常使用舊的追蹤模型（如`clmtrackr`或基於HAAR cascade的架構），這些模型在面對複雜的頭部姿勢、微表情和次優照明條件時難以維持鎖定，導致「GIGO」數據。
*   **提供高維度、高保真的人臉幾何圖**：`TFFacemesh`是TensorFlow.js (TFJS) 與MediaPipe Facemesh拓撲的整合，它能提供一個**密集的468點3D人臉幾何圖**。
*   **精確提取眼部特徵**：此密集拓撲允許計算管線提取**高保真的眼部區域塊 (ocular patches)**，即使在輕微的頭部平移或變化的環境光照下，也能**精確隔離鞏膜、虹膜和瞳孔的邊界**。
*   **為回歸模型提供優質輸入**：`TFFacemesh`產生的「高維度特徵向量」是建立可靠眼動追蹤模型的關鍵輸入。若不明確設定，系統可能沿用舊模型，導致數據質量低下。

**具體參數設定 (JavaScript)**：

```javascript
await window.webgazer
   .setTracker("TFFacemesh") //  explicitly set to the TFJS MediaPipe Facemesh backend
   // ... other configurations
   .begin();
```

#### 1.2. 必須明確設定 `ridge` regression 的原因：

*   **高效處理高維度特徵向量**：`TFFacemesh` tracker產生的468點3D人臉幾何圖帶來了高維度特徵向量。`ridge` (或 `threadedRidge`) 回歸模型能夠**高效處理這些數據**。
*   **防止過度擬合 (Overfitting) 和穩定預測**：Webcam串流的輸入本質上是帶有雜訊的。`ridge` 回歸模型通過應用**L2 regularization (L2 正則化)**來防止模型對這些固有的雜訊輸入過度擬合。
*   **確保穩定實時預測**：對於需要穩定、實時預測的標準網頁應用，`ridge` 實作提供了最穩定的基線，避免了異步競爭條件或渲染不同步問題。

**具體參數設定 (JavaScript)**：

```javascript
await window.webgazer
   // ... set tracker
   .setRegression("ridge") // 'ridge' regression maps facial features to screen coordinates using L2 regularization
   // ... other configurations
   .begin();
```

**算法邏輯 (`ridge` Regression)**：

Ridge 回歸是線性回歸的一種形式，其目標是最小化殘差平方和，並在目標函數中添加一個懲罰項 (penalty term)，即權重向量的L2範數（歐幾里得距離的平方）乘以一個正則化參數 `λ` (lambda)。

標準線性回歸 (Ordinary Least Squares, OLS) 的目標函數是：
$$ \min_w \| Xw - y \|_2^2 $$

Ridge 回歸的目標函數是：
$$ \min_w \| Xw - y \|_2^2 + \lambda \| w \|_2^2 $$

其中：
*   $X$ 是特徵矩陣（由TFFacemesh提取的眼部特徵）。
*   $w$ 是回歸係數（模型學習到的權重）。
*   $y$ 是目標變量（螢幕上的點擊或移動座標）。
*   $\| \cdot \|_2^2$ 表示L2範數的平方。
*   $\lambda$ 是正則化參數，控制正則化的強度。

這個L2正則化懲罰項的作用是**約束權重向量 $w$ 的大小**，使其係數趨向於較小的值，但不會完全歸零。這有效地「縮小係數 (shrinking coefficients)」，防止它們變得過大，從而提高模型的泛化能力，使其在面對未見過的嘈雜數據時表現更穩定。

**回歸模塊比較 (`ridge` vs. others)**：

| **Regression Module** | **Algorithmic Mechanism** | **Optimal Use Case** | **Computational Overhead** |
| :------------------ | :------------------------------------------------------ | :--------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| `ridge`             | 線性回歸，利用L2正則化懲罰。通過縮小係數來防止極端權重值。 | 需要穩定、實時預測且無過度多線程複雜性的標準網頁應用。                                            | 適中。在標準JavaScript事件循環和TFJS WebGL後端中運行。 |

### 2. 若不使用 L2 regularization 會發生什麼事？

若不使用L2 regularization，即不使用`ridge`或`threadedRidge`等帶有L2正則化的回歸模型，將會發生以下情況：

*   **模型過度擬合 (Overfitting)**：模型會過度學習訓練數據中的雜訊和異常值。Webcam串流的輸入 inherently 帶有雜訊，例如：
    *   `anomalous lighting reflections` (異常光線反射)
    *   `micro-expressions` (微表情)
    *   `slight head translations` (輕微頭部平移)
    *   其他環境變數。
    如果不加約束，模型會試圖完美擬合這些雜訊，導致對新數據的泛化能力極差。
*   **權重矩陣災難性飆升 (`catastrophic spikes in the weight matrices`)**：文獻明確指出：「Without L2 regularization, anomalous lighting reflections could cause **catastrophic spikes in the weight matrices**, throwing the prediction cursor off the screen entirely.」（若無L2正則化，異常的光線反射可能導致權重矩陣災難性飆升，將預測游標完全拋出螢幕外。）
    *   這意味著模型對輸入中的微小變化會產生極端敏感的反應，導致輸出結果（即預測的凝視座標）極度不穩定和不準確。
*   **預測游標失控**：當權重矩陣飆升時，模型可能會將微小的眼部特徵變化錯誤地映射到螢幕上非常遙遠甚至超出螢幕邊界的座標，從而導致**預測游標完全失控，無法有效追蹤用戶的凝視點 (Point of Regard, POR)**。

簡而言之，L2 regularization對於Webcam眼動追蹤至關重要，它能穩定模型的學習過程，防止因環境雜訊導致的預測錯誤，確保模型能夠從嘈雜的輸入中提取穩健的模式，而不是被單一的異常事件所主導。

## Methodology (最佳實作路徑與架構)
好的，作為 Web_cam 領域的專家，我將根據您提供的文獻，詳細解釋『Cursor-Guided Reading (Flashlight)』的實作邏輯，並說明其必要性與關鍵的 CSS 遮罩語法。

---

## Cursor-Guided Reading (Flashlight) 技術詳解

### 核心概念與目的

`Cursor-Guided Reading (Flashlight)` 是一種創新的使用者介面 (UI) 互動模式，旨在**強制性地將使用者的滑鼠游標軌跡與其視覺焦點 (Point of Regard, POR) 嚴格綁定**。其核心目標是解決 `WebGazer.js` 在標準閱讀應用中長期存在的「運動路徑與視覺路徑脫鉤」問題，從而提供連續、高精度的訓練數據給底層的 Ridge Regression 模型，徹底消除模型漂移與準確度下降。

### 為什麼需要將 `mousemove` 綁定到視覺焦點上？

在預設的 `WebGazer.js` 配置中，`webgazer.recordScreenPosition(x, y, eventType)` 方法會**自動綁定瀏覽器的 `mousemove` 和 `click` 全局事件監聽器**。這意味著系統會持續地從滑鼠移動中收集訓練數據。然而，在一般文字閱讀情境下，這會導致**災難性的數據污染 (GIGO)**，原因如下：

1.  **認知解耦 (Cognitive Decoupling):**
    *   閱讀行為包含快速跳躍的眼球運動 (saccades) 和短暫的注視 (fixations)。
    *   在數字閱讀任務中，使用者通常會將滑鼠游標停留在螢幕的某個靜態位置（例如左側邊距），而他們的眼睛則在文字區塊中活躍掃描。
2.  **錯誤的數據關聯 (Erroneous Data Association):**
    *   由於 `WebGazer` 持續監聽 `mousemove` 事件，它會錯誤地將使用者眼睛看向螢幕右側時的**眼部特徵 (ocular features)**，與靜態停留在螢幕左側的**滑鼠游標座標 (`x`, `y`)** 關聯起來。
    *   這種「眼睛在右，滑鼠在左」的**運動與視覺路徑不一致 (Motor-Visual Disconnect)**，會主動地將不準確的數據向量輸入到 Ridge Regression 矩陣中。
3.  **模型污染與漂移 (Model Pollution and Drift):**
    *   當 Ridge Regression 模型接收到大量錯誤關聯的訓練數據時，其權重矩陣 (weight matrices) 將被「污染」。
    *   長期下來，這會導致模型學習到錯誤的映射關係，使凝視預測游標產生嚴重的漂移，並大幅降低長時間會話中的追蹤準確度。

`Cursor-Guided Reading` 通過強制使用者必須將滑鼠移動到文字上才能閱讀，**結構性地消除了這種運動與視覺的脫鉤**。此時，每一個滑鼠移動的微秒，都代表著使用者眼睛的真實注視點，從而將原本污染模型的 `mousemove` 事件，轉化為連續、高精度的**「完美對齊的訓練向量」**流，持續強化模型準確性。

| **事件監聽器** | **預設閱讀任務狀態** | **Cursor-Guided Reading 狀態** | **對 Ridge Regression 模型的影響** |
| :------------ | :------------------ | :-------------------------- | :---------------------------------- |
| `click`       | 不頻繁（滾動或點擊連結）   | 不頻繁                      | 長期影響微乎其微（數據量少）         |
| `mousemove`   | 極高頻，但與眼球注視物理脫鉤 | 極高頻，透過 UI 遮罩嚴格與眼球注視綁定 | **預設：** 導致嚴重模型漂移。  **引導：** 持續硬化模型準確度，消除漂移。 |

### 實作邏輯：JavaScript 與 CSS 遮罩機制

#### 1. JavaScript 邏輯：動態遮罩與 WebGazer 訓練

JavaScript 負責監聽指定閱讀區域內的滑鼠移動事件，並根據游標位置動態更新 CSS 遮罩，同時將當前的滑鼠座標作為高精度訓練點傳遞給 `WebGazer.js`。

**關鍵 JavaScript 程式碼與邏輯：**

```javascript
document.addEventListener("DOMContentLoaded", () => {
    const readingZone = document.getElementById('guided-reading-zone'); // 閱讀內容的父容器
    const readingLens = document.getElementById('reading-lens');         // 實現手電筒效果的遮罩元素

    if (!readingZone || !readingLens) return;

    // 監聽閱讀區域內的滑鼠移動事件
    readingZone.addEventListener('mousemove', (e) => {
        // Step 1: 計算滑鼠在 `readingZone` 容器內的相對座標 (x, y)。
        // `e.clientX` 和 `e.clientY` 是視口 (viewport) 的絕對座標。
        // `getBoundingClientRect()` 獲取元素相對於視口的大小和位置。
        const rect = readingZone.getBoundingClientRect();
        const x = e.clientX - rect.left; // 相對於 readingZone 左邊緣的 X 座標
        const y = e.clientY - rect.top;  // 相對於 readingZone 上邊緣的 Y 座標

        // Step 2: 更新 CSS 遮罩的 `mask-image` (或 `-webkit-mask-image`) 屬性。
        // 使用徑向漸變 (radial-gradient) 在滑鼠座標處創建一個清晰的「光斑」。
        // `at ${x}px ${y}px` 將徑向漸變的中心動態設定為滑鼠的相對座標。
        // `circle 80px` 定義了光斑的半徑。
        // `black 40%, rgba(0,0,0,0.8) 70%, transparent 100%` 定義了漸變的顏色和透明度，
        // `black` 區域顯示清晰文字，`transparent` 區域隱藏文字，中間部分平滑過渡。
        readingLens.style.webkitMaskImage = `radial-gradient(circle 80px at ${x}px ${y}px, black 40%, rgba(0,0,0,0.8) 70%, transparent 100%)`;
        readingLens.style.maskImage = `radial-gradient(circle 80px at ${x}px ${y}px, black 40%, rgba(0,0,0,0.8) 70%, transparent 100%)`;

        // Step 3: 核心 API 調用 - 將當前的滑鼠視口座標作為訓練點餵給 WebGazer。
        // 此時，由於 UI 機制強制使用者將眼睛注視於滑鼠游標，
        // `e.clientX` 和 `e.clientY` 就是使用者真實的視覺注視點座標。
        // `eventType: 'move'` 標識這是一個移動事件的訓練數據。
        // WebGazer 收到此數據後，會提取當前攝像頭幀中的 TFFacemesh 眼部特徵，
        // 並將這些特徵與 `(e.clientX, e.clientY)` 向量進行數學映射，
        // 進而更新 Ridge Regression 模型的權重矩陣，持續優化預測模型。
        if (window.webgazer && window.webgazer.isReady()) {
            window.webgazer.recordScreenPosition(e.clientX, e.clientY, 'move');
        }
    });

    // 當滑鼠離開閱讀區域時，隱藏光斑，防止視覺干擾或錯誤追蹤。
    readingZone.addEventListener('mouseleave', () => {
        readingLens.style.webkitMaskImage = `none`;
        readingLens.style.maskImage = `none`;
    });
});
```

#### 2. HTML 結構：雙層文字與遮罩

HTML 結構採用雙層文字模式：一層是模糊的背景文字，一層是清晰的、被遮罩控制的前景文字。

```html
<main id="reading-app-container" style="display: none;">
    <div class="reading-container" id="guided-reading-zone">
        
        <!-- 底層文字：始終模糊顯示，防止直接閱讀 -->
        <div class="text-layer" id="blurred-text">
            Cognitive psychology indicates that saccadic movements jump between 
            fixations during the reading process. By utilizing a motor-guided 
            reading constraint, the spatial gap between mouse tracking and visual 
            attention is mathematically eliminated. This generates a continuous 
            stream of high-fidelity, perfectly aligned training data for the 
            Ridge Regression matrix, completely preventing long-term tracking drift.
        </div>

        <!-- 上層遮罩層：包含清晰的文字，並由 CSS 遮罩控制顯示區域 -->
        <div class="lens-mask" id="reading-lens">
            <div class="text-content">
                Cognitive psychology indicates that saccadic movements jump between 
                fixations during the reading process. By utilizing a motor-guided 
                reading constraint, the spatial gap between mouse tracking and visual 
                attention is mathematically eliminated. This generates a continuous 
                stream of high-fidelity, perfectly aligned training data for the 
                Ridge Regression matrix, completely preventing long-term tracking drift.
            </div>
        </div>
        
    </div>
</main>
```

#### 3. 關鍵 CSS 遮罩語法重點

CSS 是實現「手電筒」視覺效果的基石。

```css
/* Core Container for the reading typography */
.reading-container {
    position: relative;
    max-width: 800px;
    margin: 0 auto;
    font-size: 1.5rem;
    line-height: 2;
    color: #334155;
    cursor: none; /* 【重點】隱藏預設的作業系統游標，增強「透鏡」的沉浸感 */
    user-select: none; /* 防止文字被選取，避免干擾 */
    overflow: hidden; /* 防止內容溢出 */
}

/* The underlying text layer - heavily blurred to prevent reading without the mouse */
.text-layer {
    filter: blur(10px); /* 【重點】對底層文字應用強烈模糊濾鏡，使其無法閱讀 */
    opacity: 0.3;      /* 降低透明度，使其作為背景襯托 */
    transition: filter 0.1s;
    pointer-events: none; /* 禁用鼠標事件，使其不影響上層遮罩的事件捕捉 */
}

/* The clear 'lens' layer that dynamically follows the mouse */
.lens-mask {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    /* 【重點】允許滑鼠事件穿透此元素，使其下方的 `readingZone` 能夠捕捉 `mousemove` */
    pointer-events: none; 
    
    /* 
     * 【核心重點】Advanced CSS Masking:
     * 使用 CSS `mask-image` (或 `-webkit-mask-image` for Webkit browsers)
     * 結合 `radial-gradient` 來創建一個動態的、清晰的圓形區域。
     * 初始 `at -100px -100px` 將遮罩中心定位在螢幕外，使所有文字在互動前均被隱藏。
     */
    -webkit-mask-image: radial-gradient(
        circle 70px at -100px -100px, /* 初始位置在螢幕外 */
        black 0%,                 /* 遮罩為黑色，表示內容完全可見 */
        black 50%, 
        transparent 100%          /* 遮罩為透明，表示內容完全隱藏 */
    );
    mask-image: radial-gradient( /* 標準語法 */
        circle 70px at -100px -100px, 
        black 0%, 
        black 50%, 
        transparent 100%
    );
    background-color: transparent; /* 背景透明，僅依賴 mask-image 效果 */
}

/* The sharp typography strictly contained inside the lens */
.lens-mask .text-content {
    color: #0f172a; /* 高對比度閱讀文字，僅在遮罩清晰區域顯示 */
    font-weight: 500;
}
```

**CSS 遮罩語法重點整理：**

1.  **`cursor: none;` (在 `.reading-container`):**
    *   **作用:** 隱藏作業系統的預設滑鼠游標。
    *   **理由:** 為了營造更沉浸式的「手電筒」效果，讓使用者只看到由 `lens-mask` 動態生成的清晰區域，而不是兩個游標（系統預設 + 應用程序模擬）。

2.  **`filter: blur(10px);` (在 `.text-layer`):**
    *   **作用:** 對底層文字應用高斯模糊濾鏡。
    *   **理由:** 強制使用者無法直接閱讀模糊的文字，必須將滑鼠移動到清晰區域才能看清內容，從而達到強制性的「滑鼠引導」效果。

3.  **`pointer-events: none;` (在 `.lens-mask` 和 `.text-layer`):**
    *   **作用:** 阻止元素成為滑鼠事件的目標。
    *   **理由:** **極其關鍵。**如果 `lens-mask` 和 `text-layer` 捕獲了滑鼠事件，那麼父級的 `readingZone` 將無法監聽到 `mousemove` 事件，導致遮罩無法跟隨游標移動。設置 `none` 讓事件穿透到下方的 `readingZone` 元素。

4.  **`-webkit-mask-image` / `mask-image` (在 `.lens-mask`):**
    *   **作用:** 這是實現「手電筒」視覺效果的核心屬性。它定義了一個圖像或漸變，用於遮罩元素的內容。
    *   **語法結構 (由 JavaScript 動態更新):**
        ```css
        radial-gradient(
            circle <radius> at <x>px <y>px, 
            black <start-percent>%,          /* 完全可見區域 */
            rgba(0,0,0,0.8) <mid-percent>%,  /* 半透明過渡區域 */
            transparent <end-percent>%       /* 完全隱藏區域 */
        );
        ```
    *   **關鍵參數解釋:**
        *   `circle <radius>`: 定義遮罩的形狀為圓形，並指定其半徑大小 (e.g., `80px`)。
        *   `at <x>px <y>px`: **由 JavaScript 動態設定**，指定徑向漸變的中心點座標，這就是滑鼠游標的位置。
        *   `black 40%`: 從中心開始，到 40% 的半徑範圍內，遮罩顏色為純黑。在 CSS 遮罩中，黑色代表完全顯示元素內容。
        *   `rgba(0,0,0,0.8) 70%`: 從 40% 到 70% 的半徑範圍內，遮罩顏色從純黑漸變到半透明的黑色，創造柔和的邊緣過渡。
        *   `transparent 100%`: 從 70% 到 100% 的半徑範圍內，遮罩顏色漸變為完全透明。在 CSS 遮罩中，透明代表完全隱藏元素內容。

### 總結與影響

通過實施 `Cursor-Guided Reading (Flashlight)` 技術，應用程式**結構性地保證了使用者凝視點與物理游標座標的持續且嚴格的綁定**。這種連續、高精度的 `(x, y)` 座標向量與眼部特徵矩陣的對齊，將 `mousemove` 事件從模型污染源轉變為強大的**連續校準數據流**，從根本上解決了 `WebGazer.js` 在長時間認知任務中模型準確性退化的問題，使得從瀏覽器獲取的凝視數據更可靠，更適用於嚴格的 HCI 分析。

## Constraints & Edge Cases (軟硬體限制與極端狀況)
API 錯誤: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}

## Example Q&A (Vibe Coding 指引與範例)
作為 web_cam 領域的專家，我理解在非受限瀏覽器環境中部署基於攝像頭的眼動追蹤，要求我們超越預設配置，實施精確的架構和 UI/UX 優化。文件詳盡地說明了如何通過 `TFFacemesh`、`ridge` 回歸、強制校準、光學安全區以及鼠標引導閱讀來顯著提升 WebGazer.js 的準確性。

以下是根據這份文獻設計的 2 個 Vibe Coding 實作問答，特別針對 9 點校準中的 CSS 絕對位置設定，並包含具體參數、公式與演算法邏輯，以輔助您的 Vibe Coding 實作。

---

### Vibe Coding 實作問答 1：動態生成並精確定位 9 點校準標的

**情境：**
您正在為一個需要高精度 WebGazer.js 眼動追蹤的閱讀應用程式，實作其強制性 9 點校準模組。根據文獻指引，校準點必須位於 `10%`, `50%`, `90%` 的視口寬度（`vw`）和高度（`vh`）處，並且每個點應使用 `transform: translate(-50%, -50%)` 精確居中。您的任務是完成 `CalibrationProtocol` 類別中的 `renderNextPoint` 方法，使其能根據預定義的 `this.points` 數組，動態創建並將校準點絕對定位到正確的屏幕座標上。

**任務：**
請補充以下 JavaScript 程式碼片段中的 `renderNextPoint` 方法，實現校準點的創建和絕對定位邏輯。

**程式碼片段：**

```javascript
/**
 * 9-Point Calibration Protocol Logic
 * Generates interactive points at 10vw, 10vh, 50vw, 10vh, etc.
 */
class CalibrationProtocol {
    constructor() {
        this.gateElement = document.getElementById('calibration-gate');
        this.container = document.getElementById('calibration-points-container');
        this.appContainer = document.getElementById('reading-app-container');
        
        this.clicksRequiredPerPoint = 5; 
        this.currentPointIndex = 0;
        
        // 標準 9 點網格，映射為視口百分比 (vw, vh)
        this.points = [
            { x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 },
            { x: 10, y: 50 }, { x: 50, y: 50 }, { x: 90, y: 50 },
            { x: 10, y: 90 }, { x: 50, y: 90 }, { x: 90, y: 90 }
        ];
    }

    init() {
        const checkReady = setInterval(() => {
            if (window.webgazer && window.webgazer.isReady()) {
                clearInterval(checkReady);
                this.renderNextPoint();
            }
        }, 500);
    }

    renderNextPoint() {
        if (this.currentPointIndex >= this.points.length) {
            this.unlockApplication();
            return;
        }

        const pointData = this.points[this.currentPointIndex];
        const pointElement = document.createElement('div');
        pointElement.className = 'cal-point active';
        
        // --- 請在此處補充程式碼，實現校準點的絕對定位 ---
        // 提示：使用 pointData.x 和 pointData.y，並結合 'vw' 和 'vh' 單位
        // 居中效果 (transform: translate(-50%, -50%)) 已由 .cal-point CSS 類別處理
        
        pointElement.innerText = this.clicksRequiredPerPoint; // 初始化顯示點擊次數

        // 監聽點擊事件 (在此處略過其完整實作，專注於定位)
        pointElement.addEventListener('click', (e) => {
            // ... (Vibe Coding Question 2 將涵蓋此部分)
        });

        this.container.appendChild(pointElement);
    }

    // ... (init 和 unlockApplication 方法的其他部分在此處略過)
    unlockApplication() {
        window.webgazer.showVideoPreview(false)
                      .showFaceOverlay(false)
                      .showFaceFeedbackBox(false);

        this.gateElement.style.opacity = '0';
        setTimeout(() => {
            this.gateElement.style.display = 'none';
            this.appContainer.style.display = 'block';
        }, 500);
        
        console.log("Implicit Calibration Protocol Complete. Regression Model Anchored. App Unlocked.");
    }
}
```

**期望輸出（程式碼片段）**

```javascript
// ... (之前的程式碼)

    renderNextPoint() {
        if (this.currentPointIndex >= this.points.length) {
            this.unlockApplication();
            return;
        }

        const pointData = this.points[this.currentPointIndex];
        const pointElement = document.createElement('div');
        pointElement.className = 'cal-point active';
        
        // --- 補充部分 ---
        pointElement.style.left = `${pointData.x}vw`;
        pointElement.style.top = `${pointData.y}vh`;
        // --- 補充結束 ---
        
        pointElement.innerText = this.clicksRequiredPerPoint; 

        pointElement.addEventListener('click', (e) => {
            // ...
        });

        this.container.appendChild(pointElement);
    }

// ... (之後的程式碼)
```

**參數、公式或演算法邏輯：**
*   **CSS 絕對定位公式:** `position: absolute; left: Xvw; top: Yvh;`
    *   `X` 和 `Y` 值來自 `this.points` 數組，分別代表視口的水平和垂直百分比。
    *   `vw` (viewport width) 和 `vh` (viewport height) 是 CSS 單位，確保校準點的位置與屏幕尺寸成比例，符合文獻中 `10%`, `50%`, `90%` 的要求。
*   **居中原理:** 雖然在 JavaScript 中只設置了 `left` 和 `top`，但文獻中提供的 `.cal-point` CSS 類別包含了 `transform: translate(-50%, -50%)`。這個 CSS 屬性會將元素的中心點對齊到其 `left` 和 `top` 所指定的座標，而不是左上角，從而實現精確居中。

---

### Vibe Coding 實作問答 2：實作校準點點擊互動與 WebGazer 數據記錄

**情境：**
在上一題中，您已成功動態生成並定位了 9 點校準標的。現在，您的任務是進一步完成 `CalibrationProtocol` 中點擊事件的處理邏輯。根據文獻要求，每個校準點需要被點擊指定次數（例如 5 次），每次點擊都需要調用 `window.webgazer.recordScreenPosition` 來收集訓練數據，並在完成後將點的狀態視覺化為“完成”並推進到下一個校準點。

**任務：**
請補充以下 JavaScript 程式碼片段中的 `pointElement.addEventListener('click', ...)` 函數，實現以下邏輯：
1.  每次點擊，`clickCount` 增加，並更新校準點上的數字顯示。
2.  **核心：** 調用 `window.webgazer.recordScreenPosition`，使用當前點擊事件的屏幕座標 (`e.clientX`, `e.clientY`) 和事件類型 `'click'` 來訓練 WebGazer 模型。
3.  當 `clickCount` 達到 `this.clicksRequiredPerPoint` 時：
    *   移除 `active` 類，添加 `completed` 類。
    *   將點的文本改為 '✓'。
    *   在短暫延遲後（例如 300ms），將當前校準點從 DOM 中移除 (`style.display = 'none'`)，並調用 `this.renderNextPoint()` 推進到下一個校準點。

**程式碼片段：**

```javascript
/**
 * 9-Point Calibration Protocol Logic
 * Handles click interactions and WebGazer data recording for calibration points.
 */
class CalibrationProtocol {
    constructor() {
        this.gateElement = document.getElementById('calibration-gate');
        this.container = document.getElementById('calibration-points-container');
        this.appContainer = document.getElementById('reading-app-container');
        
        this.clicksRequiredPerPoint = 5; // 文獻中推薦的點擊次數
        this.currentPointIndex = 0;
        
        this.points = [
            { x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 },
            { x: 10, y: 50 }, { x: 50, y: 50 }, { x: 90, y: 50 },
            { x: 10, y: 90 }, { x: 50, y: 90 }, { x: 90, y: 90 }
        ];
    }

    init() {
        const checkReady = setInterval(() => {
            if (window.webgazer && window.webgazer.isReady()) {
                clearInterval(checkReady);
                this.renderNextPoint();
            }
        }, 500);
    }

    renderNextPoint() {
        if (this.currentPointIndex >= this.points.length) {
            this.unlockApplication();
            return;
        }

        const pointData = this.points[this.currentPointIndex];
        const pointElement = document.createElement('div');
        pointElement.className = 'cal-point active';
        pointElement.style.left = `${pointData.x}vw`;
        pointElement.style.top = `${pointData.y}vh`;
        
        let clickCount = 0; // 為每個點獨立的點擊計數器
        pointElement.innerText = this.clicksRequiredPerPoint - clickCount; // 初始化顯示剩餘點擊次數

        pointElement.addEventListener('click', (e) => {
            // --- 請在此處補充程式碼，實現點擊事件處理邏輯 ---
            // 提示：使用 clickCount, this.clicksRequiredPerPoint, window.webgazer.recordScreenPosition,
            // e.clientX, e.clientY, classList.remove, classList.add, setTimeout
            
        });

        this.container.appendChild(pointElement);
    }

    // ... (init 和 unlockApplication 方法的其他部分在此處略過)
    unlockApplication() {
        window.webgazer.showVideoPreview(false)
                      .showFaceOverlay(false)
                      .showFaceFeedbackBox(false);

        this.gateElement.style.opacity = '0';
        setTimeout(() => {
            this.gateElement.style.display = 'none';
            this.appContainer.style.display = 'block';
        }, 500);
        
        console.log("Implicit Calibration Protocol Complete. Regression Model Anchored. App Unlocked.");
    }
}
```

**期望輸出（程式碼片段）**

```javascript
// ... (之前的程式碼)

        pointElement.addEventListener('click', (e) => {
            clickCount++;
            pointElement.innerText = this.clicksRequiredPerPoint - clickCount;

            // --- 補充部分：WebGazer 數據記錄 ---
            // 核心 API 調用：強制 WebGazer 記錄此精確點擊作為訓練事件。
            // 這會捕獲當前的攝像頭幀，提取眼部特徵，並將這些像素排列數學映射到 (e.clientX, e.clientY) 向量。
            if (window.webgazer && window.webgazer.isReady()) {
                window.webgazer.recordScreenPosition(e.clientX, e.clientY, 'click');
            }
            // --- 補充結束 ---

            if (clickCount >= this.clicksRequiredPerPoint) {
                pointElement.classList.remove('active');
                pointElement.classList.add('completed');
                pointElement.innerText = '✓';
                
                // --- 補充部分：完成後狀態轉換與推進 ---
                // 在短暫延遲後推進狀態機，以提供使用者反饋
                setTimeout(() => {
                    pointElement.style.display = 'none'; // 從渲染樹中移除節點
                    this.currentPointIndex++;
                    this.renderNextPoint(); // 渲染下一個點
                }, 300); 
                // --- 補充結束 ---
            }
        });

// ... (之後的程式碼)
```

**參數、公式或演算法邏輯：**
*   **WebGazer 訓練演算法 (Ridge Regression):**
    *   `window.webgazer.recordScreenPosition(x, y, eventType)`: 這是 WebGazer 的核心 API，用於將用戶的實際屏幕交互 `(x, y)` 座標與當前攝像頭捕捉到的眼部特徵進行關聯。
    *   `e.clientX`, `e.clientY`: 這是 DOM `MouseEvent` 物件的屬性，表示點擊事件發生時，鼠標指針相對於瀏覽器視口左上角的水平和垂直座標。
    *   `'click'`: 作為 `eventType` 參數，明確告知 WebGazer 這是一個點擊事件，用於訓練回歸模型。Ridge 回歸（L2 正則化）利用這些點來學習如何將眼部特徵映射到屏幕位置，並防止對單一噪音輸入的過度擬合。
*   **狀態機邏輯:**
    *   `clickCount` 遞增，並用於視覺化剩餘點擊次數。
    *   **條件判斷:** `if (clickCount >= this.clicksRequiredPerPoint)` 判斷當前點是否完成校準。
    *   **CSS 類別切換:** 通過操作 `classList` 實現視覺反饋 (`active` -> `completed`)。
    *   **非同步流程控制:** `setTimeout` 用於在視覺反饋後引入短暫延遲，再移除當前點並調用 `this.renderNextPoint()`，確保校準流程平滑進行，符合文獻中「brief delay for user feedback」的要求。

---