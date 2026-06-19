const state = {
  sessionId: "",
  running: false,
  testing: false,
  currentPoint: null,
};

let modelsCache = [];

// 13 個校準點分佈在畫面上
const calibrationPoints = [
  [0.08, 0.10],  // 左上角區域
  [0.50, 0.10],  // 上方中間
  [0.92, 0.10],  // 右上角區域
  [0.08, 0.50],  // 左方中間
  [0.50, 0.50],  // 畫面中心
  [0.92, 0.50],  // 右方中間
  [0.08, 0.90],  // 左下角區域
  [0.50, 0.90],  // 下方中間
  [0.92, 0.90],  // 右下角區域
  [0.29, 0.30],  // 左上內區
  [0.71, 0.30],  // 右上內區
  [0.29, 0.70],  // 左下內區
  [0.71, 0.70],  // 右下內區
];

// 5 個驗證點（用於測試，不在校準點中）
const validationPoints = [
  [0.18, 0.22],
  [0.82, 0.22],
  [0.50, 0.50],
  [0.18, 0.78],
  [0.82, 0.78],
];

// Adaptive OneEuro Filter for smooth visual cursor movements
class OneEuroFilter {
  constructor(minCutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.x = null;
    this.dx = null;
    this.lastTime = null;
  }

  alpha(cutoff, rate) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau * rate);
  }

  filter(value, timestamp) {
    if (this.x === null || this.lastTime === null) {
      this.x = value;
      this.dx = 0.0;
      this.lastTime = timestamp;
      return value;
    }

    const dt = (timestamp - this.lastTime) / 1000.0;
    if (dt <= 0) return this.x;

    const rate = 1.0 / dt;
    const dvalue = (value - this.x) * rate;
    const edvalue = this.dx + this.alpha(this.dcutoff, rate) * (dvalue - this.dx);
    this.dx = edvalue;

    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    const filteredValue = this.x + this.alpha(cutoff, rate) * (value - this.x);
    this.x = filteredValue;
    this.lastTime = timestamp;

    return filteredValue;
  }

  reset() {
    this.x = null;
    this.dx = null;
    this.lastTime = null;
  }
}

const filterX = new OneEuroFilter(1.0, 0.007, 1.0);
const filterY = new OneEuroFilter(1.0, 0.007, 1.0);

const els = {
  video: document.querySelector("#video"),
  canvas: document.querySelector("#capture"),
  target: document.querySelector("#target"),
  participant: document.querySelector("#participantName"),
  repeats: document.querySelector("#repeatCount"),
  delay: document.querySelector("#delayTime"),
  calibration: document.querySelector("#calibration"),
  train: document.querySelector("#train"),
  test: document.querySelector("#test"),
  health: document.querySelector("#health"),
  session: document.querySelector("#session"),
  phase: document.querySelector("#phase"),
  progress: document.querySelector("#progress"),
  log: document.querySelector("#log"),
  gazeCursor: document.querySelector("#gaze-cursor"),
  bottomDock: document.querySelector("#bottomDock"),
  toggleSettings: document.querySelector("#toggle-settings"), // opens Model Center Modal
  toggleLogs: document.querySelector("#toggle-logs"),
  logsTray: document.querySelector("#logsTray"),
  gazeCoordinates: document.querySelector("#gaze-coordinates"),
  hudCoords: document.querySelector("#hudCoords"),
  heatmapOverlay: document.querySelector("#heatmap-overlay"),
  testControls: document.querySelector("#testControls"),
  testModeSelect: document.querySelector("#testModeSelect"),
  toggleHeatmap: document.querySelector("#toggleHeatmap"),
  toggleAntiShake: document.querySelector("#toggleAntiShake"),
  toggleCorridorLock: document.querySelector("#toggleCorridorLock"),

  // Collect Modal elements
  collectModal: document.querySelector("#collectModal"),
  collectMode: document.querySelector("#collectMode"),
  closeCollect: document.querySelector("#closeCollect"),
  btnCancelCollect: document.querySelector("#btnCancelCollect"),
  btnStartCollect: document.querySelector("#btnStartCollect"),

  // Train Modal elements
  trainModal: document.querySelector("#trainModal"),
  closeTrain: document.querySelector("#closeTrain"),
  btnCancelTrain: document.querySelector("#btnCancelTrain"),
  btnStartTrain: document.querySelector("#btnStartTrain"),
  selectDataset: document.querySelector("#selectDataset"),
  selectBaseModel: document.querySelector("#selectBaseModel"),
  outputModelName: document.querySelector("#outputModelName"),

  // Models Center Modal elements
  modelsModal: document.querySelector("#modelsModal"),
  closeModels: document.querySelector("#closeModels"),
  btnCloseModels: document.querySelector("#btnCloseModels"),
  modelsList: document.querySelector("#modelsList"),
};

// 記錄訊息到日誌區
function log(message) {
  const time = new Date().toLocaleTimeString();
  els.log.textContent = `[${time}] ${message}\n${els.log.textContent}`.slice(0, 4000);
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

// 將標準化座標轉換為實際視口座標（點的收集占全版面）
function pointToViewport(point) {
  const gutter = 48;
  const usableWidth = Math.max(160, window.innerWidth - gutter * 2);
  const usableHeight = Math.max(160, window.innerHeight - gutter * 2);
  return {
    x: gutter + point[0] * usableWidth,
    y: gutter + point[1] * usableHeight,
  };
}

// 移動目標圓點到指定位置
function moveTarget(point) {
  state.currentPoint = point;
  const pos = pointToViewport(point);
  els.target.style.left = `${pos.x}px`;
  els.target.style.top = `${pos.y}px`;
  return pos;
}

// 設定執行狀態
function setRunning(running) {
  state.running = running;
  if (running) {
    els.train.disabled = true;
    els.train.className = "btn-inactive";
    els.test.disabled = true;
    els.test.className = "btn-inactive";
  }
}

// 檢查後端連線狀態
async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const isOk = data.ok;
    els.health.className = isOk ? "health-dot ok" : "health-dot bad";
    if (isOk) {
      els.calibration.disabled = false;
      els.train.disabled = false;
      els.train.classList.remove("btn-inactive");
      els.test.disabled = false;
      els.test.classList.remove("btn-inactive");
      els.session.textContent = "已連線，點擊收集以開始";
    }
  } catch {
    els.health.className = "health-dot bad";
    els.session.textContent = "伺服器未連線";
  }
}

// 啟動 webcam
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
    log("攝影機已就緒");
  } catch (err) {
    log(`啟動攝影機失敗: ${err.message}`);
  }
}

// 截取當前畫面
function captureFrame() {
  const targetWidth = 640;
  const videoAspect = els.video.videoHeight ? (els.video.videoWidth / els.video.videoHeight) : (4/3);
  const targetHeight = Math.round(targetWidth / videoAspect);

  els.canvas.width = targetWidth;
  els.canvas.height = targetHeight;
  const context = els.canvas.getContext("2d", { willReadFrequently: false });
  context.drawImage(els.video, 0, 0, targetWidth, targetHeight);
  return els.canvas.toDataURL("image/jpeg", 0.80);
}

// 截取當前畫面 (二進位 Blob，超低延遲串流專用)
function captureFrameBlob() {
  const targetWidth = 640;
  const videoAspect = els.video.videoHeight ? (els.video.videoWidth / els.video.videoHeight) : (4/3);
  const targetHeight = Math.round(targetWidth / videoAspect);

  els.canvas.width = targetWidth;
  els.canvas.height = targetHeight;
  const context = els.canvas.getContext("2d", { willReadFrequently: false });
  context.drawImage(els.video, 0, 0, targetWidth, targetHeight);
  return new Promise((resolve) => {
    els.canvas.toBlob((blob) => {
      resolve(blob);
    }, "image/jpeg", 0.80);
  });
}

// 建立新 session
async function createSession() {
  const participantId = els.participant.value.trim() || "anonymous";
  const response = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ participant_id: participantId }),
  });
  if (!response.ok) {
    throw new Error(`建立 session 失敗: ${response.status}`);
  }
  const data = await response.json();
  state.sessionId = data.session_id;
  els.session.textContent = state.sessionId;
  log(`已建立數據資料夾: ${state.sessionId}`);
}

// 傳送一筆樣本到後端
async function postSample(phase, pointIndex, repeatIndex, pos) {
  const imageData = captureFrame();
  const targetXNorm = (pos.x / window.innerWidth) * 2 - 1;
  const targetYNorm = (pos.y / window.innerHeight) * 2 - 1;
  const response = await fetch("/api/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      image_data: imageData,
      target_x: pos.x,
      target_y: pos.y,
      target_x_norm: targetXNorm,
      target_y_norm: targetYNorm,
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
      screen_width: window.screen.width,
      screen_height: window.screen.height,
      phase,
      point_index: pointIndex,
      repeat_index: repeatIndex,
    }),
  });
  if (!response.ok) {
    throw new Error(`傳送樣本失敗: ${response.status}`);
  }
  return response.json();
}

// 執行一輪收集
async function runPhase(phase, points) {
  try {
    await createSession();
  } catch (err) {
    log(err.message);
    return;
  }
  
  const mode = (phase === "calibration" && els.collectMode) ? els.collectMode.value : "standard";
  let targetPoints = points;
  if (mode === "four_corners") {
    targetPoints = [
      [0.05, 0.05],  // 左上角
      [0.95, 0.05],  // 右上角
      [0.05, 0.95],  // 左下角
      [0.95, 0.95]   // 右下角
    ];
  }

  const repeats = mode === "four_corners" ? 1 : Math.max(1, Math.min(8, Number.parseInt(els.repeats.value, 10) || 1));
  const delayMs = Math.max(300, Math.min(5000, Number.parseInt(els.delay.value, 10) || 900));
  const total = mode === "four_corners" ? targetPoints.length * 10 : targetPoints.length * repeats;
  let done = 0;

  setRunning(true);
  els.bottomDock.classList.add("fade-out"); // 收集時淡出控制列，避免干擾
  els.phase.textContent = mode === "four_corners" ? "四角微調中" : (phase === "calibration" ? "收集中" : "驗證中");
  els.progress.textContent = `0 / ${total}`;

  try {
    if (mode === "four_corners") {
      // 四角長時間注視模式：4 個角落，每個點停留 3 秒並擷取 10 幀
      for (let pointIndex = 0; pointIndex < targetPoints.length; pointIndex += 1) {
        if (!state.running) {
          log(`四角微調已停止`);
          return;
        }
        const pos = moveTarget(targetPoints[pointIndex]);
        // 先等待 500ms 讓使用者視線穩定對齊角落
        await sleep(500);
        els.target.classList.add("capturing");

        // 連續擷取 10 幀，每幀間隔 250ms
        for (let frameIdx = 0; frameIdx < 10; frameIdx++) {
          if (!state.running) return;
          const result = await postSample(phase, pointIndex, frameIdx, pos);
          done += 1;
          els.progress.textContent = `${done} / ${total}`;
          log(result.ok ? `四角注視 點${pointIndex+1} 幀${frameIdx+1}/10 成功` : `四角注視 點${pointIndex+1} 幀${frameIdx+1}/10 失敗: ${result.error}`);
          await sleep(250);
        }
        els.target.classList.remove("capturing");
        await sleep(200);
      }
    } else {
      // 標準 13 點校準模式
      for (let repeat = 0; repeat < repeats; repeat += 1) {
        for (let pointIndex = 0; pointIndex < targetPoints.length; pointIndex += 1) {
          if (!state.running) {
            log(`${phase} 已停止`);
            return;
          }
          const pos = moveTarget(targetPoints[pointIndex]);
          await sleep(delayMs);
          els.target.classList.add("capturing");
          const result = await postSample(phase, pointIndex, repeat, pos);
          els.target.classList.remove("capturing");
          done += 1;
          els.progress.textContent = `${done} / ${total}`;
          log(result.ok ? `${phase} ${done}/${total} 成功` : `${phase} ${done}/${total} 失敗: ${result.error}`);
          await sleep(120);
        }
      }
    }
    
    if (done === total) {
      if (mode === "four_corners") {
        log("四角極致邊界注視數據收集完成！請點擊「訓練模型」開始訓練 Stage 2 微調模型。");
      } else {
        log("收集完成！一輪 13 個校準點已完成。請點擊「訓練模型」開始訓練。");
      }
      els.train.disabled = false;
      els.train.className = "glow-ready"; // 呼吸燈發光提示
    }
  } finally {
    els.phase.textContent = "閒置";
    setRunning(false);
    els.calibration.textContent = "資料收集";
    els.bottomDock.classList.remove("fade-out");
    
    // Enable other buttons
    els.train.disabled = false;
    els.train.classList.remove("btn-inactive");
    els.test.disabled = false;
    els.test.classList.remove("btn-inactive");
  }
}

// 眼動熱區圖座標點快取與繪製邏輯
const gazePoints = [];
let heatmapAnimFrame = null;

function drawHeatmap() {
  const canvas = els.heatmapOverlay;
  if (!state.testing || !els.toggleHeatmap.checked || !canvas) return;
  const ctx = canvas.getContext("2d");
  const w = window.innerWidth;
  const h = window.innerHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.clearRect(0, 0, w, h);

  const now = Date.now();
  const maxAge = 15000; // 點點保留 15 秒後淡出

  // 過濾掉過期點
  for (let i = gazePoints.length - 1; i >= 0; i--) {
    if (now - gazePoints[i].t > maxAge) {
      gazePoints.splice(0, i + 1);
      break;
    }
  }

  // 疊加混合模式，讓重合點亮起
  ctx.globalCompositeOperation = "screen";

  gazePoints.forEach((pt) => {
    const age = now - pt.t;
    const alpha = Math.max(0, 1 - age / maxAge) * 0.45; // 最大透明度 0.45

    const grad = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, 70);
    grad.addColorStop(0, `rgba(239, 68, 68, ${alpha})`);      // 熱區中心：紅色
    grad.addColorStop(0.3, `rgba(245, 158, 11, ${alpha * 0.65})`); // 過渡區：橙黃色
    grad.addColorStop(0.6, `rgba(16, 185, 129, ${alpha * 0.25})`);  // 邊緣區：綠色
    grad.addColorStop(1, "rgba(0, 0, 0, 0)");

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 70, 0, Math.PI * 2);
    ctx.fill();
  });
}

function startHeatmapAnimation() {
  if (heatmapAnimFrame) return;
  
  function anim() {
    if (state.testing && els.toggleHeatmap.checked) {
      drawHeatmap();
      heatmapAnimFrame = requestAnimationFrame(anim);
    } else {
      heatmapAnimFrame = null;
    }
  }
  
  heatmapAnimFrame = requestAnimationFrame(anim);
}

// 全局 WebSocket 連線與即時預測
let activeWS = null;

async function runPredictionLoop() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/predict/ws`;
  const ws = new WebSocket(wsUrl);
  activeWS = ws;

  let isProcessing = false; // Backpressure lock-step flag

  ws.onopen = () => {
    log("高速 WebSocket 視線串流已建立！");
    ws.send(JSON.stringify({ model_name: els.testModeSelect.value }));
  };

  ws.onerror = (err) => {
    log(`WebSocket 錯誤: ${err.message || "連線中斷"}`);
    isProcessing = false;
  };

  ws.onclose = () => {
    log("高速 WebSocket 視線串流已關閉。");
    isProcessing = false;
  };

  // Corridor lock variables
  let lastLockedY = null;

  ws.onmessage = (event) => {
    isProcessing = false; // Release lock as soon as server responds!
    if (!state.testing) return;
    try {
      const data = JSON.parse(event.data);
      if (data.ok) {
        const xNorm = data.screen_xy_norm[0];
        const yNorm = data.screen_xy_norm[1];
        
        // 將標準化 [-1, 1] 映射 to 視窗實際像素點
        const x = ((xNorm + 1.0) * 0.5) * window.innerWidth;
        const y = ((yNorm + 1.0) * 0.5) * window.innerHeight;
        
        let finalX = x;
        let finalY = y;
        
        // OneEuro Filter Anti-Shake
        if (els.toggleAntiShake && els.toggleAntiShake.checked) {
          const nowTs = performance.now();
          finalX = filterX.filter(x, nowTs);
          finalY = filterY.filter(y, nowTs);
        } else {
          filterX.reset();
          filterY.reset();
        }

        // Horizontal Corridor Lock (水平廊道鎖定)
        if (els.toggleCorridorLock && els.toggleCorridorLock.checked) {
          const corridorHeight = 35; // 35px height buffer per text line
          if (lastLockedY === null) {
            lastLockedY = finalY;
          } else {
            const diffY = Math.abs(finalY - lastLockedY);
            if (diffY < corridorHeight) {
              finalY = lastLockedY;
            } else {
              lastLockedY = finalY;
            }
          }
        } else {
          lastLockedY = null;
        }
        
        // 更新跟隨游標位置
        els.gazeCursor.style.left = `${finalX}px`;
        els.gazeCursor.style.top = `${finalY}px`;
        
        // 更新最左上角 HUD 的 x, y 坐標
        els.hudCoords.textContent = `X: ${Math.round(finalX)} px | Y: ${Math.round(finalY)} px`;
        
        // 若啟用熱區圖，則推入歷史點陣列並繪製
        if (els.toggleHeatmap.checked) {
          gazePoints.push({ x: finalX, y: finalY, t: Date.now() });
          drawHeatmap();
        }
      } else {
        log(`測試提示: ${data.error}`);
      }
    } catch (err) {
      log(`解析預測結果出錯: ${err.message}`);
    }
  };

  while (state.testing && ws.readyState !== WebSocket.CLOSED) {
    if (ws.readyState === WebSocket.OPEN && !isProcessing) {
      try {
        const blob = await captureFrameBlob();
        if (blob) {
          isProcessing = true; // Lock the step
          ws.send(blob);
        }
      } catch (err) {
        log(`發送影格失敗: ${err.message}`);
        isProcessing = false;
      }
    }
    // High-frequency polling (10ms) to send the next frame immediately when response is received
    await sleep(10);
  }
}

// Modal control helpers
function openModal(modalEl) {
  if (modalEl) modalEl.classList.remove("hidden");
}

function closeModal(modalEl) {
  if (modalEl) modalEl.classList.add("hidden");
}

// Function to fetch and render datasets & models for the Train Modal
async function openTrainModal() {
  const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  const pName = els.participant.value.trim() || "anonymous";
  els.outputModelName.value = `calib_${pName}_${dateStr}`;
  
  openModal(els.trainModal);
  
  // Populate datasets
  try {
    const res = await fetch("/api/list_datasets");
    const data = await res.json();
    if (data.ok) {
      els.selectDataset.innerHTML = "";
      if (data.datasets.length === 0) {
        els.selectDataset.innerHTML = `<option value="">-- 暫無可用數據集，請先收集 --</option>`;
      } else {
        data.datasets.forEach(ds => {
          const opt = document.createElement("option");
          opt.value = ds.id;
          opt.textContent = ds.display_name;
          els.selectDataset.appendChild(opt);
        });
      }
    }
  } catch (err) {
    log(`載入數據集失敗: ${err.message}`);
  }

  // Populate base models
  try {
    const res = await fetch("/api/list_models");
    const data = await res.json();
    if (data.ok) {
      els.selectBaseModel.innerHTML = `<option value="0">最原始模型 (Frozen Base)</option>`;
      data.models.forEach(model => {
        const opt = document.createElement("option");
        opt.value = model.name;
        opt.textContent = model.display_name;
        els.selectBaseModel.appendChild(opt);
      });
    }
  } catch (err) {
    log(`載入模型起點失敗: ${err.message}`);
  }
}

// Function to fetch and render models in the Model Center Modal
async function openModelsModal() {
  openModal(els.modelsModal);
  try {
    const res = await fetch("/api/list_models");
    const data = await res.json();
    if (data.ok) {
      els.modelsList.innerHTML = "";
      if (data.models.length === 0) {
        els.modelsList.innerHTML = `<div class="empty-list">暫無訓練好的模型，請先點擊「訓練模型」進行校準。</div>`;
      } else {
        data.models.forEach(model => {
          const div = document.createElement("div");
          div.className = "model-item";
          div.innerHTML = `
            <div class="model-info">
              <span class="model-info-name">${model.name}</span>
              <span class="model-info-meta">誤差: ${model.mean_px_error.toFixed(1)} px</span>
            </div>
            <span class="model-badge">Stage ${model.num_stages}</span>
          `;
          els.modelsList.appendChild(div);
        });
      }
    }
  } catch (err) {
    log(`載入模型清單失敗: ${err.message}`);
  }
}

// Function to fetch and update the Prediction Model dropdown in real-time test HUD
async function updatePredictionModelDropdown() {
  try {
    const activeVal = els.testModeSelect.value;
    const res = await fetch("/api/list_models");
    const data = await res.json();
    if (data.ok) {
      modelsCache = data.models; // Cache models list with noise levels
      els.testModeSelect.innerHTML = `<option value="before">最原始模型 (Frozen Base)</option>`;
      data.models.forEach(model => {
        const opt = document.createElement("option");
        opt.value = model.name;
        opt.textContent = model.name;
        els.testModeSelect.appendChild(opt);
      });
      // Restore previous selection if still available
      const exists = Array.from(els.testModeSelect.options).some(opt => opt.value === activeVal);
      if (exists) {
        els.testModeSelect.value = activeVal;
      }
    }
  } catch (err) {
    log(`更新預測模型下拉選單失敗: ${err.message}`);
  }
}

// Dynamic adjustment of OneEuroFilter parameters based on model noise level
function updateFilterParameters() {
  const selectedModelName = els.testModeSelect.value;
  const modelInfo = modelsCache.find(m => m.name === selectedModelName);
  const noise = modelInfo ? (modelInfo.noise_level || 0) : 0;
  
  let minCutoff = 1.0;
  let beta = 0.007;
  
  if (noise > 0) {
    if (noise < 12) {
      minCutoff = 1.5;
      beta = 0.01;
    } else if (noise > 35) {
      minCutoff = 0.45;
      beta = 0.004;
    } else {
      const t = (noise - 12) / (35 - 12);
      minCutoff = 1.5 - t * (1.5 - 0.45);
      beta = 0.01 - t * (0.01 - 0.004);
    }
    log(`[動態濾波器] 模型噪聲: ${noise.toFixed(1)} px -> 自動調整 OneEuroFilter (minCutoff: ${minCutoff.toFixed(2)}, beta: ${beta.toFixed(4)})`);
  } else {
    log(`[動態濾波器] 使用預設參數 (minCutoff: 1.0, beta: 0.007)`);
  }
  
  filterX.minCutoff = minCutoff;
  filterX.beta = beta;
  filterY.minCutoff = minCutoff;
  filterY.beta = beta;
}

// Button and Modal Event Bindings

// Calibration Button opens Collection Settings Modal
els.calibration.addEventListener("click", () => {
  if (state.running) {
    state.running = false;
    els.calibration.textContent = "資料收集";
  } else {
    openModal(els.collectModal);
  }
});

// Modal close button bindings
els.closeCollect.addEventListener("click", () => closeModal(els.collectModal));
els.btnCancelCollect.addEventListener("click", () => closeModal(els.collectModal));

els.btnStartCollect.addEventListener("click", () => {
  closeModal(els.collectModal);
  els.calibration.textContent = "停止收集";
  runPhase("calibration", calibrationPoints).catch((error) => {
    setRunning(false);
    log(error.message);
  });
});

// Train Button opens Training Settings Modal
els.train.addEventListener("click", () => {
  openTrainModal();
});

els.closeTrain.addEventListener("click", () => closeModal(els.trainModal));
els.btnCancelTrain.addEventListener("click", () => closeModal(els.trainModal));

els.btnStartTrain.addEventListener("click", async () => {
  const datasetId = els.selectDataset.value;
  const baseModel = els.selectBaseModel.value;
  const outName = els.outputModelName.value.trim();

  if (!datasetId) {
    log("訓練錯誤: 請先選擇校準數據集！");
    return;
  }
  if (!outName) {
    log("訓練錯誤: 請填寫輸出模型名稱！");
    return;
  }

  closeModal(els.trainModal);

  els.train.disabled = true;
  els.train.textContent = "訓練中...";
  els.train.classList.remove("glow-ready");
  log(`正在啟動校準訓練，儲存為: runs/${outName}.json...`);

  try {
    const response = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data_session_id: datasetId,
        base_model_name: baseModel,
        output_model_name: outName
      }),
    });
    const data = await response.json();
    if (data.ok) {
      log(`校準模型 [${outName}] 訓練成功！平均像素誤差: ${data.best_val_px_error.toFixed(1)} px (共 ${data.train_samples} 筆校準樣本)`);
      
      // Auto-switch prediction model to the newly trained model!
      await updatePredictionModelDropdown();
      els.testModeSelect.value = outName;
      log(`系統已自動將即時預測模型切換至您的新模型: 「${outName}」！`);
      
      els.test.disabled = false;
      els.test.classList.remove("btn-inactive");
    } else {
      log(`校準失敗: ${data.error}`);
      els.train.classList.add("glow-ready");
    }
  } catch (err) {
    log(`校準異常: ${err.message}`);
    els.train.classList.add("glow-ready");
  } finally {
    els.train.disabled = false;
    els.train.textContent = "訓練模型";
  }
});

// Models Management Center triggers
els.toggleSettings.addEventListener("click", () => {
  openModelsModal();
});

els.closeModels.addEventListener("click", () => closeModal(els.modelsModal));
els.btnCloseModels.addEventListener("click", () => closeModal(els.modelsModal));

// Logs tray toggle
els.toggleLogs.addEventListener("click", () => {
  els.logsTray.classList.toggle("hidden");
  els.toggleLogs.classList.toggle("active");
});

// 測試模型按鈕事件
els.test.addEventListener("click", async () => {
  state.testing = !state.testing;
  if (state.testing) {
    await updatePredictionModelDropdown();
    updateFilterParameters();

    els.test.textContent = "停止測試";
    els.calibration.disabled = true;
    els.train.disabled = true;
    els.gazeCursor.classList.remove("hidden");
    els.gazeCoordinates.classList.remove("hidden");
    els.testControls.classList.remove("hidden");
    if (els.toggleHeatmap.checked) {
      els.heatmapOverlay.classList.remove("hidden");
      startHeatmapAnimation();
    }
    log(`即時視線測試已啟動（當前模型：${els.testModeSelect.value}）！`);
    runPredictionLoop().catch((err) => log(`測試異常: ${err.message}`));
  } else {
    els.test.textContent = "測試模型";
    els.calibration.disabled = false;
    els.train.disabled = false;
    els.gazeCursor.classList.add("hidden");
    els.gazeCoordinates.classList.add("hidden");
    els.testControls.classList.add("hidden");
    els.heatmapOverlay.classList.add("hidden");
    
    // 關閉並清除 WebSocket
    if (activeWS) {
      activeWS.close();
      activeWS = null;
    }

    // 清除熱區圖快取與繪圖畫布
    gazePoints.length = 0;
    const canvas = els.heatmapOverlay;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (heatmapAnimFrame) {
      cancelAnimationFrame(heatmapAnimFrame);
      heatmapAnimFrame = null;
    }
    
    log("即時視線測試已停止。");
  }
});

// 當在測試狀態下動態切換預測模型時，通知 WebSocket 伺服器
els.testModeSelect.addEventListener("change", () => {
  if (state.testing && activeWS && activeWS.readyState === WebSocket.OPEN) {
    activeWS.send(JSON.stringify({ model_name: els.testModeSelect.value }));
    log(`即時預測模型已切換至: ${els.testModeSelect.value}`);
  }
  updateFilterParameters();
});

// 當啟用/停用防抖動時，同步更新並記錄參數狀態
els.toggleAntiShake.addEventListener("change", () => {
  updateFilterParameters();
});

// 熱區圖開關事件
els.toggleHeatmap.addEventListener("change", () => {
  if (els.toggleHeatmap.checked) {
    if (state.testing) {
      els.heatmapOverlay.classList.remove("hidden");
      startHeatmapAnimation();
    }
  } else {
    els.heatmapOverlay.classList.add("hidden");
    if (heatmapAnimFrame) {
      cancelAnimationFrame(heatmapAnimFrame);
      heatmapAnimFrame = null;
    }
  }
});

// 視窗大小改變時重新調整目標位置
window.addEventListener("resize", () => {
  moveTarget(state.currentPoint || [0.5, 0.5]);
});

// 初始化
moveTarget([0.5, 0.5]);
checkHealth();
startCamera().catch((error) => log(error.message));
