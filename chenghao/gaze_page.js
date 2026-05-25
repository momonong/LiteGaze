const calibrationPoints = [
  [0.08, 0.10],
  [0.50, 0.10],
  [0.92, 0.10],
  [0.08, 0.50],
  [0.50, 0.50],
  [0.92, 0.50],
  [0.08, 0.90],
  [0.50, 0.90],
  [0.92, 0.90],
];

const state = {
  sessionId: "",
  collecting: false,
  testing: false,
};

const els = {
  healthText: document.getElementById("healthText"),
  participantName: document.getElementById("participantName"),
  repeatCount: document.getElementById("repeatCount"),
  collectBtn: document.getElementById("collectBtn"),
  collectMeta: document.getElementById("collectMeta"),
  datasetSelect: document.getElementById("datasetSelect"),
  modelName: document.getElementById("modelName"),
  refreshBtn: document.getElementById("refreshBtn"),
  trainBtn: document.getElementById("trainBtn"),
  modelSelect: document.getElementById("modelSelect"),
  testBtn: document.getElementById("testBtn"),
  stage: document.getElementById("stage"),
  target: document.getElementById("target"),
  gazeCursor: document.getElementById("gazeCursor"),
  video: document.getElementById("video"),
  canvas: document.getElementById("capture"),
  log: document.getElementById("log"),
};

function log(message) {
  const time = new Date().toLocaleTimeString("zh-TW", { hour12: false });
  els.log.textContent = `[${time}] ${message}\n${els.log.textContent}`.slice(0, 5000);
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  els.video.srcObject = stream;
  await els.video.play();
}

function captureFrame() {
  const width = 640;
  const aspect = els.video.videoHeight ? els.video.videoWidth / els.video.videoHeight : 4 / 3;
  const height = Math.round(width / aspect);
  els.canvas.width = width;
  els.canvas.height = height;
  const ctx = els.canvas.getContext("2d");
  ctx.drawImage(els.video, 0, 0, width, height);
  return els.canvas.toDataURL("image/jpeg", 0.8);
}

function pointToStage(point) {
  const rect = els.stage.getBoundingClientRect();
  return {
    x: point[0] * rect.width,
    y: point[1] * rect.height,
    pageX: rect.left + point[0] * rect.width,
    pageY: rect.top + point[1] * rect.height,
  };
}

function moveTarget(point) {
  const pos = pointToStage(point);
  els.target.style.left = `${pos.x}px`;
  els.target.style.top = `${pos.y}px`;
  return pos;
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/gaze/health");
    const data = await res.json();
    els.healthText.textContent = data.ok ? `已連線 (${data.mode})` : "後端異常";
  } catch (err) {
    els.healthText.textContent = `後端未連線: ${err.message}`;
  }
}

async function refreshDatasets() {
  const res = await fetch("/api/gaze/datasets");
  const data = await res.json();
  els.datasetSelect.innerHTML = "";
  if (!data.datasets.length) {
    els.datasetSelect.innerHTML = '<option value="">尚無資料集</option>';
    return;
  }
  data.datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = dataset.display_name;
    els.datasetSelect.appendChild(option);
  });
}

async function refreshModels() {
  const res = await fetch("/api/gaze/models");
  const data = await res.json();
  els.modelSelect.innerHTML = '<option value="before">原始模型 / before</option>';
  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.display_name;
    els.modelSelect.appendChild(option);
  });
}

async function createSession() {
  const participantId = els.participantName.value.trim() || "anonymous";
  const data = await postJson("/api/gaze/session", { participant_id: participantId });
  state.sessionId = data.session_id;
  els.collectMeta.textContent = `資料集: ${state.sessionId}`;
  log(`建立資料集 ${state.sessionId}`);
}

async function saveSample(point, pointIndex, repeatIndex) {
  const pos = moveTarget(point);
  await sleep(650);
  els.target.classList.add("capturing");
  const rect = els.stage.getBoundingClientRect();
  const targetXNorm = (pos.pageX / window.innerWidth) * 2 - 1;
  const targetYNorm = (pos.pageY / window.innerHeight) * 2 - 1;
  await postJson("/api/gaze/sample", {
    session_id: state.sessionId,
    image_data: captureFrame(),
    target_x: pos.pageX,
    target_y: pos.pageY,
    target_x_norm: targetXNorm,
    target_y_norm: targetYNorm,
    viewport_width: window.innerWidth,
    viewport_height: window.innerHeight,
    stage_width: rect.width,
    stage_height: rect.height,
    phase: "calibration",
    point_index: pointIndex,
    repeat_index: repeatIndex,
  });
  els.target.classList.remove("capturing");
}

async function collect() {
  if (state.collecting) {
    state.collecting = false;
    return;
  }
  state.collecting = true;
  els.collectBtn.textContent = "停止收集";
  await createSession();

  const repeats = Math.max(1, Math.min(5, Number.parseInt(els.repeatCount.value, 10) || 1));
  const total = calibrationPoints.length * repeats;
  let done = 0;
  try {
    for (let repeat = 0; repeat < repeats; repeat += 1) {
      for (let pointIndex = 0; pointIndex < calibrationPoints.length; pointIndex += 1) {
        if (!state.collecting) return;
        await saveSample(calibrationPoints[pointIndex], pointIndex, repeat);
        done += 1;
        els.collectMeta.textContent = `${state.sessionId} - ${done} / ${total}`;
        log(`收集 ${done}/${total}`);
        await sleep(140);
      }
    }
    log("收集完成，可以訓練模型");
    await refreshDatasets();
  } catch (err) {
    log(`收集失敗: ${err.message}`);
  } finally {
    state.collecting = false;
    els.collectBtn.textContent = "開始收集";
  }
}

async function train() {
  const datasetId = els.datasetSelect.value;
  const outputModelName = els.modelName.value.trim();
  if (!datasetId) {
    log("請先選擇資料集");
    return;
  }
  if (!outputModelName) {
    log("請輸入模型名稱");
    return;
  }

  els.trainBtn.disabled = true;
  try {
    const data = await postJson("/api/gaze/train", {
      data_session_id: datasetId,
      output_model_name: outputModelName,
    });
    log(`模型 ${data.model_name} 已建立 (${data.train_samples} samples)`);
    await refreshModels();
    els.modelSelect.value = data.model_name;
  } catch (err) {
    log(`訓練失敗: ${err.message}`);
  } finally {
    els.trainBtn.disabled = false;
  }
}

async function predictOnce() {
  const data = await postJson("/api/gaze/predict", {
    image_data: captureFrame(),
    model_name: els.modelSelect.value,
    viewport_width: window.innerWidth,
    viewport_height: window.innerHeight,
  });
  const x = data.screen_xy_px?.[0] ?? window.innerWidth / 2;
  const y = data.screen_xy_px?.[1] ?? window.innerHeight / 2;
  const rect = els.stage.getBoundingClientRect();
  els.gazeCursor.style.left = `${x - rect.left}px`;
  els.gazeCursor.style.top = `${y - rect.top}px`;
}

async function testLoop() {
  while (state.testing) {
    try {
      await predictOnce();
    } catch (err) {
      log(`測試失敗: ${err.message}`);
      state.testing = false;
    }
    await sleep(120);
  }
  els.gazeCursor.classList.add("hidden");
  els.testBtn.textContent = "開始測試";
}

function toggleTest() {
  state.testing = !state.testing;
  if (state.testing) {
    els.gazeCursor.classList.remove("hidden");
    els.testBtn.textContent = "停止測試";
    testLoop();
  }
}

els.collectBtn.addEventListener("click", () => collect());
els.refreshBtn.addEventListener("click", async () => {
  await refreshDatasets();
  await refreshModels();
  log("已重新整理資料集與模型");
});
els.trainBtn.addEventListener("click", () => train());
els.testBtn.addEventListener("click", () => toggleTest());

moveTarget([0.5, 0.5]);
refreshHealth();
refreshDatasets();
refreshModels();
startCamera().then(() => log("攝影機已就緒")).catch((err) => log(`攝影機啟動失敗: ${err.message}`));
