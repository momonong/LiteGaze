const calibrationPoints = [
  [0.08, 0.10], [0.50, 0.10], [0.92, 0.10],
  [0.08, 0.50], [0.50, 0.50], [0.92, 0.50],
  [0.08, 0.90], [0.50, 0.90], [0.92, 0.90],
  [0.29, 0.30], [0.71, 0.30],
  [0.29, 0.70], [0.71, 0.70],
];

const state = {
  sessionId: "",
  collecting: false,
  testing: false,
};

const renameTarget = { type: "", id: "" };
let _testLoopId = 0;
let _testAbort = null;

const els = {
  health: document.getElementById("health"),
  participantName: document.getElementById("participantName"),
  repeatCount: document.getElementById("repeatCount"),
  delayTime: document.getElementById("delayTime"),
  collectMode: document.getElementById("collectMode"),
  calibrationBtn: document.getElementById("calibration"),
  collectModal: document.getElementById("collectModal"),
  btnStartCollect: document.getElementById("btnStartCollect"),
  btnCancelCollect: document.getElementById("btnCancelCollect"),
  closeCollect: document.getElementById("closeCollect"),
  selectDataset: document.getElementById("selectDataset"),
  outputModelName: document.getElementById("outputModelName"),
  trainBtn: document.getElementById("train"),
  trainModal: document.getElementById("trainModal"),
  btnStartTrain: document.getElementById("btnStartTrain"),
  btnCancelTrain: document.getElementById("btnCancelTrain"),
  closeTrain: document.getElementById("closeTrain"),
  selectBaseModel: document.getElementById("selectBaseModel"),
  testBtn: document.getElementById("test"),
  stage: document.getElementById("stage"),
  target: document.getElementById("target"),
  gazeCursor: document.getElementById("gaze-cursor"),
  video: document.getElementById("video"),
  canvas: document.getElementById("capture"),
  log: document.getElementById("log"),
  session: document.getElementById("session"),
  phase: document.getElementById("phase"),
  progress: document.getElementById("progress"),
  testControls: document.getElementById("testControls"),
  testModeSelect: document.getElementById("testModeSelect"),
  toggleHeatmap: document.getElementById("toggleHeatmap"),
  toggleAntiShake: document.getElementById("toggleAntiShake"),
  toggleCorridorLock: document.getElementById("toggleCorridorLock"),
  btnFullscreen: document.getElementById("btnFullscreen"),
  toggleSettings: document.getElementById("toggle-settings"),

  modelsModal: document.getElementById("modelsModal"),
  closeModels: document.getElementById("closeModels"),
  btnCloseModels: document.getElementById("btnCloseModels"),
  datasetsList: document.getElementById("datasetsList"),
  modelsList: document.getElementById("modelsList"),
  renameDialog: document.getElementById("renameDialog"),
  renameInput: document.getElementById("renameInput"),
  btnConfirmRename: document.getElementById("btnConfirmRename"),
  btnCancelRename: document.getElementById("btnCancelRename"),
  closeRename: document.getElementById("closeRename"),
};

function showModal(modal) {
  modal.classList.remove("hidden");
}

function hideModal(modal) {
  modal.classList.add("hidden");
}

function log(message) {
  const time = new Date().toLocaleTimeString("zh-TW", { hour12: false });
  els.log.textContent = `[${time}] ${message}\n${els.log.textContent}`.slice(0, 5000);
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

async function postJson(url, body, signal) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
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

function captureFrame(quality) {
  const width = 320;
  const aspect = els.video.videoHeight ? els.video.videoWidth / els.video.videoHeight : 4 / 3;
  const height = Math.round(width / aspect);
  els.canvas.width = width;
  els.canvas.height = height;
  const ctx = els.canvas.getContext("2d");
  ctx.drawImage(els.video, 0, 0, width, height);
  return els.canvas.toDataURL("image/jpeg", quality || 0.5);
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
    els.health.className = `health-dot ${data.ok ? "ok" : "bad"}`;
  } catch (err) {
    els.health.className = "health-dot bad";
  }
}

async function refreshDatasets() {
  const res = await fetch("/api/gaze/datasets");
  const data = await res.json();
  els.selectDataset.innerHTML = "";
  if (!data.datasets.length) {
    els.selectDataset.innerHTML = '<option value="">尚無資料集</option>';
    return;
  }
  data.datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = dataset.display_name;
    els.selectDataset.appendChild(option);
  });
}

async function refreshModels() {
  const res = await fetch("/api/gaze/models");
  const data = await res.json();
  els.selectBaseModel.innerHTML = '<option value="before">原始模型 / before</option>';
  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.display_name;
    els.selectBaseModel.appendChild(option);
  });
}

async function refreshTestModels() {
  const res = await fetch("/api/gaze/models");
  const data = await res.json();
  els.testModeSelect.innerHTML = '<option value="before">原始模型</option>';
  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.display_name;
    els.testModeSelect.appendChild(option);
  });
}

async function refreshDatasetsList() {
  const res = await fetch("/api/gaze/datasets");
  const data = await res.json();
  els.datasetsList.innerHTML = "";
  if (!data.datasets.length) {
    els.datasetsList.innerHTML = '<div class="empty-list">尚無資料集</div>';
    return;
  }
  data.datasets.forEach((ds) => {
    const div = document.createElement("div");
    div.className = "model-item";
    div.innerHTML =
      '<div class="model-info">' +
        '<div class="model-info-name">' + escHtml(ds.participant) + '</div>' +
        '<div class="model-info-meta">' + escHtml(ds.id) + ' &middot; ' + ds.sample_count + ' samples</div>' +
      '</div>' +
      '<div class="model-actions">' +
        '<button class="btn-icon-sm" data-act="ren-ds" data-id="' + escHtml(ds.id) + '" data-participant="' + escHtml(ds.participant) + '"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg></button>' +
        '<button class="btn-icon-sm danger" data-act="del-ds" data-id="' + escHtml(ds.id) + '"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></button>' +
      '</div>';
    els.datasetsList.appendChild(div);
  });
}

async function refreshModelsList() {
  const res = await fetch("/api/gaze/models");
  const data = await res.json();
  els.modelsList.innerHTML = "";
  if (!data.models.length) {
    els.modelsList.innerHTML = '<div class="empty-list">尚無模型，請先訓練。</div>';
    return;
  }
  data.models.forEach((model) => {
    const div = document.createElement("div");
    div.className = "model-item";
    const meta = '誤差: ' + model.mean_px_error.toFixed(1) + ' px &middot; ' + model.train_samples + ' samples';
    div.innerHTML =
      '<div class="model-info">' +
        '<div class="model-info-name">' + escHtml(model.name) + '</div>' +
        '<div class="model-info-meta">' + meta + '</div>' +
      '</div>' +
      '<div class="model-actions">' +
        '<button class="btn-icon-sm" data-act="ren-md" data-id="' + escHtml(model.name) + '"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg></button>' +
        '<button class="btn-icon-sm danger" data-act="del-md" data-id="' + escHtml(model.name) + '"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></button>' +
      '</div>';
    els.modelsList.appendChild(div);
  });
}

async function createSession() {
  const participantId = els.participantName.value.trim() || "anonymous";
  const data = await postJson("/api/gaze/session", { participant_id: participantId });
  state.sessionId = data.session_id;
  els.session.textContent = state.sessionId;
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
  await createSession();
  state.collecting = true;
  els.calibrationBtn.disabled = true;
  els.phase.textContent = "收集中";

  els.target.style.zIndex = "99999";
  els.target.style.display = "";
  els.target.classList.remove("hidden");

  const repeats = Math.max(1, Math.min(5, Number.parseInt(els.repeatCount.value, 10) || 1));
  const total = calibrationPoints.length * repeats;
  let done = 0;
  try {
    for (let repeat = 0; repeat < repeats; repeat += 1) {
      for (let pointIndex = 0; pointIndex < calibrationPoints.length; pointIndex += 1) {
        if (!state.collecting) return;
        await saveSample(calibrationPoints[pointIndex], pointIndex, repeat);
        done += 1;
        els.progress.textContent = `${done} / ${total}`;
        log(`收集 ${done}/${total}`);
        await sleep(140);
      }
    }
    log("收集完成，可以訓練模型");
    els.phase.textContent = "收集完成";
    await refreshDatasets();
  } catch (err) {
    log(`收集失敗: ${err.message}`);
    els.phase.textContent = "收集失敗";
  } finally {
    state.collecting = false;
    els.calibrationBtn.disabled = false;
    els.target.style.zIndex = "";
  }
}

async function train() {
  const datasetId = els.selectDataset.value;
  const outputModelName = els.outputModelName.value.trim();
  if (!datasetId) {
    log("請先選擇資料集");
    return;
  }
  if (!outputModelName) {
    log("請輸入模型名稱");
    return;
  }

  els.trainBtn.disabled = true;
  els.phase.textContent = "訓練中";
  try {
    const data = await postJson("/api/gaze/train", {
      data_session_id: datasetId,
      output_model_name: outputModelName,
    });
    log(`模型 ${data.model_name} 已建立 (${data.train_samples} samples)`);
    els.phase.textContent = "訓練完成";
    els.target.classList.add("hidden");
    await refreshModels();
    els.selectBaseModel.value = data.model_name;
  } catch (err) {
    log(`訓練失敗: ${err.message}`);
    els.phase.textContent = "訓練失敗";
  } finally {
    els.trainBtn.disabled = false;
  }
}

async function predictOnce(signal) {
  const data = await postJson("/api/gaze/predict", {
    image_data: captureFrame(0.5),
    model_name: els.testModeSelect.value,
    viewport_width: window.innerWidth,
    viewport_height: window.innerHeight,
  }, signal);
  const x = data.screen_xy_px?.[0] ?? window.innerWidth / 2;
  const y = data.screen_xy_px?.[1] ?? window.innerHeight / 2;
  const rect = els.stage.getBoundingClientRect();
  els.gazeCursor.style.left = `${x - rect.left}px`;
  els.gazeCursor.style.top = `${y - rect.top}px`;
}

async function testLoop(id) {
  while (state.testing) {
    if (id !== _testLoopId) return;
    try {
      await predictOnce(_testAbort?.signal);
    } catch (err) {
      if (err.name === "AbortError") return;
    }
    if (id !== _testLoopId) return;
    await sleep(30);
  }
}

async function toggleTest() {
  state.testing = !state.testing;
  if (state.testing) {
    _testAbort = new AbortController();
    const id = ++_testLoopId;
    els.gazeCursor.classList.remove("hidden");
    els.testControls.classList.remove("hidden");
    els.testBtn.textContent = "停止測試";
    els.phase.textContent = "測試中";
    await refreshTestModels();
    testLoop(id);
  } else {
    if (_testAbort) _testAbort.abort();
    els.gazeCursor.classList.add("hidden");
    els.testControls.classList.add("hidden");
    els.testBtn.textContent = "測試";
    els.phase.textContent = "待命";
  }
}

// Collect modal
els.calibrationBtn.addEventListener("click", () => {
  if (state.collecting) {
    state.collecting = false;
    return;
  }
  showModal(els.collectModal);
});

els.btnStartCollect.addEventListener("click", () => {
  hideModal(els.collectModal);
  collect();
});

els.btnCancelCollect.addEventListener("click", () => hideModal(els.collectModal));
els.closeCollect.addEventListener("click", () => hideModal(els.collectModal));
els.collectModal.addEventListener("click", (e) => {
  if (e.target === els.collectModal) hideModal(els.collectModal);
});

// Train modal
els.trainBtn.addEventListener("click", () => {
  refreshDatasets();
  showModal(els.trainModal);
});

els.btnStartTrain.addEventListener("click", () => {
  hideModal(els.trainModal);
  train();
});

els.btnCancelTrain.addEventListener("click", () => hideModal(els.trainModal));
els.closeTrain.addEventListener("click", () => hideModal(els.trainModal));
els.trainModal.addEventListener("click", (e) => {
  if (e.target === els.trainModal) hideModal(els.trainModal);
});

// Test toggle
els.testBtn.addEventListener("click", () => toggleTest());

// Models modal
els.toggleSettings.addEventListener("click", () => {
  refreshDatasetsList();
  refreshModelsList();
  showModal(els.modelsModal);
});

els.closeModels.addEventListener("click", () => hideModal(els.modelsModal));
els.btnCloseModels.addEventListener("click", () => hideModal(els.modelsModal));
els.modelsModal.addEventListener("click", (e) => {
  if (e.target === els.modelsModal) hideModal(els.modelsModal);
});

// Fullscreen
els.btnFullscreen.addEventListener("click", () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen();
  } else {
    document.exitFullscreen();
  }
});

// Model center: delete / rename actions
els.datasetsList.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const act = btn.dataset.act;
  const id = btn.dataset.id;
  if (act === "del-ds") {
    if (!confirm("刪除資料集 " + id + "？")) return;
    try {
      const res = await fetch("/api/gaze/datasets/" + encodeURIComponent(id), { method: "DELETE" });
      const d = await res.json();
      if (!d.ok) throw new Error(d.error);
      log("資料集已刪除");
    } catch (err) { log("刪除失敗: " + err.message); }
    await refreshDatasetsList();
  } else if (act === "ren-ds") {
    renameTarget.type = "dataset";
    renameTarget.id = id;
    els.renameInput.value = btn.dataset.participant || id;
    showModal(els.renameDialog);
  }
});

els.modelsList.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const act = btn.dataset.act;
  const id = btn.dataset.id;
  if (act === "del-md") {
    if (!confirm("刪除模型 " + id + "？")) return;
    try {
      const res = await fetch("/api/gaze/models/" + encodeURIComponent(id), { method: "DELETE" });
      const d = await res.json();
      if (!d.ok) throw new Error(d.error);
      log("模型已刪除");
    } catch (err) { log("刪除失敗: " + err.message); }
    await refreshModelsList();
  } else if (act === "ren-md") {
    renameTarget.type = "model";
    renameTarget.id = id;
    els.renameInput.value = id;
    showModal(els.renameDialog);
  }
});

// Rename dialog
els.btnConfirmRename.addEventListener("click", async () => {
  const newName = els.renameInput.value.trim();
  if (!newName) return;
  hideModal(els.renameDialog);
  const endpoint = renameTarget.type === "dataset"
    ? "/api/gaze/datasets/" + encodeURIComponent(renameTarget.id)
    : "/api/gaze/models/" + encodeURIComponent(renameTarget.id);
  try {
    const res = await fetch(endpoint, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_name: newName }),
    });
    const d = await res.json();
    if (!d.ok) throw new Error(d.error);
    log("重新命名成功");
  } catch (err) { log("重新命名失敗: " + err.message); }
  refreshDatasetsList();
  refreshModelsList();
  renameTarget.type = "";
  renameTarget.id = "";
});

els.btnCancelRename.addEventListener("click", () => hideModal(els.renameDialog));
els.closeRename.addEventListener("click", () => hideModal(els.renameDialog));
els.renameDialog.addEventListener("click", (e) => {
  if (e.target === els.renameDialog) hideModal(els.renameDialog);
});

moveTarget([0.5, 0.5]);
refreshHealth();
refreshDatasets();
refreshModels();
startCamera().then(() => log("攝影機已就緒")).catch((err) => log(`攝影機啟動失敗: ${err.message}`));
