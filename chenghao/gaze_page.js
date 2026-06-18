const calibrationPoints = [
  [0.08, 0.10], [0.50, 0.10], [0.92, 0.10],
  [0.08, 0.50], [0.50, 0.50], [0.92, 0.50],
  [0.08, 0.90], [0.50, 0.90], [0.92, 0.90],
  [0.29, 0.30], [0.71, 0.30],
  [0.29, 0.70], [0.71, 0.70],
];

class LowPassFilter {
  constructor(alpha) { this.alpha = alpha; this.value = null; }
  filter(value) {
    if (this.value === null) { this.value = value; return value; }
    this.value = this.value + this.alpha * (value - this.value);
    return this.value;
  }
  reset() { this.value = null; }
}

const state = {
  sessionId: "",
  collecting: false,
  testing: false,
  heatmap: false,
  antiShake: true,
  corridorLock: false,
  filterX: new LowPassFilter(0.28),
  filterY: new LowPassFilter(0.28),
  _lockedY: null,
  mediaRecorder: null,
  recordedBlobs: [],
  timelineTargets: [],
  recordingStartTime: 0,
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
  heatmapCanvas: document.getElementById("heatmap-overlay"),
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
  recordVideo: document.getElementById("recordVideo"),
  uploadVideoFile: document.getElementById("uploadVideoFile"),
  uploadTimelineFile: document.getElementById("uploadTimelineFile"),
  btnUploadOffline: document.getElementById("btnUploadOffline"),
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

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function saveSample(point, pointIndex, repeatIndex) {
  const pos = moveTarget(point);
  await sleep(650);
  els.target.classList.add("capturing");
  const rect = els.stage.getBoundingClientRect();
  const targetXNorm = (pos.pageX / window.innerWidth) * 2 - 1;
  const targetYNorm = (pos.pageY / window.innerHeight) * 2 - 1;
  
  if (els.recordVideo && els.recordVideo.checked && state.recordingStartTime > 0) {
    const timeOffsetMs = performance.now() - state.recordingStartTime;
    state.timelineTargets.push({
      timestamp_ms: timeOffsetMs,
      target_x: pos.pageX,
      target_y: pos.pageY,
      target_x_norm: targetXNorm,
      target_y_norm: targetYNorm,
      point_index: pointIndex,
      repeat_index: repeatIndex,
      phase: "calibration",
      screen_width: window.innerWidth,
      screen_height: window.innerHeight
    });
  }

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

  const recordVideo = els.recordVideo && els.recordVideo.checked;
  if (recordVideo) {
    state.recordedBlobs = [];
    state.timelineTargets = [];
    const options = { mimeType: 'video/webm;codecs=vp8,opus' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options.mimeType = 'video/webm';
    }
    try {
      state.mediaRecorder = new MediaRecorder(els.video.srcObject, options);
    } catch (e) {
      console.error('Exception while creating MediaRecorder:', e);
      state.mediaRecorder = new MediaRecorder(els.video.srcObject);
    }
    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        state.recordedBlobs.push(event.data);
      }
    };
    state.mediaRecorder.start(10);
    state.recordingStartTime = performance.now();
    log("已啟動鏡頭影片錄製...");
  } else {
    state.recordingStartTime = 0;
  }

  const repeats = Math.max(1, Math.min(5, Number.parseInt(els.repeatCount.value, 10) || 1));
  const total = calibrationPoints.length * repeats;
  let done = 0;
  try {
    for (let repeat = 0; repeat < repeats; repeat += 1) {
      for (let pointIndex = 0; pointIndex < calibrationPoints.length; pointIndex += 1) {
        if (!state.collecting) break;
        await saveSample(calibrationPoints[pointIndex], pointIndex, repeat);
        done += 1;
        els.progress.textContent = `${done} / ${total}`;
        log(`收集 ${done}/${total}`);
        await sleep(140);
      }
    }
    
    if (recordVideo && state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
      state.mediaRecorder.stop();
      state.mediaRecorder.onstop = async () => {
        els.phase.textContent = "影片校正中...";
        log("正在上傳並處理校正影片...");
        
        const videoBlob = new Blob(state.recordedBlobs, { type: "video/webm" });
        const participantId = els.participantName.value.trim() || "anonymous";
        const timelineData = {
          participant_id: participantId,
          viewport_width: window.innerWidth,
          viewport_height: window.innerHeight,
          targets: state.timelineTargets
        };
        
        const formData = new FormData();
        formData.append("video", videoBlob, "calibration.webm");
        formData.append("timeline", JSON.stringify(timelineData));
        
        try {
          const res = await fetch("/api/demo/upload_video", {
            method: "POST",
            body: formData
          });
          const result = await res.json();
          if (result.ok) {
            log(`影片校正完成！Session ID: ${result.session_id}`);
            log(`已擷取 ${result.processed_samples} 幀，失敗 ${result.failed_samples} 幀`);
            log(`已自動訓練個人化影片模型: ${result.model_name}`);
            
            // Backup download
            downloadBlob(videoBlob, `${participantId}_calibration.webm`);
            downloadBlob(new Blob([JSON.stringify(timelineData, null, 2)], { type: "application/json" }), `${participantId}_timeline.json`);
            log("已為您下載影片與時間軸 JSON 檔案作為本地備份。");
            
            els.phase.textContent = "影片校正成功";
            await refreshDatasets();
            await refreshModels();
            els.selectBaseModel.value = result.model_name;
          } else {
            log(`影片校正失敗: ${result.error}`);
            els.phase.textContent = "影片校正失敗";
          }
        } catch (err) {
          log(`影片校正上傳失敗: ${err.message}`);
          els.phase.textContent = "處理失敗";
        }
      };
    } else {
      log("收集完成，可以訓練模型");
      els.phase.textContent = "收集完成";
      await refreshDatasets();
    }
  } catch (err) {
    log(`收集失敗: ${err.message}`);
    els.phase.textContent = "收集失敗";
    if (recordVideo && state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
      state.mediaRecorder.stop();
    }
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
    image_data: captureFrame(),
    model_name: els.testModeSelect.value,
    viewport_width: window.innerWidth,
    viewport_height: window.innerHeight,
  }, signal);
  let x = data.screen_xy_px?.[0] ?? window.innerWidth / 2;
  let y = data.screen_xy_px?.[1] ?? window.innerHeight / 2;

  if (state.antiShake) {
    x = state.filterX.filter(x);
    y = state.filterY.filter(y);
  } else {
    state.filterX.reset();
    state.filterY.reset();
  }

  if (state.corridorLock) {
    const corridorH = 50;
    if (state._lockedY === null) {
      state._lockedY = y;
    } else if (Math.abs(y - state._lockedY) <= corridorH) {
      y = state._lockedY;
    } else {
      state._lockedY = y;
    }
  } else {
    state._lockedY = null;
  }

  if (state.heatmap) {
    drawHeatmapPoint(x, y);
  }

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

function drawHeatmapPoint(x, y) {
  const canvas = els.heatmapCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "rgba(255, 100, 50, 0.12)";
    ctx.beginPath();
    ctx.arc(x, y, 18, 0, Math.PI * 2);
    ctx.fill();
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

async function uploadExistingVideo() {
  const videoFile = els.uploadVideoFile.files[0];
  const timelineFile = els.uploadTimelineFile.files[0];
  
  if (!videoFile || !timelineFile) {
    log("請先選擇離線校正影片檔與時間軸 JSON 檔");
    alert("請先選擇影片與時間軸 JSON 檔案！");
    return;
  }
  
  hideModal(els.collectModal);
  els.phase.textContent = "上傳離線檔案...";
  log("正在上傳並處理離線校正影片與時間軸...");
  
  const reader = new FileReader();
  reader.onload = async (e) => {
    try {
      const timelineText = e.target.result;
      const timeline = JSON.parse(timelineText);
      
      const formData = new FormData();
      formData.append("video", videoFile);
      formData.append("timeline", JSON.stringify(timeline));
      
      const res = await fetch("/api/demo/upload_video", {
        method: "POST",
        body: formData
      });
      const result = await res.json();
      if (result.ok) {
        log(`離線影片校正成功！Session ID: ${result.session_id}`);
        log(`已從影片擷取 ${result.processed_samples} 幀，失敗 ${result.failed_samples} 幀`);
        log(`已自動訓練個人化影片模型: ${result.model_name}`);
        
        els.phase.textContent = "離線校正成功";
        await refreshDatasets();
        await refreshModels();
        els.selectBaseModel.value = result.model_name;
      } else {
        log(`離線影片校正失敗: ${result.error}`);
        els.phase.textContent = "離線校正失敗";
      }
    } catch (err) {
      log(`離線校正上傳處理失敗: ${err.message}`);
      els.phase.textContent = "處理失敗";
    }
  };
  reader.readAsText(timelineFile);
}

els.btnUploadOffline.addEventListener("click", () => {
  uploadExistingVideo();
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

// Test control checkboxes
els.toggleHeatmap.addEventListener("change", () => {
  state.heatmap = els.toggleHeatmap.checked;
  const canvas = els.heatmapCanvas;
  if (!state.heatmap && canvas) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }
  if (canvas) {
    canvas.classList.toggle("hidden", !state.heatmap);
    if (state.heatmap) {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
  }
});

els.toggleAntiShake.addEventListener("change", () => {
  state.antiShake = els.toggleAntiShake.checked;
  state.filterX.reset();
  state.filterY.reset();
});

els.toggleCorridorLock.addEventListener("change", () => {
  state.corridorLock = els.toggleCorridorLock.checked;
  state._lockedY = null;
});

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
