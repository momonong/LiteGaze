const STUDY_KEY = "lexigaze.participantStudy.v1";

const ui = Object.fromEntries([
  "alert", "visitMeta", "participantId", "setupPanel", "setupStatus", "profileForm",
  "englishL1", "englishAoa", "weeklyReading", "visionCorrection", "educationBand",
  "privateSpace", "evenLighting", "comfortableDistance", "cameraPreview", "cameraStatus",
  "deviceSummary", "checkCameraBtn", "saveSetupBtn", "calibrationPanel", "practicePanel",
  "practicePassage", "practiceForm", "practiceBtn", "validationPanel", "validationTitle",
  "validationProgress", "validationBtn", "readingPanel", "roundLabel", "difficulty", "timer",
  "passage", "readingHint", "videoStatus", "startReadingBtn", "finishReadingBtn", "reviewPanel", "reviewForm",
  "understanding", "mentalEffort", "readComplete", "interrupted", "submitRoundBtn",
  "completePanel", "qualitySummary", "captureCanvas", "liveStatus", "targetOverlay",
  "validationTarget", "geometryQualityNotice",
].map((id) => [id, document.getElementById(id)]));

const state = {
  context: null,
  session: null,
  protocol: null,
  gazeMeasurementContract: null,
  practice: null,
  stream: null,
  modelName: "",
  device: null,
  current: null,
  readingStartedAt: 0,
  readingTimer: null,
  sampling: false,
  samplingPromise: null,
  telemetryQueue: [],
  wordLayout: [],
  scrollOrigin: 0,
  scrollOccurred: false,
  resumedRound: false,
  mediaRecorder: null,
  videoChunks: [],
  videoStopPromise: null,
  videoRecordingId: "",
  videoMimeType: "",
  pendingReadingVideo: null,
  readingVideoUploaded: false,
  readingFinishedElapsed: null,
  captureContract: null,
};

const READING_VIDEO_SCOPE = "retain_reading_video_self_development";
const READING_VIDEO_MAX_BYTES = 64 * 1024 * 1024;
const READING_VIDEO_BITS_PER_SECOND = 750_000;

function showAlert(message) {
  ui.alert.textContent = message;
  ui.alert.classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function clearAlert() {
  ui.alert.textContent = "";
  ui.alert.classList.add("hidden");
}

function announce(message) {
  ui.liveStatus.textContent = message;
}

function readContext() {
  try {
    return JSON.parse(sessionStorage.getItem(STUDY_KEY) || "null");
  } catch (_) {
    return null;
  }
}

function draftKey() {
  return `lexigaze.generalCollectionDraft.${state.context.study_session_id}`;
}

function practiceKey() {
  return `lexigaze.generalCollectionPractice.${state.context.study_session_id}`;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.context?.access_token) {
    headers.set("Authorization", `Bearer ${state.context.access_token}`);
    headers.set("X-Lexigaze-Study-Session", state.context.study_session_id);
  }
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function showOnly(panel) {
  [
    ui.setupPanel, ui.calibrationPanel, ui.practicePanel, ui.validationPanel,
    ui.readingPanel, ui.reviewPanel, ui.completePanel,
  ].forEach((item) => item.classList.toggle("hidden", item !== panel));
}

function browserFamily() {
  const agent = navigator.userAgent;
  if (/Edg\//.test(agent)) return "edge_chromium";
  if (/Firefox\//.test(agent)) return "firefox";
  if (/Chrome\//.test(agent) || /Chromium\//.test(agent)) return "chromium";
  if (/Safari\//.test(agent) && !/Chrome\//.test(agent)) return "safari";
  return "other";
}

function dprBucket() {
  const value = window.devicePixelRatio || 1;
  if (value < 1.25) return "under_1_25";
  if (value < 1.75) return "1_25_1_74";
  if (value < 2.5) return "1_75_2_49";
  return "2_5_plus";
}

function fpsBand(frameRate) {
  if (!Number.isFinite(frameRate)) return "unknown";
  if (frameRate < 15) return "under_15";
  if (frameRate < 24) return "15_23";
  if (frameRate < 31) return "24_30";
  return "over_30";
}

async function ensureCamera() {
  if (state.stream) return;
  if (!(window.isSecureContext || ["localhost", "127.0.0.1"].includes(location.hostname))) {
    throw new Error("此流程只允許 HTTPS 或 localhost 使用相機。");
  }
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("瀏覽器不支援相機 API。");
  if (!globalThis.LexiGazeCapture) throw new Error("相機 capture contract 元件未載入。");
  state.stream = await navigator.mediaDevices.getUserMedia(
    globalThis.LexiGazeCapture.mediaConstraints(),
  );
  ui.cameraPreview.srcObject = state.stream;
  await ui.cameraPreview.play();
  const settings = state.stream.getVideoTracks()[0].getSettings();
  state.captureContract = globalThis.LexiGazeCapture.captureContract(ui.cameraPreview);
  const width = state.captureContract.source_width_px;
  const height = state.captureContract.source_height_px;
  if (width < 640 || height < 480) throw new Error(`相機解析度 ${width}×${height} 低於 640×480。`);
  state.device = {
    device_class: /Mobi|Android/i.test(navigator.userAgent) ? "mobile" : "desktop",
    browser_family: browserFamily(),
    viewport_width: window.innerWidth,
    viewport_height: window.innerHeight,
    device_pixel_ratio_bucket: dprBucket(),
    camera_width: width,
    camera_height: height,
    estimated_camera_fps_band: fpsBand(state.captureContract.source_frame_rate_hz),
  };
  ui.cameraStatus.textContent = "相機已就緒";
  ui.deviceSummary.textContent = `${width}×${height} @ ${state.captureContract.source_frame_rate_hz || "?"} fps · 傳輸 ${state.captureContract.transport_width_px}×${state.captureContract.transport_height_px}（維持比例） · ${state.device.browser_family}`;
}

async function ensureCameraAndModel() {
  await ensureCamera();
  if (state.modelName) return;
  const result = await api("/api/gaze/models");
  const linked = state.session?.linked_data?.model_name || state.context.model_name;
  const model = (result.models || []).find((item) => item.name === linked);
  if (!model) throw new Error("找不到這次校正產生的匿名模型，請返回重新校正。");
  state.modelName = model.name;
}

function stopCamera() {
  state.sampling = false;
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    try { state.mediaRecorder.stop(); } catch (_) { /* Page teardown is best-effort. */ }
  }
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  ui.cameraPreview.srcObject = null;
}

async function checkCamera() {
  clearAlert();
  ui.checkCameraBtn.disabled = true;
  try {
    await ensureCamera();
    ui.saveSetupBtn.disabled = false;
  } catch (error) {
    stopCamera();
    showAlert(error.message);
  } finally {
    ui.checkCameraBtn.disabled = false;
  }
}

function profilePayload() {
  return {
    english_l1: ui.englishL1.value,
    english_age_of_acquisition_band: ui.englishAoa.value,
    weekly_english_reading_band: ui.weeklyReading.value,
    vision_correction: ui.visionCorrection.value,
    education_band: ui.educationBand.value,
  };
}

async function saveSetup() {
  clearAlert();
  if (!ui.profileForm.reportValidity()) return;
  if (![ui.privateSpace, ui.evenLighting, ui.comfortableDistance].every((item) => item.checked)) {
    showAlert("請確認隱私、光線與裝置距離三個條件。");
    return;
  }
  if (!state.device) {
    showAlert("請先完成相機檢查。");
    return;
  }
  const checks = {
    secure_context: window.isSecureContext || ["localhost", "127.0.0.1"].includes(location.hostname),
    camera_api: Boolean(navigator.mediaDevices?.getUserMedia),
    camera_permission: Boolean(state.stream),
    minimum_viewport: innerWidth >= 1024 && innerHeight >= 700,
    minimum_camera_resolution: state.device.camera_width >= 640 && state.device.camera_height >= 480,
    document_visible: document.visibilityState === "visible",
    stable_network: navigator.onLine,
    single_participant_private_space: ui.privateSpace.checked,
    face_evenly_lit: ui.evenLighting.checked,
    screen_at_comfortable_distance: ui.comfortableDistance.checked,
  };
  if (Object.values(checks).some((value) => value !== true)) {
    const failed = Object.entries(checks).filter(([, value]) => !value).map(([key]) => key);
    showAlert(`系統條件未通過：${failed.join("、")}`);
    return;
  }
  ui.saveSetupBtn.disabled = true;
  try {
    await api(`/api/study/sessions/${state.context.study_session_id}/general/profile`, {
      method: "POST",
      body: JSON.stringify({ profile: profilePayload() }),
    });
    const result = await api(`/api/study/sessions/${state.context.study_session_id}/general/system-check`, {
      method: "POST",
      body: JSON.stringify({ checks, device: state.device }),
    });
    state.session = result.session;
    stopCamera();
    showOnly(ui.calibrationPanel);
  } catch (error) {
    showAlert(error.message);
    ui.saveSetupBtn.disabled = false;
  }
}

function renderText(container, text) {
  container.replaceChildren();
  const pattern = /\b[\w'-]+\b/g;
  let cursor = 0;
  let wordIndex = 0;
  for (const match of text.matchAll(pattern)) {
    container.append(document.createTextNode(text.slice(cursor, match.index)));
    const span = document.createElement("span");
    span.className = "assessment-word";
    span.dataset.wordIndex = String(wordIndex++);
    span.textContent = match[0];
    container.append(span);
    cursor = match.index + match[0].length;
  }
  container.append(document.createTextNode(text.slice(cursor)));
}

function reviewRow(probe, groupPrefix) {
  const row = document.createElement("div");
  row.className = "review-row";
  const word = document.createElement("span");
  word.className = "review-word";
  word.textContent = probe.surface;
  row.append(word);
  const choices = [
    ["no_review", "不需要回顧"],
    ["unsure", "不確定"],
    ["review_needed", "需要回顧"],
  ];
  for (const [value, labelText] of choices) {
    const label = document.createElement("label");
    label.className = "review-choice";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = `${groupPrefix}-${probe.probe_id}`;
    input.value = value;
    label.append(input, document.createTextNode(labelText));
    row.append(label);
  }
  return row;
}

function showPractice() {
  showOnly(ui.practicePanel);
  renderText(ui.practicePassage, state.practice.text);
  ui.practiceForm.replaceChildren(...state.practice.probes.map((probe) => reviewRow(probe, "practice")));
}

async function finishPractice() {
  for (const probe of state.practice.probes) {
    if (!ui.practiceForm.querySelector(`input[name="practice-${CSS.escape(probe.probe_id)}"]:checked`)) {
      showAlert("請完成三個練習單字的選擇。");
      return;
    }
  }
  sessionStorage.setItem(practiceKey(), "done");
  await beginCollection();
}

async function beginCollection() {
  clearAlert();
  try {
    await ensureCameraAndModel();
    const result = await api(`/api/study/sessions/${state.context.study_session_id}/general/start`, {
      method: "POST",
      body: JSON.stringify({
        assessment_viewport: {
          width_px: window.innerWidth,
          height_px: window.innerHeight,
        },
      }),
    });
    state.session = result.session;
    state.gazeMeasurementContract = state.session.general_collection
      ?.gaze_measurement_contract?.contract || null;
    assertAssessmentViewportStable();
    await routeCollectionPhase();
  } catch (error) {
    showAlert(error.message);
  }
}

function frozenAssessmentViewport() {
  const raw = state.session?.general_collection?.assessment_viewport;
  const width = Number(raw?.width_px);
  const height = Number(raw?.height_px);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error("找不到 server-frozen assessment viewport；gaze 已停用，請重新執行 system check。");
  }
  return { width_px: width, height_px: height };
}

function assertAssessmentViewportStable() {
  const viewport = frozenAssessmentViewport();
  if (window.innerWidth !== viewport.width_px || window.innerHeight !== viewport.height_px) {
    const error = new Error(
      `Assessment viewport 已由 ${viewport.width_px}×${viewport.height_px} 變為 ${window.innerWidth}×${window.innerHeight}；本輪 gaze 已中止，請恢復原尺寸或重新執行 system check。`,
    );
    error.code = "assessment_viewport_changed";
    throw error;
  }
  return viewport;
}

function frameSnapshot() {
  const snapshot = globalThis.LexiGazeCapture.captureSnapshot(
    ui.cameraPreview,
    ui.captureCanvas,
  );
  state.captureContract = snapshot.capture_contract;
  return snapshot;
}

async function predictFrame() {
  const viewport = assertAssessmentViewportStable();
  const snapshot = frameSnapshot();
  return api("/api/gaze/predict", {
    method: "POST",
    body: JSON.stringify({
      image_data: snapshot.image_data,
      capture_contract: snapshot.capture_contract,
      model_name: state.modelName,
      viewport_width: viewport.width_px,
      viewport_height: viewport.height_px,
      study_session_id: state.context.study_session_id,
      study_access_token: state.context.access_token,
      allow_cuda: false,
    }),
  });
}

const delay = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

async function runValidation() {
  clearAlert();
  ui.validationBtn.disabled = true;
  const phase = state.session.general_collection.phase === "end_validation_required" ? "end" : "start";
  try {
    await ensureCameraAndModel();
    const viewport = assertAssessmentViewportStable();
    const points = state.gazeMeasurementContract?.target_independence
      ?.selected_validation_targets;
    if (!Array.isArray(points) || points.length !== 5) {
      throw new Error("找不到 frozen five-point held-out validation contract。");
    }
    const samples = [];
    ui.targetOverlay.classList.remove("hidden");
    for (const point of points) {
      const targetId = String(point.target_id || "");
      const xFraction = Number(point.target_x_viewport_fraction);
      const yFraction = Number(point.target_y_viewport_fraction);
      const targetXNorm = Number(point.target_x_norm);
      const targetYNorm = Number(point.target_y_norm);
      if (!targetId || ![xFraction, yFraction, targetXNorm, targetYNorm].every(Number.isFinite)) {
        throw new Error("Frozen validation target contract 格式錯誤。");
      }
      const targetX = Math.round(viewport.width_px * xFraction);
      const targetY = Math.round(viewport.height_px * yFraction);
      ui.validationTarget.style.left = `${targetX}px`;
      ui.validationTarget.style.top = `${targetY}px`;
      await delay(700);
      for (let repeat = 0; repeat < 3; repeat += 1) {
        try {
          const result = await predictFrame();
          samples.push({
            target_id: targetId,
            target_x_px: targetX,
            target_y_px: targetY,
            target_x_norm: targetXNorm,
            target_y_norm: targetYNorm,
            prediction_success: true,
            predicted_x_px: result.screen_xy_px[0],
            predicted_y_px: result.screen_xy_px[1],
          });
        } catch (error) {
          if (error?.code === "assessment_viewport_changed") throw error;
          samples.push({
            target_id: targetId,
            target_x_px: targetX,
            target_y_px: targetY,
            target_x_norm: targetXNorm,
            target_y_norm: targetYNorm,
            prediction_success: false,
          });
        }
        ui.validationProgress.textContent = `${samples.length} / 15`;
        await delay(250);
      }
    }
    ui.targetOverlay.classList.add("hidden");
    const result = await api(`/api/study/sessions/${state.context.study_session_id}/general/validation`, {
      method: "POST",
      body: JSON.stringify({
        phase,
        samples,
        capture_contract: state.captureContract,
      }),
    });
    state.session = result.session;
    await routeCollectionPhase();
  } catch (error) {
    ui.targetOverlay.classList.add("hidden");
    showAlert(error.message);
  } finally {
    ui.validationBtn.disabled = false;
  }
}

function formatElapsed(milliseconds) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function renderRound(result) {
  state.current = result;
  state.resumedRound = result.phase === "reading_active" && Boolean(sessionStorage.getItem(draftKey()));
  state.mediaRecorder = null;
  state.videoChunks = [];
  state.videoStopPromise = null;
  state.videoRecordingId = "";
  state.videoMimeType = "";
  state.pendingReadingVideo = null;
  state.readingVideoUploaded = false;
  state.readingFinishedElapsed = null;
  showOnly(ui.readingPanel);
  ui.roundLabel.textContent = `文章 ${result.round_number} / ${result.round_count}`;
  ui.difficulty.textContent = result.passage.difficulty_band;
  ui.timer.textContent = "00:00";
  ui.readingHint.textContent = "按下開始後才計時；至少閱讀 20 秒。";
  renderProvisionalGeometryQuality();
  if (readingVideoEnabled()) {
    ui.videoStatus.textContent = "本篇開始閱讀後會同步錄製無聲 webcam 影片；只用於你的系統開發資料。";
    ui.videoStatus.classList.remove("hidden");
  } else {
    ui.videoStatus.textContent = "";
    ui.videoStatus.classList.add("hidden");
  }
  ui.startReadingBtn.classList.remove("hidden");
  ui.startReadingBtn.disabled = false;
  ui.finishReadingBtn.classList.add("hidden");
  ui.finishReadingBtn.disabled = true;
  renderText(ui.passage, result.passage.text);
  ui.passage.classList.add("reading");
  window.scrollTo({ top: ui.readingPanel.offsetTop - 82, behavior: "smooth" });
}

function captureWordLayout() {
  return [...ui.passage.querySelectorAll(".assessment-word")].map((word, wordIndex) => {
    const box = word.getBoundingClientRect();
    return {
      word_index: wordIndex,
      left_px: box.left,
      top_px: box.top,
      right_px: box.right,
      bottom_px: box.bottom,
    };
  });
}

function nearestWord(x, y) {
  let bestIndex = null;
  let bestDistance = 100;
  for (const item of state.wordLayout) {
    const dx = x - (item.left_px + item.right_px) / 2;
    const dy = y - (item.top_px + item.bottom_px) / 2;
    const distance = Math.hypot(dx, dy);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = item.word_index;
    }
  }
  return bestIndex;
}

function coarseFailure(error) {
  const message = String(error?.message || "").toLowerCase();
  if (message.includes("no face")) return "no_face";
  if (message.includes("timeout")) return "timeout";
  if (message.includes("capture contract")) return "capture_contract_mismatch";
  if (!navigator.onLine) return "network_error";
  return "prediction_failed";
}

function renderProvisionalGeometryQuality() {
  const quality = state.session?.general_collection?.provisional_geometry_quality;
  const integrity = state.session?.general_collection?.gaze_integrity;
  ui.geometryQualityNotice.className = "quality-notice hidden";
  ui.geometryQualityNotice.textContent = "";
  if (integrity?.eligible === false) {
    ui.geometryQualityNotice.classList.add("degraded");
    ui.geometryQualityNotice.textContent = "閱讀 viewport 或 reading segment 已中斷；本次 gaze 已永久降級為 behavioral-only，但仍可完成閱讀與 word-review。";
    ui.geometryQualityNotice.classList.remove("hidden");
    return;
  }
  if (!quality) return;
  const mode = quality.recommended_gaze_mode;
  const captureContractStatus = String(quality.capture_contract_status || "unavailable");
  const contractMismatch = quality.capture_contract_compatible === false;
  const captureContractUnavailable = !contractMismatch
    && (captureContractStatus !== "compatible" || quality.capture_contract_compatible !== true);
  const targetIndependenceStatus = String(quality.target_independence_status || "unavailable");
  const independenceFailed = targetIndependenceStatus === "failed"
    || quality.validation_targets_independent === false;
  const independenceUnavailable = !independenceFailed
    && (targetIndependenceStatus !== "passed" || quality.validation_targets_independent !== true);
  if (mode === "word_level_candidate") {
    ui.geometryQualityNotice.textContent = "閱讀前 sensor-only 暫定結果：幾何達到 rehearsal 的 word-level candidate 描述帶。最終品質仍需閱讀後驗證，這不是正式通過門檻。";
  } else if (mode === "passage_level_only") {
    ui.geometryQualityNotice.classList.add("coarse");
    ui.geometryQualityNotice.textContent = "閱讀前 sensor-only 暫定結果：眼動只建議用於 passage-level 描述，不應解讀為精確逐字注視。你仍可繼續閱讀；若需要逐字 gaze，建議先重新校準。";
  } else {
    ui.geometryQualityNotice.classList.add("degraded");
    const blockers = [];
    if (contractMismatch) {
      blockers.push("相機 capture contract 與校準不一致。");
    } else if (captureContractUnavailable) {
      blockers.push("缺少可驗證的 calibration camera capture provenance，capture compatibility 無法證明。");
    }
    if (independenceFailed) {
      blockers.push("held-out targets 與實際 calibration fit targets 重疊，target-independence 驗證失敗。");
    } else if (independenceUnavailable) {
      blockers.push("缺少可驗證的 calibration fit-target provenance，target independence 無法證明。");
    }
    if (blockers.length) {
      ui.geometryQualityNotice.textContent = `閱讀前檢查：${blockers.join(" ")}Gaze 應停用，但本次仍可繼續收集 behavioral word-review；若要使用 gaze，請重新校準後再驗證。`;
    } else {
      ui.geometryQualityNotice.textContent = "閱讀前 sensor-only 暫定結果：幾何品質不足。本次仍可繼續收集 behavioral word-review，但不應使用逐字 gaze；若要使用 gaze，建議重新校準。";
    }
  }
  ui.geometryQualityNotice.classList.remove("hidden");
}

function normalizeFaceBox(box) {
  if (!box || !Number.isFinite(Number(box.x_norm)) || !Number.isFinite(Number(box.y_norm))) return null;
  const left = Number(box.x_norm);
  const top = Number(box.y_norm);
  return [left, top, left + Number(box.w_norm), top + Number(box.h_norm)];
}

function batchId() {
  const random = globalThis.crypto?.randomUUID?.().replaceAll("-", "") || `${Date.now()}${Math.random()}`.replace(".", "");
  return `B-${random.slice(0, 20)}`;
}

async function flushTelemetry() {
  if (!state.telemetryQueue.length) return;
  const samples = state.telemetryQueue.splice(0, 64);
  const payload = {
    batch_id: batchId(),
    passage_id: state.current.passage.passage_id,
    viewport: frozenAssessmentViewport(),
    samples,
  };
  try {
    await api(`/api/study/sessions/${state.context.study_session_id}/general/telemetry`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  } catch (error) {
    state.telemetryQueue.unshift(...samples);
    throw error;
  }
}

async function samplingLoop() {
  while (state.sampling) {
    const cycleStarted = performance.now();
    const elapsed = cycleStarted - state.readingStartedAt;
    try {
      const result = await predictFrame();
      const bbox = normalizeFaceBox(result.face_bbox);
      if (!bbox) throw new Error("prediction returned invalid face box");
      const [x, y] = result.screen_xy_px;
      state.telemetryQueue.push({
        monotonic_elapsed_ms: elapsed,
        prediction_success: true,
        screen_xy_norm: result.screen_xy_norm,
        screen_xy_px: result.screen_xy_px,
        gaze_pitch_yaw: result.gaze_pitch_yaw,
        head_pose_pitch_yaw: result.head_pose_pitch_yaw,
        normalized_face_bbox: bbox,
        nearest_word_index: Number.isFinite(x) && Number.isFinite(y) ? nearestWord(x, y) : null,
      });
    } catch (error) {
      if (error?.code === "assessment_viewport_changed") {
        state.sampling = false;
        state.telemetryQueue.push({
          monotonic_elapsed_ms: elapsed,
          prediction_success: false,
          coarse_failure_code: "viewport_contract_mismatch",
        });
        try {
          await flushTelemetry();
        } catch (_) {
          // The queued integrity failure is retried before behavioral completion.
        }
        showAlert(error.message);
      } else {
        state.telemetryQueue.push({
          monotonic_elapsed_ms: elapsed,
          prediction_success: false,
          coarse_failure_code: coarseFailure(error),
        });
      }
    }
    if (state.telemetryQueue.length >= 16) {
      try { await flushTelemetry(); } catch (_) { /* Retried at reading completion. */ }
    }
    const remaining = Math.max(0, 250 - (performance.now() - cycleStarted));
    if (state.sampling && remaining) await delay(remaining);
  }
}

function scrollWatcher() {
  if (state.sampling && Math.abs(window.scrollY - state.scrollOrigin) > 2) state.scrollOccurred = true;
}

function readingVideoEnabled() {
  const sessionScopes = state.session?.optional_scopes || {};
  const contextScopes = state.context?.optional_scopes || {};
  return sessionScopes[READING_VIDEO_SCOPE] === true || contextScopes[READING_VIDEO_SCOPE] === true;
}

function readingVideoMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || "";
}

function recordingId() {
  const random = globalThis.crypto?.randomUUID?.().replaceAll("-", "")
    || `${Date.now()}${Math.random()}`.replaceAll(".", "");
  return `VID-${random.toUpperCase().slice(0, 24).padEnd(20, "0")}`;
}

function startReadingVideo() {
  if (!readingVideoEnabled()) return;
  if (typeof MediaRecorder === "undefined" || typeof MediaStream === "undefined") {
    throw new Error("這個瀏覽器不支援閱讀影片錄製；本次尚未開始，請改用最新版 Chromium 或 Edge。");
  }
  const videoTracks = state.stream?.getVideoTracks().filter((track) => track.readyState === "live") || [];
  if (videoTracks.length !== 1) throw new Error("找不到唯一且有效的 webcam 影像軌；本次尚未開始。");
  const mimeType = readingVideoMimeType();
  if (!mimeType) throw new Error("這個瀏覽器沒有可用的閱讀影片格式；本次尚未開始。");

  const videoOnlyStream = new MediaStream([videoTracks[0]]);
  const recorder = new MediaRecorder(videoOnlyStream, {
    mimeType,
    videoBitsPerSecond: READING_VIDEO_BITS_PER_SECOND,
  });
  state.videoChunks = [];
  state.videoRecordingId = recordingId();
  state.videoMimeType = recorder.mimeType || mimeType;
  state.videoStopPromise = new Promise((resolve, reject) => {
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data?.size) state.videoChunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      resolve(new Blob(state.videoChunks, { type: state.videoMimeType }));
    }, { once: true });
    recorder.addEventListener("error", (event) => {
      reject(new Error(event.error?.message || "閱讀影片錄製失敗"));
    }, { once: true });
  });
  state.mediaRecorder = recorder;
  recorder.start(5_000);
  ui.videoStatus.textContent = "正在同步錄製本篇無聲閱讀影片（僅 self-development）。";
  announce("無聲閱讀影片與衍生眼動取樣已開始");
}

async function stopReadingVideo(durationMs) {
  if (!readingVideoEnabled()) return null;
  if (state.pendingReadingVideo) return state.pendingReadingVideo;
  const recorder = state.mediaRecorder;
  if (!recorder || !state.videoStopPromise) throw new Error("找不到本篇閱讀影片錄製狀態。");
  if (recorder.state !== "inactive") recorder.stop();
  const blob = await state.videoStopPromise;
  if (!blob.size || blob.size > READING_VIDEO_MAX_BYTES) {
    throw new Error(`閱讀影片大小 ${blob.size} bytes 不在允許範圍內。`);
  }
  state.pendingReadingVideo = {
    blob,
    metadata: {
      recording_id: state.videoRecordingId,
      passage_id: state.current.passage.passage_id,
      round_number: state.current.round_number,
      duration_ms: Math.min(480_000, Math.round(durationMs)),
      mime_type: state.videoMimeType,
    },
  };
  return state.pendingReadingVideo;
}

async function uploadReadingVideo(durationMs) {
  if (!readingVideoEnabled() || state.readingVideoUploaded) return;
  ui.videoStatus.textContent = "正在保存本篇無聲閱讀影片，請勿關閉頁面……";
  const pending = await stopReadingVideo(durationMs);
  const extension = pending.metadata.mime_type.startsWith("video/mp4") ? "mp4" : "webm";
  const form = new FormData();
  form.append("metadata", JSON.stringify(pending.metadata));
  form.append("reading_video", pending.blob, `${pending.metadata.recording_id}.${extension}`);
  await api(`/api/study/sessions/${state.context.study_session_id}/general/reading-video`, {
    method: "POST",
    body: form,
  });
  state.readingVideoUploaded = true;
  state.pendingReadingVideo = null;
  ui.videoStatus.textContent = "本篇無聲閱讀影片已保存為 self-development data。";
}

async function startReading() {
  clearAlert();
  ui.startReadingBtn.disabled = true;
  try {
    assertAssessmentViewportStable();
    await ensureCameraAndModel();
    startReadingVideo();
  } catch (error) {
    ui.startReadingBtn.disabled = false;
    showAlert(error.message);
    return;
  }
  state.telemetryQueue = [];
  state.wordLayout = captureWordLayout();
  state.scrollOrigin = window.scrollY;
  state.scrollOccurred = false;
  state.readingStartedAt = performance.now();
  state.sampling = true;
  window.addEventListener("scroll", scrollWatcher, { passive: true });
  ui.startReadingBtn.classList.add("hidden");
  ui.finishReadingBtn.classList.remove("hidden");
  ui.readingHint.textContent = "請自然閱讀；20 秒後可以完成。";
  state.readingTimer = window.setInterval(() => {
    const elapsed = performance.now() - state.readingStartedAt;
    ui.timer.textContent = formatElapsed(elapsed);
    ui.finishReadingBtn.disabled = elapsed < 20_000;
    if (elapsed >= 20_000) ui.readingHint.textContent = "讀完後可按下完成；請不要返回修改文章。";
  }, 250);
  state.samplingPromise = samplingLoop();
  if (!readingVideoEnabled()) announce("閱讀計時與衍生眼動取樣已開始");
}

async function finishReading() {
  const elapsed = state.readingFinishedElapsed ?? (performance.now() - state.readingStartedAt);
  if (elapsed < 20_000) {
    showAlert("固定流程至少需要 20 秒閱讀時間。");
    return;
  }
  clearAlert();
  ui.finishReadingBtn.disabled = true;
  if (state.readingFinishedElapsed === null) {
    state.readingFinishedElapsed = elapsed;
    state.sampling = false;
    window.clearInterval(state.readingTimer);
    window.removeEventListener("scroll", scrollWatcher);
    await state.samplingPromise;
    const draft = {
      passage_id: state.current.passage.passage_id,
      reading_elapsed_ms: Math.min(480_000, Math.round(elapsed)),
      scroll_occurred: state.scrollOccurred,
      zoom_ratio: Number(window.visualViewport?.scale || 1),
      word_layout: state.wordLayout,
      resumed: state.resumedRound,
    };
    sessionStorage.setItem(draftKey(), JSON.stringify(draft));
  }
  try {
    await uploadReadingVideo(elapsed);
    while (state.telemetryQueue.length) await flushTelemetry();
    const draft = JSON.parse(sessionStorage.getItem(draftKey()) || "null");
    if (!draft) throw new Error("本篇閱讀完成快照遺失。");
    const result = await api(`/api/study/sessions/${state.context.study_session_id}/general/round/probes`, {
      method: "POST",
      body: JSON.stringify({ passage_id: draft.passage_id }),
    });
    renderReviews(result.probes, draft);
  } catch (error) {
    showAlert(`本篇資料尚未送出：${error.message}`);
    ui.finishReadingBtn.disabled = false;
  }
}

function renderReviews(probes, draft) {
  showOnly(ui.reviewPanel);
  ui.reviewForm.replaceChildren(...probes.map((probe) => reviewRow(probe, "review")));
  ui.reviewForm.dataset.draft = JSON.stringify(draft);
  ui.understanding.value = "";
  ui.mentalEffort.value = "";
  ui.readComplete.checked = true;
  ui.interrupted.checked = Boolean(draft.resumed);
  ui.submitRoundBtn.disabled = false;
  window.scrollTo({ top: ui.reviewPanel.offsetTop - 82, behavior: "smooth" });
}

async function restoreOpenReviews() {
  const draft = JSON.parse(sessionStorage.getItem(draftKey()) || "null");
  if (!draft || draft.passage_id !== state.current.passage.passage_id) {
    throw new Error("找不到本篇閱讀完成快照；請聯絡研究者，不要自行重做或填補資料。");
  }
  const result = await api(`/api/study/sessions/${state.context.study_session_id}/general/round/probes`, {
    method: "POST",
    body: JSON.stringify({ passage_id: draft.passage_id }),
  });
  renderReviews(result.probes, draft);
}

async function submitRound() {
  clearAlert();
  const draft = JSON.parse(ui.reviewForm.dataset.draft || "null");
  if (!draft) return showAlert("本篇閱讀快照遺失，資料未送出。");
  const reviews = {};
  for (const row of ui.reviewForm.querySelectorAll(".review-row")) {
    const input = row.querySelector('input[type="radio"]:checked');
    const any = row.querySelector('input[type="radio"]');
    if (!input) return showAlert("請完成全部 8 個單字標記。");
    const probeId = any.name.slice("review-".length);
    reviews[probeId] = input.value;
  }
  if (!ui.understanding.value || !ui.mentalEffort.value) return showAlert("請完成文章理解與心力自評。");
  ui.submitRoundBtn.disabled = true;
  try {
    const result = await api(`/api/study/sessions/${state.context.study_session_id}/general/round`, {
      method: "POST",
      body: JSON.stringify({
        ...draft,
        word_reviews: reviews,
        passage_self_report: {
          understanding: Number(ui.understanding.value),
          mental_effort: Number(ui.mentalEffort.value),
          read_complete: ui.readComplete.checked,
          interrupted: ui.interrupted.checked || draft.resumed,
        },
      }),
    });
    sessionStorage.removeItem(draftKey());
    state.session = result.session;
    await routeCollectionPhase();
  } catch (error) {
    showAlert(error.message);
    ui.submitRoundBtn.disabled = false;
  }
}

async function beginRound() {
  const result = await api(`/api/study/sessions/${state.context.study_session_id}/general/round/start`, {
    method: "POST",
    body: "{}",
  });
  if (result.is_finished) {
    const status = await api(`/api/study/sessions/${state.context.study_session_id}`);
    state.session = status.session;
    await routeCollectionPhase();
    return;
  }
  renderRound(result);
  if (result.phase === "probes_open") await restoreOpenReviews();
  if (result.phase === "reading_active" && sessionStorage.getItem(draftKey())) {
    await restoreOpenReviews();
  }
}

function showCompletion() {
  showOnly(ui.completePanel);
  const quality = state.session.quality?.general_collection;
  const bandLabels = {
    word_level_candidate: "眼動資料達到 rehearsal 的 word-level candidate 描述帶",
    passage_level_only: "眼動資料只建議用於 passage-level 描述",
    behavioral_only: "眼動品質不足；只保留 behavioral word-review 資料",
  };
  ui.qualitySummary.textContent = quality
    ? `${bandLabels[quality.gaze_quality_band] || quality.gaze_quality_band}。這不是正式通過門檻。`
    : "流程已完成；品質摘要尚未提供。";
  stopCamera();
}

async function routeCollectionPhase() {
  const phase = state.session.general_collection?.phase;
  if (state.session.state === "completed" || phase === "completed") return showCompletion();
  if (phase === "start_validation_required" || phase === "end_validation_required") {
    showOnly(ui.validationPanel);
    ui.validationTitle.textContent = phase === "end_validation_required"
      ? "閱讀後 5 點漂移驗證"
      : "閱讀前 5 點品質驗證";
    ui.validationProgress.textContent = "0 / 15";
    ui.validationBtn.textContent = phase === "end_validation_required" ? "開始結束驗證" : "開始 5 點驗證";
    return;
  }
  if (["reading_ready", "reading_active", "probes_open"].includes(phase)) {
    assertAssessmentViewportStable();
    return beginRound();
  }
  throw new Error(`無法恢復收集階段：${phase || "missing"}`);
}

async function restore() {
  state.context = readContext();
  if (!state.context?.study_session_id || state.context.mode !== "rehearsal") {
    location.replace("/study");
    return;
  }
  try {
    const status = await api(`/api/study/sessions/${state.context.study_session_id}`);
    state.session = status.session;
    const collection = state.session.general_collection;
    const design = collection?.assessment_id
      ? null
      : await api("/api/study/general-collection/protocol");
    state.protocol = design?.protocol || null;
    state.practice = design?.practice || null;
    state.gazeMeasurementContract = collection?.assessment_id
      ? collection.gaze_measurement_contract?.contract || null
      : design?.gaze_measurement_contract || null;
    state.context.optional_scopes = state.session.optional_scopes || state.context.optional_scopes || {};
    ui.participantId.textContent = state.session.participant_id;
    const assignment = state.session.collection_assignment || {};
    ui.visitMeta.textContent = `Visit ${assignment.visit_index || "—"} · Form ${assignment.form_id || "—"}`;
    state.context.model_name = state.session.linked_data?.model_name || "";
    sessionStorage.setItem(STUDY_KEY, JSON.stringify(state.context));

    if (state.session.state === "consented") return showOnly(ui.setupPanel);
    if (["system_check_passed", "calibration_in_progress"].includes(state.session.state)) return showOnly(ui.calibrationPanel);
    if (state.session.state === "calibration_complete") {
      await ensureCameraAndModel();
      if (sessionStorage.getItem(practiceKey()) !== "done") return showPractice();
      return beginCollection();
    }
    if (state.session.state === "assessment_in_progress") {
      await ensureCameraAndModel();
      return routeCollectionPhase();
    }
    if (state.session.state === "completed") return showCompletion();
    throw new Error(`無法處理目前的研究狀態：${state.session.state}`);
  } catch (error) {
    showAlert(error.message);
  }
}

ui.checkCameraBtn.addEventListener("click", checkCamera);
ui.saveSetupBtn.addEventListener("click", saveSetup);
ui.practiceBtn.addEventListener("click", () => finishPractice().catch((error) => showAlert(error.message)));
ui.validationBtn.addEventListener("click", runValidation);
ui.startReadingBtn.addEventListener("click", () => startReading().catch((error) => showAlert(error.message)));
ui.finishReadingBtn.addEventListener("click", () => finishReading().catch((error) => showAlert(error.message)));
ui.submitRoundBtn.addEventListener("click", () => submitRound().catch((error) => showAlert(error.message)));
window.addEventListener("pagehide", stopCamera);

restore();
