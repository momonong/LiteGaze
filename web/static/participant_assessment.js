const STUDY_KEY = "lexigaze.participantStudy.v1";
const gazeCapture = window.LexiGazeCapture;

const ui = Object.fromEntries([
  "alert", "participantId", "modelStatus", "cameraBtn", "beginBtn", "leaveBtn",
  "cameraPreview", "captureCanvas", "setupPanel", "readingPanel", "roundLabel",
  "difficulty", "timer", "passage", "readingHint", "startReadingBtn",
  "finishReadingBtn", "quizPanel", "qualityStatus", "quizForm", "submitQuizBtn",
  "resultPanel", "resultSummary", "liveStatus",
].map((id) => [id, document.getElementById(id)]));

const state = {
  context: null,
  stream: null,
  modelName: "",
  assessmentId: "",
  current: null,
  history: [],
  readingStartedAt: 0,
  timerHandle: null,
  sampling: false,
  gaze: null,
  metrics: null,
};

function showAlert(message) {
  ui.alert.textContent = message;
  ui.alert.classList.remove("hidden");
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

function studyBody(extra = {}) {
  return {
    ...extra,
    study_session_id: state.context.study_session_id,
  };
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${state.context.access_token}`);
  headers.set("X-Lexigaze-Study-Session", state.context.study_session_id);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  const result = await response.json().catch(() => ({}));
  if (!response.ok || result.ok === false) {
    throw new Error(result.error || `Request failed (${response.status})`);
  }
  return result;
}

function stopCamera() {
  state.sampling = false;
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  ui.cameraPreview.srcObject = null;
}

async function initializeCameraAndModel() {
  clearAlert();
  ui.cameraBtn.disabled = true;
  ui.modelStatus.textContent = "檢查中…";
  try {
    if (!gazeCapture) throw new Error("相機 capture contract 元件未載入");
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("此頁需要 HTTPS 或 localhost 才能使用相機");
    }
    if (!state.stream) {
      state.stream = await navigator.mediaDevices.getUserMedia(
        gazeCapture.mediaConstraints(),
      );
      ui.cameraPreview.srcObject = state.stream;
      await ui.cameraPreview.play();
    }
    const models = await api("/api/gaze/models");
    const linked = state.context.model_name;
    const model = (models.models || []).find((item) => item.name === linked);
    if (!model) throw new Error("找不到這次校準產生的匿名模型，請返回重新校準");
    state.modelName = model.name;
    ui.modelStatus.textContent = "相機與匿名模型已就緒";
    ui.cameraPreview.classList.remove("hidden");
    ui.beginBtn.disabled = false;
    announce("相機與模型檢查完成，可以開始評量");
  } catch (error) {
    stopCamera();
    ui.modelStatus.textContent = "檢查未通過";
    showAlert(error.message);
  } finally {
    ui.cameraBtn.disabled = false;
  }
}

function renderPassageText(text) {
  ui.passage.replaceChildren();
  let wordIndex = 0;
  for (const part of text.split(/(\s+)/)) {
    if (/^\s+$/.test(part)) {
      ui.passage.append(document.createTextNode(part));
      continue;
    }
    const span = document.createElement("span");
    span.className = "assessment-word";
    span.dataset.wordIndex = String(wordIndex++);
    span.textContent = part;
    ui.passage.append(span);
  }
}

function renderRound(round) {
  state.current = round;
  state.metrics = null;
  ui.setupPanel.classList.add("hidden");
  ui.quizPanel.classList.add("hidden");
  ui.resultPanel.classList.add("hidden");
  ui.readingPanel.classList.remove("hidden");
  ui.roundLabel.textContent = `第 ${round.round} 輪，共 ${round.min_rounds}–${round.max_rounds} 輪`;
  ui.difficulty.textContent = `難度 ${round.difficulty}`;
  ui.timer.textContent = "00:00";
  ui.startReadingBtn.classList.remove("hidden");
  ui.finishReadingBtn.classList.add("hidden");
  ui.readingHint.textContent = "按下「開始閱讀」後計時；讀完再按「完成閱讀」。";
  renderPassageText(round.text);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function beginAssessment() {
  clearAlert();
  ui.beginBtn.disabled = true;
  try {
    const result = await api("/api/inspector/adaptive/start", {
      method: "POST",
      body: JSON.stringify(studyBody()),
    });
    state.assessmentId = result.assessment_id;
    state.history = result.resume_history || [];
    if (result.is_finished) {
      await finishAssessmentReport();
      return;
    }
    renderRound(result);
  } catch (error) {
    showAlert(error.message);
    ui.beginBtn.disabled = false;
  }
}

function formatElapsed(milliseconds) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function resetGazeStats() {
  state.gaze = { attempts: 0, successful: 0, hits: [] };
}

function nearestWord(x, y) {
  const words = [...ui.passage.querySelectorAll(".assessment-word")];
  let best = null;
  let bestDistance = 95;
  for (const word of words) {
    const box = word.getBoundingClientRect();
    if (box.bottom < 0 || box.top > innerHeight) continue;
    const dx = x - (box.left + box.right) / 2;
    const dy = y - (box.top + box.bottom) / 2;
    const distance = Math.hypot(dx, dy);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = Number(word.dataset.wordIndex);
    }
  }
  return best;
}

async function sampleGaze() {
  if (!state.sampling || !state.stream) return;
  state.gaze.attempts += 1;
  const snapshot = gazeCapture.captureSnapshot(
    ui.cameraPreview,
    ui.captureCanvas,
  );
  try {
    const result = await api("/api/gaze/predict", {
      method: "POST",
      body: JSON.stringify(studyBody({
        image_data: snapshot.image_data,
        capture_contract: snapshot.capture_contract,
        model_name: state.modelName,
        viewport_width: innerWidth,
        viewport_height: innerHeight,
      })),
    });
    state.gaze.successful += 1;
    const [x, y] = result.screen_xy_px || [];
    if (Number.isFinite(x) && Number.isFinite(y)) {
      const wordIndex = nearestWord(x, y);
      if (wordIndex !== null && state.gaze.hits.length < 3000) {
        state.gaze.hits.push({ wordIndex, at: performance.now() });
      }
    }
  } catch (_) {
    // Individual inference failures are summarized by the quality metric.
  } finally {
    if (state.sampling) window.setTimeout(sampleGaze, 700);
  }
}

function startReading() {
  clearAlert();
  resetGazeStats();
  state.readingStartedAt = performance.now();
  state.sampling = true;
  ui.passage.classList.add("reading-active");
  ui.startReadingBtn.classList.add("hidden");
  ui.finishReadingBtn.classList.remove("hidden");
  ui.readingHint.textContent = "請自然閱讀；完成後按下按鈕。";
  state.timerHandle = window.setInterval(() => {
    ui.timer.textContent = formatElapsed(performance.now() - state.readingStartedAt);
  }, 250);
  sampleGaze();
  announce("閱讀計時與非儲存式眼動推論已開始");
}

function deriveMetrics(durationMs) {
  const hits = state.gaze.hits;
  let regressions = 0;
  let fixationTotal = 0;
  let fixationCount = 0;
  let runStartedAt = hits[0]?.at;
  for (let index = 1; index < hits.length; index += 1) {
    if (hits[index].wordIndex < hits[index - 1].wordIndex - 1) regressions += 1;
    if (hits[index].wordIndex !== hits[index - 1].wordIndex) {
      fixationTotal += hits[index - 1].at - runStartedAt;
      fixationCount += 1;
      runStartedAt = hits[index].at;
    }
  }
  if (hits.length && runStartedAt != null) {
    fixationTotal += hits[hits.length - 1].at - runStartedAt + 700;
    fixationCount += 1;
  }
  const successRate = state.gaze.attempts ? state.gaze.successful / state.gaze.attempts : 0;
  const quality = successRate >= 0.65 && hits.length >= 8
    ? "good"
    : successRate >= 0.3 && hits.length >= 3 ? "limited" : "insufficient";
  return {
    wpm: Math.round((state.current.word_count / (durationMs / 60000)) * 10) / 10,
    regression_rate: hits.length > 1 ? regressions / (hits.length - 1) : 0,
    avg_fixation_duration_ms: fixationCount ? fixationTotal / fixationCount : null,
    data_quality_status: quality,
  };
}

function renderQuiz() {
  ui.readingPanel.classList.add("hidden");
  ui.quizPanel.classList.remove("hidden");
  ui.qualityStatus.textContent = `眼動品質：${state.metrics.data_quality_status}`;
  ui.quizForm.replaceChildren();
  state.current.quiz.forEach((question, questionIndex) => {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "quiz-question";
    const legend = document.createElement("legend");
    legend.textContent = `${questionIndex + 1}. ${question.question}`;
    fieldset.append(legend);
    Object.entries(question.options).forEach(([key, value]) => {
      const label = document.createElement("label");
      label.className = "option";
      const input = document.createElement("input");
      input.type = "radio";
      input.name = question.question_id;
      input.value = key;
      const text = document.createElement("span");
      text.textContent = `${key}. ${value}`;
      label.append(input, text);
      fieldset.append(label);
    });
    ui.quizForm.append(fieldset);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function finishReading() {
  const durationMs = performance.now() - state.readingStartedAt;
  if (durationMs < 5000) {
    showAlert("閱讀時間少於 5 秒，請確認文章已讀完再繼續。");
    return;
  }
  clearAlert();
  state.sampling = false;
  window.clearInterval(state.timerHandle);
  state.timerHandle = null;
  state.metrics = deriveMetrics(durationMs);
  ui.passage.classList.remove("reading-active");
  renderQuiz();
}

async function submitQuiz() {
  clearAlert();
  const responses = {};
  for (const question of state.current.quiz) {
    const selected = ui.quizForm.querySelector(`input[name="${CSS.escape(question.question_id)}"]:checked`);
    if (!selected) {
      showAlert("請完成所有題目再送出。");
      return;
    }
    responses[question.question_id] = selected.value;
  }
  ui.submitQuizBtn.disabled = true;
  try {
    const scored = await api("/api/inspector/adaptive/score", {
      method: "POST",
      body: JSON.stringify(studyBody({
        assessment_id: state.assessmentId,
        round: state.current.round,
        round_token: state.current.round_token,
        passage_id: state.current.passage_id,
        responses,
        metrics: state.metrics,
      })),
    });
    state.history.push({
      round: state.current.round,
      passage_id: state.current.passage_id,
      result_token: scored.result_token,
    });
    const next = await api("/api/inspector/adaptive/next", {
      method: "POST",
      body: JSON.stringify(studyBody({
        assessment_id: state.assessmentId,
        history: state.history,
      })),
    });
    if (next.is_finished) {
      await finishAssessmentReport();
    } else {
      renderRound(next);
    }
  } catch (error) {
    showAlert(error.message);
  } finally {
    ui.submitQuizBtn.disabled = false;
  }
}

async function finishAssessmentReport() {
  const result = await api("/api/inspector/adaptive/report", {
    method: "POST",
    body: JSON.stringify(studyBody({
      assessment_id: state.assessmentId,
      history: state.history,
      persist: false,
    })),
  });
  stopCamera();
  ui.setupPanel.classList.add("hidden");
  ui.readingPanel.classList.add("hidden");
  ui.quizPanel.classList.add("hidden");
  ui.resultPanel.classList.remove("hidden");
  const rate = result.summary.comprehension_rate;
  ui.resultSummary.textContent = rate == null
    ? "這次資料不足以計算作答比例；研究流程仍已安全結束。"
    : `本次題組的觀察答對率為 ${rate}%。這不是標準化能力分數。`;
  announce("閱讀評量已完成");
}

async function restore() {
  state.context = readContext();
  if (!state.context || state.context.mode !== "pilot" || !state.context.study_session_id) {
    location.replace("/study");
    return;
  }
  ui.participantId.textContent = state.context.participant_id;
  try {
    const status = await api(`/api/study/sessions/${state.context.study_session_id}`);
    const session = status.session;
    state.context.model_name = session.linked_data?.model_name || "";
    sessionStorage.setItem(STUDY_KEY, JSON.stringify(state.context));
    if (session.state === "completed") {
      ui.setupPanel.classList.add("hidden");
      ui.resultPanel.classList.remove("hidden");
      ui.resultSummary.textContent = "這次評量已完成；你可以回研究首頁查看退出與聯絡資訊。";
      return;
    }
    if (!["calibration_complete", "assessment_in_progress"].includes(session.state)) {
      throw new Error("目前研究階段尚未完成校準，請返回研究首頁。");
    }
    await initializeCameraAndModel();
    if (session.state === "assessment_in_progress") await beginAssessment();
  } catch (error) {
    showAlert(error.message);
  }
}

ui.cameraBtn.addEventListener("click", initializeCameraAndModel);
ui.beginBtn.addEventListener("click", beginAssessment);
ui.startReadingBtn.addEventListener("click", startReading);
ui.finishReadingBtn.addEventListener("click", finishReading);
ui.submitQuizBtn.addEventListener("click", submitQuiz);
ui.leaveBtn.addEventListener("click", () => { stopCamera(); location.assign("/study"); });
window.addEventListener("pagehide", stopCamera);

restore();
