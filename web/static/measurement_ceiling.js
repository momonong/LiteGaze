(function measurementCeilingPage() {
  "use strict";

  const API_BASE = "/api/measurement-ceiling";
  const STORAGE_KEY = "lexigaze.measurementCeiling.v1";
  const RUN_ID_HEADER = "X-Lexigaze-Measurement-Run-Id";
  const RUN_TOKEN_HEADER = "X-Lexigaze-Measurement-Run-Token";
  const CHALLENGE_TOKEN_HEADER = "X-Lexigaze-Measurement-Challenge-Token";
  const PREFLIGHT_TOKEN_HEADER = "X-Lexigaze-Measurement-Preflight-Token";
  const CREATE_REQUEST_ID_HEADER = "X-Lexigaze-Measurement-Create-Request-Id";

  const ui = {
    setupPanel: document.getElementById("setupPanel"),
    blockPanel: document.getElementById("blockPanel"),
    runPanel: document.getElementById("runPanel"),
    completionPanel: document.getElementById("completionPanel"),
    cameraPreview: document.getElementById("cameraPreview"),
    captureCanvas: document.getElementById("captureCanvas"),
    cameraSummary: document.getElementById("cameraSummary"),
    preflightSummary: document.getElementById("preflightSummary"),
    startCameraButton: document.getElementById("startCameraButton"),
    preflightButton: document.getElementById("preflightButton"),
    startRunButton: document.getElementById("startRunButton"),
    resumeRunButton: document.getElementById("resumeRunButton"),
    confirmBlockButton: document.getElementById("confirmBlockButton"),
    retryButton: document.getElementById("retryButton"),
    abortButton: document.getElementById("abortButton"),
    analyzeRunButton: document.getElementById("analyzeRunButton"),
    verifyArtifactButton: document.getElementById("verifyArtifactButton"),
    refreshStatusButton: document.getElementById("refreshStatusButton"),
    blockTitle: document.getElementById("blockTitle"),
    blockInstruction: document.getElementById("blockInstruction"),
    targetDot: document.getElementById("targetDot"),
    runMessage: document.getElementById("runMessage"),
    progressText: document.getElementById("progressText"),
    phaseText: document.getElementById("phaseText"),
    completionMessage: document.getElementById("completionMessage"),
    completionTitle: document.getElementById("completionTitle"),
    analysisSummary: document.getElementById("analysisSummary"),
  };

  const state = {
    stream: null,
    captureContract: null,
    viewport: null,
    preflightToken: null,
    preflightReady: false,
    pendingCreate: null,
    recoverableContext: null,
    credentials: null,
    challenge: null,
    gateState: null,
    gateFrame: null,
    pendingCapture: null,
    retryMode: null,
    lastAcknowledgedBlock: null,
    busy: false,
  };

  function loadStoredContext() {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (stored === null) return null;
    try {
      const context = JSON.parse(stored);
      if (!context || typeof context !== "object") {
        return { state: "invalid_stored_context" };
      }
      if (globalThis.LexiGazeMeasurementClientPolicy.isPendingCreate(context)) {
        const replacingPreflight = (
          globalThis.LexiGazeMeasurementClientPolicy.canReplacePendingPreflight(context)
        );
        if (
          (!context.preflight_token && !replacingPreflight)
          || !context.capture_contract
          || !context.viewport
        ) {
          return { state: "invalid_stored_context" };
        }
        return context;
      }
      if (!context.run_id || !context.run_token) {
        return { state: "invalid_stored_context" };
      }
      return context;
    } catch (_error) {
      return { state: "invalid_stored_context" };
    }
  }

  function persistContext() {
    if (!state.credentials && state.pendingCreate) {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
        state: "pending_create",
        create_request_id: state.pendingCreate.createRequestId,
        run_token: state.pendingCreate.runToken,
        preflight_token: state.pendingCreate.preflightToken,
        preflight_replacement_required: (
          state.pendingCreate.preflightReplacementRequired === true
        ),
        capture_contract: state.pendingCreate.captureContract,
        viewport: state.pendingCreate.viewport,
      }));
      return;
    }
    if (!state.credentials) {
      sessionStorage.removeItem(STORAGE_KEY);
      return;
    }
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
      state: "active_run",
      run_id: state.credentials.runId,
      run_token: state.credentials.runToken,
      challenge_token: state.challenge?.token || null,
      challenge_id: state.challenge?.id || null,
      schedule_row: state.challenge?.row || null,
      last_acknowledged_block: state.lastAcknowledgedBlock,
      frozen_viewport: state.viewport,
    }));
  }

  function authHeaders({ challenge = false } = {}) {
    if (!state.credentials) throw new Error("量測憑證不存在。請建立新的量測。");
    const headers = {
      "Content-Type": "application/json",
      [RUN_ID_HEADER]: state.credentials.runId,
      [RUN_TOKEN_HEADER]: state.credentials.runToken,
    };
    if (challenge) {
      if (!state.challenge?.token) throw new Error("目標 challenge 憑證不存在。");
      headers[CHALLENGE_TOKEN_HEADER] = state.challenge.token;
    }
    return headers;
  }

  async function requestJson(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
      ...options,
    });
    let payload = {};
    try {
      payload = await response.json();
    } catch (_error) {
      payload = { ok: false, error: "本機 server 回傳了無效 JSON。" };
    }
    return { response, payload };
  }

  function currentViewport() {
    return globalThis.LexiGazeMeasurementGate.frozenViewport(
      globalThis.innerWidth,
      globalThis.innerHeight,
      globalThis.devicePixelRatio || 1,
    );
  }

  function viewportMatches(left, right) {
    return Boolean(left && right)
      && Math.abs(Number(left.width) - Number(right.width)) <= 0.5
      && Math.abs(Number(left.height) - Number(right.height)) <= 0.5
      && Math.abs(
        Number(left.device_pixel_ratio) - Number(right.device_pixel_ratio),
      ) <= 0.001;
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("瀏覽器不支援相機 API。");
    if (!globalThis.LexiGazeCapture) throw new Error("capture contract 元件未載入。");
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = await navigator.mediaDevices.getUserMedia(
      globalThis.LexiGazeCapture.mediaConstraints(),
    );
    ui.cameraPreview.srcObject = state.stream;
    await ui.cameraPreview.play();
    state.captureContract = globalThis.LexiGazeCapture.captureContract(ui.cameraPreview);
    state.viewport = currentViewport();
    ui.cameraSummary.textContent = [
      `實際來源 ${state.captureContract.source_width_px}×${state.captureContract.source_height_px}`,
      `transport ${state.captureContract.transport_width_px}×${state.captureContract.transport_height_px}`,
      `FPS ${state.captureContract.source_frame_rate_hz || "未知"}`,
    ].join(" · ");
    state.preflightToken = null;
    state.preflightReady = false;
    ui.preflightSummary.textContent = "相機已啟動；尚未完成連續 3 幀 readiness preflight。";
    updateButtons();
  }

  function captureSnapshot() {
    if (!state.stream) throw new Error("相機尚未啟動。");
    return globalThis.LexiGazeCapture.captureSnapshot(
      ui.cameraPreview,
      ui.captureCanvas,
    );
  }

  function updateButtons() {
    const creationBlocked = globalThis.LexiGazeMeasurementClientPolicy.blocksNewRun(
      state.recoverableContext,
      state.credentials,
    );
    const replacingPreflight = Boolean(
      state.pendingCreate?.preflightReplacementRequired === true,
    );
    ui.startCameraButton.disabled = state.busy;
    ui.preflightButton.disabled = (
      state.busy || !state.stream || (creationBlocked && !replacingPreflight)
    );
    ui.startRunButton.disabled = state.busy || !state.preflightReady || creationBlocked;
    ui.resumeRunButton.disabled = (
      state.busy
      || (replacingPreflight && !state.pendingCreate?.preflightToken)
    );
    ui.confirmBlockButton.disabled = state.busy;
    ui.retryButton.disabled = state.busy;
    ui.abortButton.disabled = state.busy;
    ui.analyzeRunButton.disabled = state.busy;
    ui.verifyArtifactButton.disabled = state.busy;
    ui.refreshStatusButton.disabled = state.busy;
  }

  function setBusy(busy, message = "") {
    state.busy = busy;
    updateButtons();
    if (message) ui.runMessage.textContent = message;
  }

  function nextAnimationFrame() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
  }

  async function runReadinessPreflight() {
    const replacingPendingPreflight = Boolean(
      state.pendingCreate?.preflightReplacementRequired === true
      && globalThis.LexiGazeMeasurementClientPolicy.isPendingCreate(
        state.recoverableContext,
      ),
    );
    if ((state.recoverableContext || state.credentials) && !replacingPendingPreflight) {
      throw new Error(
        "這個分頁仍有可恢復的舊 run；請先繼續它，或完成 authenticated abort+cleanup。",
      );
    }
    if (!state.stream || !state.captureContract || !state.viewport) {
      throw new Error("請先啟動相機。");
    }
    state.preflightToken = null;
    state.preflightReady = false;
    setBusy(true, "正在做不含 target、也不落盤的相機 readiness preflight…");
    try {
      const started = await requestJson("/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          capture_contract: state.captureContract,
          viewport: state.viewport,
        }),
      });
      if (!started.response.ok || started.payload.ok !== true) {
        throw new Error(started.payload.error || "無法啟動相機 readiness preflight。");
      }
      state.preflightToken = started.response.headers.get(PREFLIGHT_TOKEN_HEADER);
      if (!state.preflightToken) throw new Error("本機 server 沒有回傳 preflight 憑證。");

      let finalPreflightResult = null;
      for (let attempt = 1; attempt <= 3; attempt += 1) {
        await nextAnimationFrame();
        await nextAnimationFrame();
        const snapshot = captureSnapshot();
        ui.preflightSummary.textContent = `Readiness preflight：正在檢查第 ${attempt} / 3 幀…`;
        const result = await requestJson("/preflight/frames", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            [PREFLIGHT_TOKEN_HEADER]: state.preflightToken,
          },
          body: JSON.stringify({
            image_data: snapshot.image_data,
            capture_contract: snapshot.capture_contract,
          }),
        });
        if (!result.response.ok || result.payload.ok !== true) {
          state.preflightReady = false;
          const noFace = result.payload.classification === "attributable_sensor_failure";
          ui.preflightSummary.textContent = noFace
            ? "Readiness 未通過：沒有偵測到完整臉部。請面向光源、避免背光、置中臉部並露出雙眼，再重新檢查 3 幀。"
            : (result.payload.error || "Readiness hard error；請檢查相機與本機 model 後重試。");
          return;
        }
        const consecutive = Number(result.payload.consecutive_successes || 0);
        finalPreflightResult = result.payload;
        ui.preflightSummary.textContent = `Readiness preflight：連續成功 ${consecutive} / 3 幀。`;
      }
      if (
        finalPreflightResult?.ready !== true
        || Number(finalPreflightResult.consecutive_successes) !== 3
      ) {
        throw new Error("本機 server 未確認連續 3 幀 readiness；不建立 measurement run。");
      }
      state.preflightReady = true;
      if (replacingPendingPreflight) {
        // Preserve the same create authority. Only the expired readiness proof
        // and its not-yet-bound capture context are replaced.
        state.pendingCreate.preflightToken = state.preflightToken;
        state.pendingCreate.captureContract = state.captureContract;
        state.pendingCreate.viewport = state.viewport;
        state.pendingCreate.preflightReplacementRequired = false;
        state.recoverableContext = {
          state: "pending_create",
          create_request_id: state.pendingCreate.createRequestId,
          run_token: state.pendingCreate.runToken,
          preflight_token: state.pendingCreate.preflightToken,
          preflight_replacement_required: false,
          capture_contract: state.pendingCreate.captureContract,
          viewport: state.pendingCreate.viewport,
        };
        persistContext();
        ui.resumeRunButton.hidden = false;
        ui.resumeRunButton.textContent = "以原 authority 繼續建立量測";
      }
      ui.preflightSummary.textContent = (
        "Readiness 通過：3 幀 baseline face+inference 成功。這只是 sensor availability，"
        + "不會成為 calibration/evaluation 樣本。"
      );
    } finally {
      setBusy(false);
    }
  }

  function updateProgress(payload = {}) {
    const progress = payload.progress || payload.status?.progress || {};
    const completed = Number(
      progress.next_sequence_index
      ?? progress.total_count
      ?? payload.completed_count
      ?? 0,
    );
    const bounded = Number.isFinite(completed) ? Math.max(0, Math.min(193, completed)) : 0;
    ui.progressText.textContent = `${bounded} / 193`;
    ui.phaseText.textContent = String(payload.phase || payload.status?.phase || "量測進行中");
  }

  function credentialsFromCreateResponse(response, pendingCreate) {
    const runId = response.headers.get(RUN_ID_HEADER);
    const echoedToken = response.headers.get(RUN_TOKEN_HEADER);
    if (!runId) throw new Error("本機 server 沒有回傳 run ID。");
    if (echoedToken && echoedToken !== pendingCreate.runToken) {
      throw new Error("本機 server 回傳了不同的 run authority；拒絕接管。");
    }
    return { runId, runToken: pendingCreate.runToken };
  }

  function challengeFromResponse(response, payload) {
    const token = response.headers.get(CHALLENGE_TOKEN_HEADER);
    const row = payload.schedule_row;
    globalThis.LexiGazeMeasurementGate.validateScheduleRow(row);
    if (!token || !payload.challenge_id) throw new Error("本機 server 沒有回傳目標 challenge。");
    return { token, id: payload.challenge_id, row };
  }

  function markPendingCreatePreflightReplacement(payload) {
    if (
      !state.pendingCreate
      || payload?.classification !== "pending_create_preflight_required"
      || payload?.existing_run !== false
      || payload?.authority_retained !== true
      || payload?.replace_preflight_allowed !== true
    ) {
      return false;
    }
    state.pendingCreate.preflightToken = null;
    state.pendingCreate.preflightReplacementRequired = true;
    state.preflightToken = null;
    state.preflightReady = false;
    state.recoverableContext = {
      state: "pending_create",
      create_request_id: state.pendingCreate.createRequestId,
      run_token: state.pendingCreate.runToken,
      preflight_token: null,
      preflight_replacement_required: true,
      capture_contract: state.pendingCreate.captureContract,
      viewport: state.pendingCreate.viewport,
    };
    persistContext();
    ui.setupPanel.hidden = false;
    ui.resumeRunButton.hidden = false;
    ui.resumeRunButton.textContent = "先重新做 readiness preflight";
    ui.preflightSummary.textContent = (
      "Server 已確認同一 authority 尚未建立 run。請保留本分頁，重新做 3 幀 "
      + "readiness；create ID 與 256-bit run secret 不會更換。"
    );
    updateButtons();
    return true;
  }

  async function createRun() {
    if (state.recoverableContext || state.credentials) {
      throw new Error(
        "不可覆寫可恢復的舊 run；請先繼續它，或完成 authenticated abort+cleanup。",
      );
    }
    if (!state.captureContract || !state.viewport) throw new Error("請先啟動相機。");
    if (!state.preflightReady || !state.preflightToken) {
      throw new Error("建立 run 前必須通過連續 3 幀 readiness preflight。");
    }
    setBusy(true, "正在建立不含 participant invite 的本機量測…");
    try {
      const authority = globalThis.LexiGazeMeasurementClientPolicy.newCreateAuthority(
        (bytes) => globalThis.crypto.getRandomValues(bytes),
      );
      state.pendingCreate = {
        createRequestId: authority.create_request_id,
        runToken: authority.run_token,
        preflightToken: state.preflightToken,
        preflightReplacementRequired: false,
        captureContract: state.captureContract,
        viewport: state.viewport,
      };
      state.recoverableContext = {
        state: "pending_create",
        create_request_id: authority.create_request_id,
        run_token: authority.run_token,
        preflight_token: state.preflightToken,
        preflight_replacement_required: false,
        capture_contract: state.captureContract,
        viewport: state.viewport,
      };
      // Persist authority before the request so a lost response cannot orphan
      // a server-created run.
      persistContext();
      const { response, payload } = await submitPendingCreate();
      if (!response.ok || payload.ok !== true) {
        if (markPendingCreatePreflightReplacement(payload)) return;
        throw new Error(payload.error || "建立量測失敗；pending-create authority 已保留。");
      }
      await activateCreatedRun(response, payload);
    } finally {
      setBusy(false);
    }
  }

  async function submitPendingCreate() {
    if (!state.pendingCreate) throw new Error("pending-create authority 不存在。");
    const headers = {
      "Content-Type": "application/json",
      [CREATE_REQUEST_ID_HEADER]: state.pendingCreate.createRequestId,
      [RUN_TOKEN_HEADER]: state.pendingCreate.runToken,
    };
    if (state.pendingCreate.preflightToken) {
      headers[PREFLIGHT_TOKEN_HEADER] = state.pendingCreate.preflightToken;
    }
    return requestJson("/runs", {
      method: "POST",
      headers,
      body: JSON.stringify({
        capture_contract: state.pendingCreate.captureContract,
        viewport: state.pendingCreate.viewport,
      }),
    });
  }

  async function activateCreatedRun(response, payload) {
    state.credentials = credentialsFromCreateResponse(response, state.pendingCreate);
    state.pendingCreate = null;
    state.recoverableContext = {
      state: "active_run",
      run_id: state.credentials.runId,
      run_token: state.credentials.runToken,
    };
    state.preflightToken = null;
    state.preflightReady = false;
    state.challenge = null;
    state.lastAcknowledgedBlock = null;
    persistContext();
    ui.setupPanel.hidden = true;
    ui.abortButton.hidden = false;
    updateProgress(payload);
    await issueChallenge();
  }

  async function status({ includeChallenge = false } = {}) {
    const { response, payload } = await requestJson("/status", {
      method: "GET",
      headers: authHeaders({ challenge: includeChallenge }),
    });
    if (!response.ok || payload.ok !== true) throw new Error(payload.error || "無法讀取量測狀態。");
    updateProgress(payload);
    return payload;
  }

  async function rotateChallenge() {
    const { response, payload } = await requestJson("/challenges/rotate", {
      method: "POST",
      headers: authHeaders(),
      body: "{}",
    });
    if (!response.ok || payload.ok !== true) throw new Error(payload.error || "無法旋轉遺失的 challenge。");
    state.challenge = challengeFromResponse(response, payload);
    persistContext();
    await presentChallenge();
  }

  async function resumeRun() {
    const stored = state.recoverableContext || loadStoredContext();
    if (!stored) throw new Error("這個分頁沒有可恢復的量測憑證。");
    if (globalThis.LexiGazeMeasurementClientPolicy.isInvalidStoredContext(stored)) {
      throw new Error(
        "sessionStorage recovery context 已損壞；為避免孤兒 run，禁止建立新 run。"
        + "請保留本分頁並由 researcher 檢查 dedicated run registry。",
      );
    }
    if (globalThis.LexiGazeMeasurementClientPolicy.isPendingCreate(stored)) {
      state.pendingCreate = {
        createRequestId: stored.create_request_id,
        runToken: stored.run_token,
        preflightToken: stored.preflight_token,
        preflightReplacementRequired: stored.preflight_replacement_required === true,
        captureContract: stored.capture_contract,
        viewport: stored.viewport,
      };
      if (
        state.pendingCreate.preflightReplacementRequired
        && !state.pendingCreate.preflightToken
      ) {
        ui.setupPanel.hidden = false;
        ui.resumeRunButton.hidden = false;
        ui.resumeRunButton.textContent = "先重新做 readiness preflight";
        ui.preflightSummary.textContent = (
          "同一 create authority 已保留；請先啟動相機並重新完成 3 幀 readiness。"
        );
        updateButtons();
        return;
      }
      setBusy(true, "正在用原 create request authority 恢復既有 run…");
      try {
        const { response, payload } = await submitPendingCreate();
        if (!response.ok || payload.ok !== true) {
          if (markPendingCreatePreflightReplacement(payload)) return;
          throw new Error(payload.error || "pending create 尚未恢復；authority 已保留。");
        }
        await activateCreatedRun(response, payload);
      } finally {
        setBusy(false);
      }
      return;
    }
    state.credentials = { runId: stored.run_id, runToken: stored.run_token };
    state.lastAcknowledgedBlock = stored.last_acknowledged_block || null;
    if (stored.challenge_token) {
      state.challenge = {
        token: stored.challenge_token,
        id: stored.challenge_id || null,
        row: stored.schedule_row || null,
      };
    }
    setBusy(true, "正在核對本機量測狀態…");
    try {
      const current = await status({ includeChallenge: Boolean(state.challenge?.token) });
      ui.setupPanel.hidden = true;
      ui.abortButton.hidden = false;
      if (globalThis.LexiGazeMeasurementClientPolicy.isCalibrationUnusableNegative(current)) {
        showCalibrationNegative(current);
        return;
      }
      if (current.phase === "artifact_verified" || current.phase === "capture_sealed") {
        showCompletion(current);
        return;
      }
      if (globalThis.LexiGazeMeasurementClientPolicy.abortRequired(current)) {
        showAbortRequired(current);
        return;
      }
      const recovery = current.challenge_recovery;
      if (recovery && state.challenge?.token) {
        if (
          recovery.status === "committed"
          || globalThis.LexiGazeMeasurementClientPolicy.serverSpoolRetryAvailable(
            recovery,
          )
        ) {
          await resumeServerSpool();
          return;
        }
        if (recovery.status !== "active") {
          throw new Error("Server 回傳未知的 challenge recovery 狀態。");
        }
        globalThis.LexiGazeMeasurementGate.validateScheduleRow(recovery.schedule_row);
        state.challenge = {
          token: state.challenge.token,
          id: recovery.challenge_id,
          row: recovery.schedule_row,
        };
      }
      if (stored.frozen_viewport) {
        const frozen = globalThis.LexiGazeMeasurementGate.frozenViewport(
          stored.frozen_viewport.width,
          stored.frozen_viewport.height,
          stored.frozen_viewport.device_pixel_ratio,
        );
        if (!viewportMatches(frozen, currentViewport())) {
          throw new Error(
            `這個 run 綁定 ${frozen.width}×${frozen.height} @ DPR ${frozen.device_pixel_ratio}；`
            + "請恢復原本視窗大小與縮放後再繼續。",
          );
        }
        state.viewport = frozen;
      }
      if (current.challenge_outstanding === true) {
        if (state.challenge?.row) {
          if (!state.stream) await startCamera();
          persistContext();
          await presentChallenge();
        } else {
          await rotateChallenge();
        }
      } else {
        state.challenge = null;
        persistContext();
        await issueChallenge();
      }
    } finally {
      setBusy(false);
    }
  }

  async function issueChallenge() {
    const { response, payload } = await requestJson("/challenges", {
      method: "POST",
      headers: authHeaders(),
      body: "{}",
    });
    if (response.status === 409 && payload.classification === "calibration_finalize_required") {
      await finalizeCalibration();
      return;
    }
    if (response.status === 409 && payload.classification === "capture_complete") {
      showCompletion(payload);
      return;
    }
    if (!response.ok || payload.ok !== true) throw new Error(payload.error || "無法取得下一個目標。");
    state.challenge = challengeFromResponse(response, payload);
    state.pendingCapture = null;
    persistContext();
    updateProgress(payload);
    await presentChallenge();
  }

  function blockGuidance(row) {
    if (row.distance === "near") return "身體與臉部稍微靠近 webcam，仍讓完整臉部留在畫面內。";
    if (row.distance === "far") return "身體與臉部稍微遠離 webcam，仍保持正面與完整臉部。";
    if (row.posture === "left") return "上半身稍微向左移，臉部仍朝向螢幕並看得到完整五官。";
    if (row.posture === "right") return "上半身稍微向右移，臉部仍朝向螢幕並看得到完整五官。";
    return "回到平常的中立坐姿與距離，臉部朝向螢幕。";
  }

  function targetPositionClass(axis, fraction) {
    const percent = Math.round(Number(fraction) * 100);
    const allowed = axis === "x"
      ? new Set([8, 18, 29, 39, 50, 61, 71, 82, 92])
      : new Set([10, 20, 30, 40, 50, 60, 70, 80, 90]);
    if (!allowed.has(percent)) throw new Error("target fraction is outside frozen CSS positions");
    return `target-${axis}-${String(percent).padStart(3, "0")}`;
  }

  function applyTargetPosition(row) {
    for (const name of Array.from(ui.targetDot.classList)) {
      if (name.startsWith("target-x-") || name.startsWith("target-y-")) {
        ui.targetDot.classList.remove(name);
      }
    }
    ui.targetDot.classList.add(
      targetPositionClass("x", row.target_x_viewport_fraction),
      targetPositionClass("y", row.target_y_viewport_fraction),
    );
  }

  async function presentChallenge() {
    cancelGateLoop();
    ui.targetDot.hidden = true;
    ui.retryButton.hidden = true;
    state.retryMode = null;
    ui.runPanel.hidden = false;
    const row = state.challenge.row;
    updateProgress({ completed_count: row.sequence_index, phase: row.block_role });
    if (state.lastAcknowledgedBlock !== row.block_id) {
      ui.blockTitle.textContent = `${row.block_id} · ${row.block_role}`;
      ui.blockInstruction.textContent = blockGuidance(row);
      ui.blockPanel.hidden = false;
      ui.runMessage.textContent = "請先依區塊指示調整姿勢。";
      return;
    }
    await showTargetAndGate();
  }

  async function confirmBlock() {
    state.lastAcknowledgedBlock = state.challenge.row.block_id;
    persistContext();
    ui.blockPanel.hidden = true;
    await showTargetAndGate();
  }

  async function showTargetAndGate() {
    if (!state.challenge) return;
    state.viewport = state.viewport || currentViewport();
    applyTargetPosition(state.challenge.row);
    ui.targetDot.hidden = false;
    ui.runMessage.textContent = "請只注視圓點；分頁失焦、隱藏或縮放會重新計時。";
    state.gateState = globalThis.LexiGazeMeasurementGate.begin(
      state.challenge.row,
      state.viewport,
      performance.now(),
    );
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    gateLoop();
  }

  function gateLoop() {
    cancelGateLoop();
    const step = (nowMs) => {
      if (!state.challenge || ui.targetDot.hidden) return;
      if (state.busy) {
        state.gateFrame = requestAnimationFrame(step);
        return;
      }
      const rect = ui.targetDot.getBoundingClientRect();
      state.gateState = globalThis.LexiGazeMeasurementGate.observe(
        state.gateState,
        state.challenge.row,
        {
          now_ms: nowMs,
          visibility_state: document.visibilityState,
          document_focused: document.hasFocus(),
          viewport: currentViewport(),
          target_rect: {
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height,
          },
        },
      );
      if (state.gateState.ready) {
        void captureCurrentTarget();
        return;
      }
      ui.runMessage.textContent = state.gateState.reason === "dwelling"
        ? "保持注視圓點…"
        : "等待分頁可見、焦點與目標位置穩定…";
      state.gateFrame = requestAnimationFrame(step);
    };
    state.gateFrame = requestAnimationFrame(step);
  }

  function cancelGateLoop() {
    if (state.gateFrame !== null) cancelAnimationFrame(state.gateFrame);
    state.gateFrame = null;
  }

  async function advanceAfterConsumedCapture(payload) {
    const wasSensorFailure = payload.classification === "attributable_sensor_failure";
    state.challenge = null;
    state.pendingCapture = null;
    state.retryMode = null;
    persistContext();
    ui.runMessage.textContent = wasSensorFailure
      ? "這一列記錄為 no-face/sensor failure（不重抽直到成功）。"
      : "這一列已記錄。";
    if (wasSensorFailure) {
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
    if (payload.next_action === "finalize_calibration") await finalizeCalibration();
    else if (payload.next_action === "verify_artifact") showCompletion(payload);
    else await issueChallenge();
  }

  async function resumeServerSpool() {
    if (!state.challenge?.token) {
      throw new Error("Server-spool recovery requires the stored challenge token.");
    }
    setBusy(true, "正在從加密 server spool 恢復同一幀；不會擷取新 frame…");
    try {
      const { response, payload } = await requestJson("/captures", {
        method: "POST",
        headers: authHeaders({ challenge: true }),
        body: JSON.stringify({ resume_server_spool: true }),
      });
      updateProgress(payload);
      if (payload.consumed === true) {
        await advanceAfterConsumedCapture(payload);
        return;
      }
      if (globalThis.LexiGazeMeasurementClientPolicy.abortRequired(payload)) {
        showAbortRequired(payload);
        return;
      }
      if (
        payload.classification === "server_spool_unavailable"
        && payload.new_frame_retry_allowed === true
      ) {
        state.pendingCapture = null;
        state.retryMode = null;
        if (!state.stream) await startCamera();
        await presentChallenge();
        return;
      }
      if (payload.retryable === true) {
        state.retryMode = "server_spool";
        ui.runMessage.textContent = (
          payload.error || "加密 server spool 尚未提交；請重試同一 server-side frame。"
        );
        ui.retryButton.hidden = false;
        return;
      }
      throw new Error(payload.error || `Server-spool recovery failed (HTTP ${response.status}).`);
    } finally {
      setBusy(false);
    }
  }

  async function captureCurrentTarget({ exactRetry = false } = {}) {
    if (state.busy || !state.challenge) return;
    cancelGateLoop();
    ui.targetDot.hidden = true;
    setBusy(true, "本機 CPU 推論中；請保持目前姿勢…");
    try {
      if (!exactRetry || !state.pendingCapture) {
        const snapshot = captureSnapshot();
        state.pendingCapture = {
          image_data: snapshot.image_data,
          capture_contract: snapshot.capture_contract,
          client_gate: globalThis.LexiGazeMeasurementGate.evidence(state.gateState),
        };
      }
      let result;
      try {
        result = await requestJson("/captures", {
          method: "POST",
          headers: authHeaders({ challenge: true }),
          body: JSON.stringify(state.pendingCapture),
        });
      } catch (_networkError) {
        ui.runMessage.textContent = "連線中斷；將用相同 frame 與 challenge 重送，不會刻意重跑推論。";
        state.retryMode = "capture";
        ui.retryButton.hidden = false;
        return;
      }
      const { response, payload } = result;
      updateProgress(payload);
      if (payload.consumed === true) {
        await advanceAfterConsumedCapture(payload);
        return;
      }
      if (globalThis.LexiGazeMeasurementClientPolicy.abortRequired(payload)) {
        showAbortRequired(payload);
        return;
      }
      if (payload.retryable === true) {
        const exactRetry = (
          globalThis.LexiGazeMeasurementClientPolicy.exactFrameRetryRequired(payload)
        );
        const serverSpoolRetry = (
          globalThis.LexiGazeMeasurementClientPolicy.serverSpoolRetryAvailable(payload)
        );
        if (!exactRetry && !serverSpoolRetry) state.pendingCapture = null;
        ui.runMessage.textContent = serverSpoolRetry
          ? (payload.error || "Server 已保留加密同一幀；重試時不會送新 frame。")
          : exactRetry
          ? (payload.error || "prepared frame 尚未提交；將保留同一 frame 做 exact retry。")
          : (payload.error || "pre-inference hard error；目前 challenge 未消耗，可以重新擷取。")
        state.retryMode = serverSpoolRetry ? "server_spool" : "capture";
        ui.retryButton.hidden = false;
        return;
      }
      throw new Error(payload.error || `擷取失敗（HTTP ${response.status}）。`);
    } finally {
      setBusy(false);
    }
  }

  async function retryCurrentTarget() {
    ui.retryButton.hidden = true;
    if (state.retryMode === "server_spool") {
      await resumeServerSpool();
      return;
    }
    if (state.pendingCapture) await captureCurrentTarget({ exactRetry: true });
    else await showTargetAndGate();
  }

  async function retryOrRecover() {
    if (state.retryMode === "recover") {
      ui.retryButton.hidden = true;
      await resumeRun();
      return;
    }
    await retryCurrentTarget();
  }

  async function finalizeCalibration() {
    setBusy(true, "正在 CPU-only 訓練並封存 65-row calibration；這可能需要幾分鐘…");
    try {
      const { response, payload } = await requestJson("/calibration/finalize", {
        method: "POST",
        headers: authHeaders(),
        body: "{}",
      });
      if (globalThis.LexiGazeMeasurementClientPolicy.isCalibrationUnusableNegative(payload)) {
        showCalibrationNegative(payload);
        return;
      }
      if (!response.ok || payload.ok !== true) {
        throw new Error(payload.error || "calibration 無法封存；本次負面結果已保留。");
      }
      updateProgress(payload);
      await issueChallenge();
    } finally {
      setBusy(false);
    }
  }

  function showCompletion(payload) {
    cancelGateLoop();
    state.challenge = null;
    state.pendingCapture = null;
    persistContext();
    ui.setupPanel.hidden = true;
    ui.blockPanel.hidden = true;
    ui.runPanel.hidden = true;
    ui.completionPanel.hidden = false;
    ui.completionTitle.textContent = "本次 193-row 擷取已結束";
    ui.verifyArtifactButton.hidden = false;
    ui.analyzeRunButton.hidden = payload.acquisition_artifact_verified !== true;
    ui.analysisSummary.hidden = true;
    ui.refreshStatusButton.hidden = false;
    ui.abortButton.hidden = (
      payload.phase === "artifact_verified"
      || payload.acquisition_artifact_verified === true
    );
    ui.completionMessage.textContent = payload.acquisition_artifact_verified === true
      ? "193-row acquisition artifact 已通過結構驗證；這仍不是 gaze 準確性通過。"
      : "193 列已封存，尚待獨立 artifact 驗證。";
    updateProgress(payload);
  }

  function showCalibrationNegative(payload) {
    cancelGateLoop();
    state.challenge = null;
    state.pendingCapture = null;
    persistContext();
    ui.setupPanel.hidden = true;
    ui.blockPanel.hidden = true;
    ui.runPanel.hidden = true;
    ui.completionPanel.hidden = false;
    ui.completionTitle.textContent = "Calibration 保留為 terminal negative result";
    const negative = (
      globalThis.LexiGazeMeasurementClientPolicy.calibrationNegativeDisplay(payload)
    );
    const attempts = negative.attempts || Number(payload.progress?.calibration_count ?? 65);
    const usable = negative.usable;
    const purge = negative.purge_verified ? "已驗證" : "尚未驗證";
    ui.completionMessage.textContent = (
      `classification=calibration_unusable_negative_result；attempts=${attempts}，`
      + `usable=${usable}，calibration image purge=${purge}。`
      + "不進入 evaluation、不補抽直到成功，也不自動建立新 run。"
    );
    ui.verifyArtifactButton.hidden = true;
    ui.analyzeRunButton.hidden = true;
    ui.analysisSummary.hidden = true;
    ui.refreshStatusButton.hidden = false;
    ui.abortButton.hidden = payload.cleanup_verified === true;
    updateProgress(payload);
  }

  function showAbortRequired(payload) {
    cancelGateLoop();
    state.pendingCapture = null;
    persistContext();
    ui.setupPanel.hidden = true;
    ui.blockPanel.hidden = true;
    ui.runPanel.hidden = true;
    ui.completionPanel.hidden = false;
    ui.completionTitle.textContent = "Run 已停止，必須完成 authenticated cleanup";
    ui.completionMessage.textContent = (
      `classification=${payload.classification || "abort_required"}；`
      + "不可繼續擷取或自動建立新 run。Run token 已保留；請按下中止，"
      + "只有 server 回覆 cleanup_verified=true 後才會清除本分頁憑證。"
    );
    ui.verifyArtifactButton.hidden = true;
    ui.analyzeRunButton.hidden = true;
    ui.analysisSummary.hidden = true;
    ui.refreshStatusButton.hidden = false;
    ui.abortButton.hidden = false;
    updateProgress(payload);
  }

  async function verifyArtifact() {
    setBusy(true, "正在重新讀取並驗證封存 artifact…");
    try {
      const { response, payload } = await requestJson("/artifact/verify", {
        method: "POST",
        headers: authHeaders(),
        body: "{}",
      });
      if (!response.ok || payload.ok !== true) throw new Error(payload.error || "artifact 驗證失敗。");
      showCompletion(payload);
    } finally {
      setBusy(false);
    }
  }

  async function analyzeRun() {
    setBusy(true, "正在由 authenticated live runner 重新驗證並分析…");
    try {
      const { response, payload } = await requestJson("/analysis", {
        method: "POST",
        headers: authHeaders(),
        body: "{}",
      });
      if (!response.ok || payload.ok !== true) {
        throw new Error(payload.error || "描述性分析失敗。");
      }
      const selected = payload.evaluation?.selected_personal_model || {};
      const uncertainty = payload.uncertainty || {};
      ui.analysisSummary.textContent = [
        `status=${payload.status}`,
        `successful=${selected.successful_count ?? "not evaluable"}/128`,
        `median_error_px=${selected.median_spatial_error_px ?? "not evaluable"}`,
        `p90_error_px=${selected.p90_spatial_error_px ?? "not evaluable"}`,
        `uncertainty=${uncertainty.status || "not evaluable"}`,
        `analysis_sha256=${payload.analysis_sha256}`,
        "measurement_claim_authorized=false",
        "threshold_selected=false",
      ].join("\n");
      ui.analysisSummary.hidden = false;
      ui.analyzeRunButton.hidden = true;
      ui.completionMessage.textContent = (
        "描述性分析已完成；數值只適用於本次 frozen 193-row self-development run，"
        + "不代表自然閱讀、族群準確率或 quality-band promotion。"
      );
    } finally {
      setBusy(false);
    }
  }

  async function abortRun() {
    if (!state.credentials) return;
    setBusy(true, "正在中止本次量測並清除專用暫存影像…");
    try {
      const { response, payload } = await requestJson("/abort", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ reason: "operator_aborted_browser_measurement" }),
      });
      if (!globalThis.LexiGazeMeasurementClientPolicy.cleanupConfirmed(response.ok, payload)) {
        throw new Error(
          payload.error
          || "abort/cleanup 未獲 server 明確驗證；run token 已保留，可再次重試。",
        );
      }
      state.credentials = null;
      state.pendingCreate = null;
      state.recoverableContext = null;
      state.challenge = null;
      state.pendingCapture = null;
      persistContext();
      globalThis.location.reload();
    } finally {
      setBusy(false);
    }
  }

  function showError(error) {
    const message = error instanceof Error ? error.message : String(error);
    ui.runMessage.textContent = message;
    if (state.challenge) {
      state.retryMode = "capture";
      ui.retryButton.textContent = "重試目前目標";
      ui.retryButton.hidden = false;
    } else if (state.credentials || state.pendingCreate || state.recoverableContext) {
      state.retryMode = "recover";
      ui.retryButton.textContent = "核對狀態並恢復";
      ui.retryButton.hidden = false;
      ui.runPanel.hidden = false;
    } else {
      state.retryMode = null;
      ui.retryButton.hidden = true;
    }
  }

  async function guarded(action) {
    try {
      await action();
    } catch (error) {
      showError(error);
      setBusy(false);
    }
  }

  ui.startCameraButton.addEventListener("click", () => guarded(startCamera));
  ui.preflightButton.addEventListener("click", () => guarded(runReadinessPreflight));
  ui.startRunButton.addEventListener("click", () => guarded(createRun));
  ui.resumeRunButton.addEventListener("click", () => guarded(resumeRun));
  ui.confirmBlockButton.addEventListener("click", () => guarded(confirmBlock));
  ui.retryButton.addEventListener("click", () => guarded(retryOrRecover));
  ui.verifyArtifactButton.addEventListener("click", () => guarded(verifyArtifact));
  ui.analyzeRunButton.addEventListener("click", () => guarded(analyzeRun));
  ui.refreshStatusButton.addEventListener("click", () => guarded(async () => {
    const current = await status();
    if (globalThis.LexiGazeMeasurementClientPolicy.isCalibrationUnusableNegative(current)) {
      showCalibrationNegative(current);
    } else if (globalThis.LexiGazeMeasurementClientPolicy.abortRequired(current)) {
      showAbortRequired(current);
    } else {
      showCompletion(current);
    }
  }));
  ui.abortButton.addEventListener("click", () => guarded(abortRun));
  globalThis.addEventListener("beforeunload", () => {
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  });

  const stored = loadStoredContext();
  if (stored) {
    state.recoverableContext = stored;
    if (globalThis.LexiGazeMeasurementClientPolicy.isPendingCreate(stored)) {
      state.pendingCreate = {
        createRequestId: stored.create_request_id,
        runToken: stored.run_token,
        preflightToken: stored.preflight_token,
        preflightReplacementRequired: stored.preflight_replacement_required === true,
        captureContract: stored.capture_contract,
        viewport: stored.viewport,
      };
    }
    ui.resumeRunButton.hidden = false;
    ui.resumeRunButton.textContent = (
      globalThis.LexiGazeMeasurementClientPolicy.isInvalidStoredContext(stored)
        ? "Recovery context 損壞（禁止新建）"
        : globalThis.LexiGazeMeasurementClientPolicy.isPendingCreate(stored)
        ? "恢復建立中的量測"
        : (stored.challenge_token
          ? "繼續本分頁的目前目標"
          : "核對並恢復本分頁的量測")
    );
  }
  updateButtons();
})();
