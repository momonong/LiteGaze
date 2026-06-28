(function () {
  const els = {
    openPage: document.getElementById("openGazePageBtn"),
    modelSelect: document.getElementById("gazeModelSelect"),
    toggle: document.getElementById("gazeToggle"),
    toggleLabel: document.getElementById("gazeToggleLabel"),
    debounceMode: document.getElementById("gazeDebounceMode"),
    smoothSlider: document.getElementById("gazeSmoothSlider"),
    smoothValue: document.getElementById("gazeSmoothValue"),
    corridorInput: document.getElementById("gazeCorridorInput"),
    dwellInput: document.getElementById("gazeDwellInput"),
    intervalSlider: document.getElementById("gazeIntervalSlider"),
    intervalValue: document.getElementById("gazeIntervalValue"),
    status: document.getElementById("gazeStatus"),
    gazeCursor: document.getElementById("gaze-cursor"),
  };

  if (!els.toggle) return;

  const state = {
    enabled: false,
    stream: null,
    video: document.createElement("video"),
    canvas: document.createElement("canvas"),
    lastLockedY: null,
    dwellCandidate: null,
    dwellSince: 0,
    filterX: null,
    filterY: null,
  };

  // ── Gaze Buffer & History ─────────────────────────────────────────────
  // Accumulates per-word dwell and fixation counts during a reading session.
  // Flushed to POST /api/fuse when the user triggers "Export" or session ends.
  const gazeBuffer = {};   // { wordKey: { word, dwell_count, fixation_count, confidence } }
  const gazeHistory = [];  // Chronological trace: [ { word, index, confidence, timestamp_ms } ]
  let _lastGazeWord = null;  // tracks previous word to detect new fixations

  // ── Mouse Cursor Ground Truth Tracking ──────────────────────────────
  const cursorGazePairs = [];
  let lastMouseX = 0;
  let lastMouseY = 0;
  window.addEventListener("mousemove", (e) => {
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  function recordGazeHit(word, confidence, index) {
    const key = word.toLowerCase();
    if (!gazeBuffer[key]) {
      gazeBuffer[key] = { word, dwell_count: 0, fixation_count: 0, confidence };
    }
    gazeBuffer[key].dwell_count += 1;
    if (_lastGazeWord !== key) {
      gazeBuffer[key].fixation_count += 1;
      _lastGazeWord = key;
    }
    const rank = { high: 2, medium: 1, low: 0 };
    if ((rank[confidence] || 0) > (rank[gazeBuffer[key].confidence] || 0)) {
      gazeBuffer[key].confidence = confidence;
    }

    // Append to chronological sequence log
    gazeHistory.push({
      word: word,
      index: typeof index === "number" ? index : -1,
      confidence: confidence,
      timestamp_ms: Date.now()
    });
  }

  function flushGazeBuffer() {
    return Object.values(gazeBuffer).map(entry => ({
      word:           entry.word,
      confidence:     entry.confidence,
      dwell_count:    entry.dwell_count,
      fixation_count: entry.fixation_count,
      timestamp_ms:   Date.now(),
    }));
  }

  function flushGazeHistory() {
    return [...gazeHistory];
  }

  function clearGazeBuffer() {
    Object.keys(gazeBuffer).forEach(k => delete gazeBuffer[k]);
    _lastGazeWord = null;
    gazeHistory.length = 0;
  }

  window.gazeBuffer        = gazeBuffer;
  window.gazeHistory       = gazeHistory;
  window.flushGazeBuffer   = flushGazeBuffer;
  window.flushGazeHistory  = flushGazeHistory;
  window.clearGazeBuffer   = clearGazeBuffer;
  window.recordGazeHit     = recordGazeHit;

  class LowPassFilter {
    constructor(alpha) {
      this.alpha = alpha;
      this.value = null;
    }

    filter(value) {
      if (this.value === null) {
        this.value = value;
        return value;
      }
      this.value = this.value + this.alpha * (value - this.value);
      return this.value;
    }

    reset() {
      this.value = null;
    }
  }

  function setStatus(message) {
    if (els.status) {
      els.status.textContent = message;
    }
    const guideRtStatus = document.getElementById("guideRtStatus");
    if (guideRtStatus) {
      guideRtStatus.textContent = `狀態: ${message}`;
      const cameraTip = document.getElementById("guideRtCameraTip");
      if (message.includes("失敗") || message.includes("不支援") || message.includes("錯誤")) {
        guideRtStatus.style.color = "var(--danger)";
        guideRtStatus.style.backgroundColor = "var(--danger-soft)";
        if (cameraTip) cameraTip.style.display = "block";
      } else {
        guideRtStatus.style.color = "var(--accent)";
        guideRtStatus.style.backgroundColor = "var(--accent-soft)";
        if (cameraTip) cameraTip.style.display = "none";
      }
    }
  }

  function updateSmoothLabel() {
    els.smoothValue.textContent = els.smoothSlider.value;
    const alpha = 0.08 + (Number(els.smoothSlider.value) / 100) * 0.42;
    if (state.filterX) state.filterX.alpha = alpha;
    if (state.filterY) state.filterY.alpha = alpha;
  }

  async function refreshModels() {
    try {
      const active = els.modelSelect ? els.modelSelect.value : "before";
      const res = await fetch("/api/gaze/models");
      const data = await res.json();
      const selects = document.querySelectorAll(".gaze-model-select");
      selects.forEach(select => {
        select.innerHTML = '<option value="before">原始模型 / before</option>';
        if (data.ok) {
          data.models.forEach((model) => {
            const option = document.createElement("option");
            option.value = model.name;
            option.textContent = model.display_name;
            select.appendChild(option);
          });
        }
        // Restore previous selection if still available
        if (active !== 'before' && [...select.options].some((o) => o.value === active)) {
          select.value = active;
        } else if (active === 'before' && data.ok && data.models?.length > 0) {
          // Auto-select the most recent model when none was selected yet
          select.value = data.models[data.models.length - 1].name;
        }
      });

      // If the user navigated to /gaze and came back, auto-confirm step 2
      // (they went there specifically to train a model — count as active selection)
      if (sessionStorage.getItem('lexiWentToGazePage') === '1' && data.ok && data.models?.length > 0) {
        sessionStorage.removeItem('lexiWentToGazePage');
        sessionStorage.setItem('lexiModelSelectedThisSession', '1');
        window.lexiModelSelectedThisSession = true;
      }

      setStatus(`模型清單已更新，共 ${data.models?.length || 0} 個選項`);
    } catch (err) {
      setStatus(`模型清單載入失敗：${err.message}`);
    }
  }

  async function startCamera() {
    console.log("[Gaze] startCamera request. navigator.mediaDevices:", !!navigator.mediaDevices);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error("[Gaze] navigator.mediaDevices.getUserMedia is not available");
      throw new Error("不支援相機存取");
    }
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      console.log("[Gaze] getUserMedia stream acquired successfully:", state.stream.id);
    } catch (exc) {
      console.error("[Gaze] getUserMedia failed:", exc);
      throw exc;
    }
    state.video.srcObject = state.stream;
    state.video.muted = true;
    state.video.playsInline = true;
    await state.video.play();

    // Stream to guide preview circle if it exists
    const guideVideo = document.getElementById('guideWebcamVideo');
    if (guideVideo) {
      guideVideo.srcObject = state.stream;
      const previewRow = document.getElementById('guideWebcamPreviewRow');
      if (previewRow) previewRow.style.display = 'flex';
    }
    // Stream to gaze panel preview circle if it exists
    const gazeVideo = document.getElementById('gazeWebcamVideo');
    if (gazeVideo) {
      gazeVideo.srcObject = state.stream;
      const gazePreviewRow = document.getElementById('gazeWebcamPreviewRow');
      if (gazePreviewRow) gazePreviewRow.style.display = 'flex';
    }
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    state.video.srcObject = null;

    // Reset guide preview circle
    const guideVideo = document.getElementById('guideWebcamVideo');
    if (guideVideo) {
      guideVideo.srcObject = null;
      const previewRow = document.getElementById('guideWebcamPreviewRow');
      if (previewRow) previewRow.style.display = 'none';
    }
    // Reset gaze preview circle
    const gazeVideo = document.getElementById('gazeWebcamVideo');
    if (gazeVideo) {
      gazeVideo.srcObject = null;
      const gazePreviewRow = document.getElementById('gazeWebcamPreviewRow');
      if (gazePreviewRow) gazePreviewRow.style.display = 'none';
    }
  }

  function captureFrame() {
    const width = 240;
    const aspect = state.video.videoHeight ? state.video.videoWidth / state.video.videoHeight : 4 / 3;
    const height = Math.round(width / aspect);
    state.canvas.width = width;
    state.canvas.height = height;
    const ctx = state.canvas.getContext("2d");
    ctx.drawImage(state.video, 0, 0, width, height);
    return state.canvas.toDataURL("image/jpeg", 0.5);
  }

  async function predict(signal) {
    const res = await fetch("/api/gaze/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: captureFrame(),
        model_name: els.modelSelect.value,
        viewport_width: window.innerWidth,
        viewport_height: window.innerHeight,
      }),
      signal,
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data.screen_xy_px || [
      ((data.screen_xy_norm[0] + 1) * 0.5) * window.innerWidth,
      ((data.screen_xy_norm[1] + 1) * 0.5) * window.innerHeight,
    ];
  }

  function applyDebounce(x, y) {
    const mode = els.debounceMode.value;
    let finalX = x;
    let finalY = y;

    if (mode === "one-euro" || mode === "one-euro-corridor") {
      finalX = state.filterX.filter(finalX);
      finalY = state.filterY.filter(finalY);
    } else {
      state.filterX.reset();
      state.filterY.reset();
    }

    if (mode === "corridor" || mode === "one-euro-corridor") {
      const corridorHeight = Number(els.corridorInput.value) || 35;
      if (state.lastLockedY === null) {
        state.lastLockedY = finalY;
      } else if (Math.abs(finalY - state.lastLockedY) <= corridorHeight) {
        finalY = state.lastLockedY;
      } else {
        state.lastLockedY = finalY;
      }
    } else {
      state.lastLockedY = null;
    }

    if (mode === "dwell") {
      const now = performance.now();
      const dwellMs = Number(els.dwellInput.value) || 180;
      const candidateDistance = state.dwellCandidate
        ? Math.hypot(finalX - state.dwellCandidate.x, finalY - state.dwellCandidate.y)
        : Infinity;
      if (!state.dwellCandidate || candidateDistance > 32) {
        state.dwellCandidate = { x: finalX, y: finalY };
        state.dwellSince = now;
        return null;
      }
      if (now - state.dwellSince < dwellMs) return null;
      finalX = state.dwellCandidate.x;
      finalY = state.dwellCandidate.y;
    } else {
      state.dwellCandidate = null;
      state.dwellSince = 0;
    }

    return { x: finalX, y: finalY };
  }

  async function loop() {
    while (state.enabled) {
      if (document.hidden) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        continue;
      }

      const ac = new AbortController();
      state._currentAbort = ac;
      const timeout = window.setTimeout(() => ac.abort(), 5000);

      try {
        const [x, y] = await predict(ac.signal);
        clearTimeout(timeout);
        state._currentAbort = null;

        const point = applyDebounce(x, y);
        if (point) {
          if (els.gazeCursor) {
            els.gazeCursor.style.left = `${point.x}px`;
            els.gazeCursor.style.top = `${point.y}px`;
          }
          // Log paired cursor and predicted gaze coordinates
          cursorGazePairs.push({
            timestamp_ms: Date.now(),
            cursor_x: lastMouseX,
            cursor_y: lastMouseY,
            gaze_x: Math.round(point.x),
            gaze_y: Math.round(point.y)
          });
          if (typeof window.processGazeOnExtractedData === "function") {
            window.processGazeOnExtractedData(point.x, point.y);
            setStatus(`啟用中：${Math.round(point.x)}, ${Math.round(point.y)} (${els.modelSelect.value})`);
          }
        }
      } catch (err) {
        clearTimeout(timeout);
        state._currentAbort = null;
        if (!state.enabled) {
          break;
        }
        if (err.name === "AbortError") {
          setStatus("推論超時，重試中...");
        } else {
          setStatus(`推論失敗：${err.message}`);
        }
        await new Promise((resolve) => window.setTimeout(resolve, 300));
        continue;
      }

      await new Promise((resolve) => window.setTimeout(resolve, els.intervalSlider ? Number(els.intervalSlider.value) : 90));
    }
  }

  // ── Fusion export ─────────────────────────────────────────────────────────
  async function exportFusion(cognitiveResult, sessionId, persist = false) {
    const events = flushGazeBuffer();
    if (events.length === 0) {
      console.warn("[Fusion] gazeBuffer is empty — nothing to fuse");
      return null;
    }
    try {
      const res = await fetch("/api/fuse/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id:       sessionId || `session_${Date.now()}`,
          persist:          persist,
          cognitive_result: cognitiveResult || {},
          gaze_events:      events,
        }),
      });
      const data = await res.json();
      console.log("[Fusion] RDS result:", data);
      return data;
    } catch (err) {
      console.error("[Fusion] POST /api/fuse failed:", err);
      return null;
    }
  }

  window.exportFusion = exportFusion;

  async function setEnabled(enabled) {
    state.enabled = enabled;
    els.toggle.classList.toggle("active", enabled);
    els.toggleLabel.textContent = enabled ? "啟用眼動推論（開啟）" : "啟用眼動推論（關閉）";

    const guideBtn = document.getElementById("guideRtToggleBtn");
    if (guideBtn) {
      guideBtn.textContent = enabled ? "關閉相機" : "開啟相機";
      guideBtn.className = enabled ? "btn btn-danger" : "btn btn-secondary";
    }

    if (enabled) {
      const mappingToggle = document.getElementById("gazeMappingToggle");
      if (mappingToggle && !window.gazeMappingOn) {
        mappingToggle.click();
      }

      if (els.gazeCursor) {
        els.gazeCursor.style.display = "block";
      }

      try {
        await startCamera();
        state.filterX = new LowPassFilter(0.08);
        state.filterY = new LowPassFilter(0.08);
        setStatus("攝影機已啟動，開始推論");
        loop();
      } catch (err) {
        state.enabled = false;
        els.toggle.classList.remove("active");
        els.toggleLabel.textContent = "啟用眼動推論（關閉）";
        if (guideBtn) {
          guideBtn.textContent = "開啟相機";
          guideBtn.className = "btn btn-secondary";
        }
        if (els.gazeCursor) {
          els.gazeCursor.style.display = "none";
        }
        setStatus(`攝影機啟動失敗：${err.message}`);
      }
    } else {
      if (state._currentAbort) {
        state._currentAbort.abort();
        state._currentAbort = null;
      }
      stopCamera();
      if (els.gazeCursor) {
        els.gazeCursor.style.display = "none";
      }
      uploadCursorGazePairs();
    }

    async function uploadCursorGazePairs() {
      if (cursorGazePairs.length === 0) {
        setStatus("已停止眼動推論");
        return;
      }
      const numPairs = cursorGazePairs.length;
      const sessId = `gt_${Date.now()}`;
      setStatus(`正在儲存 ${numPairs} 筆鼠標-眼動配對數據...`);
      try {
        const res = await fetch("/api/gaze/save_pairs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessId,
            viewport_width: window.innerWidth,
            viewport_height: window.innerHeight,
            pairs: [...cursorGazePairs]
          })
        });
        const data = await res.json();
        if (data.ok) {
          setStatus(`已停止眼動推論。已成功儲存 ${numPairs} 筆配對數據！`);
          cursorGazePairs.length = 0;
        } else {
          setStatus(`儲存配對數據失敗：${data.error}`);
        }
      } catch (err) {
        setStatus(`儲存配對數據錯誤：${err.message}`);
      }
    }

    if (typeof window.updateGuideUI === "function") {
      window.updateGuideUI();
    }
  }

  function updateIntervalLabel() {
    if (els.intervalSlider && els.intervalValue) {
      els.intervalValue.textContent = `${els.intervalSlider.value} ms`;
    }
  }

  // Set click listener on all calibration/training buttons
  document.querySelectorAll(".open-gaze-page-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      // Mark that we're leaving for the gaze training page so we can auto-confirm step 2 on return
      sessionStorage.setItem('lexiWentToGazePage', '1');
      window.location.href = "/gaze";
    });
  });

  // Synchronize model selection dropdowns
  document.querySelectorAll(".gaze-model-select").forEach(select => {
    select.addEventListener("change", (e) => {
      document.querySelectorAll(".gaze-model-select").forEach(other => {
        if (other !== e.target) other.value = e.target.value;
      });
      // Mark that the user actively selected a model — persisted in sessionStorage
      // so navigation to /gaze and back doesn't reset it
      sessionStorage.setItem('lexiModelSelectedThisSession', '1');
      window.lexiModelSelectedThisSession = true;
      if (typeof window.updateGuideUI === "function") window.updateGuideUI();
    });
  });

  const handleToggle = () => setEnabled(!state.enabled);
  els.toggle.addEventListener("click", handleToggle);

  const guideRtBtn = document.getElementById("guideRtToggleBtn");
  if (guideRtBtn) {
    guideRtBtn.addEventListener("click", handleToggle);
  }
  els.smoothSlider.addEventListener("input", updateSmoothLabel);
  if (els.intervalSlider) {
    els.intervalSlider.addEventListener("input", updateIntervalLabel);
  }

  state.filterX = new LowPassFilter(0.08);
  state.filterY = new LowPassFilter(0.08);
  updateSmoothLabel();
  updateIntervalLabel();

  // Restore session flags from sessionStorage (survive navigation within same tab)
  if (sessionStorage.getItem('lexiModelSelectedThisSession') === '1') {
    window.lexiModelSelectedThisSession = true;
  }

  refreshModels().then(() => {
    if (typeof window.updateGuideUI === "function") window.updateGuideUI();
  });
})();
