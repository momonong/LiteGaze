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

  // ── Gaze Buffer (for fusion) ─────────────────────────────────────────────
  // Accumulates per-word dwell and fixation counts during a reading session.
  // Flushed to POST /api/fuse when the user triggers "Export" or session ends.
  const gazeBuffer = {};   // { wordKey: { word, dwell_count, fixation_count, confidence } }
  let _lastGazeWord = null;  // tracks previous word to detect new fixations

  function recordGazeHit(word, confidence) {
    const key = word.toLowerCase();
    if (!gazeBuffer[key]) {
      gazeBuffer[key] = { word, dwell_count: 0, fixation_count: 0, confidence };
    }
    gazeBuffer[key].dwell_count += 1;
    // A new fixation is counted when the gaze moves to a different word
    if (_lastGazeWord !== key) {
      gazeBuffer[key].fixation_count += 1;
      _lastGazeWord = key;
    }
    // Keep the highest-confidence label seen for this word
    const rank = { high: 2, medium: 1, low: 0 };
    if ((rank[confidence] || 0) > (rank[gazeBuffer[key].confidence] || 0)) {
      gazeBuffer[key].confidence = confidence;
    }
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

  function clearGazeBuffer() {
    Object.keys(gazeBuffer).forEach(k => delete gazeBuffer[k]);
    _lastGazeWord = null;
  }

  // Expose for use by word_track.html inline script
  window.gazeBuffer        = gazeBuffer;
  window.flushGazeBuffer   = flushGazeBuffer;
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
    els.status.textContent = message;
  }

  function updateSmoothLabel() {
    els.smoothValue.textContent = els.smoothSlider.value;
    const alpha = 0.08 + (Number(els.smoothSlider.value) / 100) * 0.42;
    state.filterX = new LowPassFilter(alpha);
    state.filterY = new LowPassFilter(alpha);
  }

  async function refreshModels() {
    try {
      const active = els.modelSelect.value;
      const res = await fetch("/api/gaze/models");
      const data = await res.json();
      els.modelSelect.innerHTML = '<option value="before">原始模型 / before</option>';
      if (data.ok) {
        data.models.forEach((model) => {
          const option = document.createElement("option");
          option.value = model.name;
          option.textContent = model.display_name;
          els.modelSelect.appendChild(option);
        });
      }
      if ([...els.modelSelect.options].some((option) => option.value === active)) {
        els.modelSelect.value = active;
      }
      setStatus(`模型清單已更新，共 ${els.modelSelect.options.length} 個選項`);
    } catch (err) {
      setStatus(`模型清單載入失敗：${err.message}`);
    }
  }

  async function startCamera() {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    state.video.srcObject = state.stream;
    state.video.muted = true;
    state.video.playsInline = true;
    await state.video.play();
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    state.video.srcObject = null;
  }

  function captureFrame() {
    const width = 360;
    const aspect = state.video.videoHeight ? state.video.videoWidth / state.video.videoHeight : 4 / 3;
    const height = Math.round(width / aspect);
    state.canvas.width = width;
    state.canvas.height = height;
    const ctx = state.canvas.getContext("2d");
    ctx.drawImage(state.video, 0, 0, width, height);
    return state.canvas.toDataURL("image/jpeg", 0.75);
  }

  async function predict() {
    const res = await fetch("/api/gaze/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: captureFrame(),
        model_name: els.modelSelect.value,
        viewport_width: window.innerWidth,
        viewport_height: window.innerHeight,
      }),
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
      try {
        const [x, y] = await predict();
        const point = applyDebounce(x, y);
        if (point) {
          // Update visual gaze cursor position
          if (els.gazeCursor) {
            els.gazeCursor.style.left = `${point.x}px`;
            els.gazeCursor.style.top = `${point.y}px`;
          }
          if (typeof window.processGazeOnExtractedData === "function") {
            window.processGazeOnExtractedData(point.x, point.y);
            setStatus(`啟用中：${Math.round(point.x)}, ${Math.round(point.y)} (${els.modelSelect.value})`);
          }
        }
      } catch (err) {
        setStatus(`推論失敗：${err.message}`);
        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
      await new Promise((resolve) => window.setTimeout(resolve, 120));
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

    if (enabled) {
      // Auto-enable Gaze Mapping highlights if currently disabled
      if (typeof gazeMappingToggle !== "undefined" && typeof gazeMappingOn !== "undefined" && !gazeMappingOn) {
        gazeMappingToggle.click();
      }
      
      // Show visual cursor
      if (els.gazeCursor) {
        els.gazeCursor.style.display = "block";
      }

      try {
        await startCamera();
        setStatus("攝影機已啟動，開始推論");
        loop();
      } catch (err) {
        state.enabled = false;
        els.toggle.classList.remove("active");
        els.toggleLabel.textContent = "啟用眼動推論（關閉）";
        if (els.gazeCursor) {
          els.gazeCursor.style.display = "none";
        }
        setStatus(`攝影機啟動失敗：${err.message}`);
      }
    } else {
      stopCamera();
      // Hide visual cursor
      if (els.gazeCursor) {
        els.gazeCursor.style.display = "none";
      }
      setStatus("已停止眼動推論");
    }
  }

  els.openPage.addEventListener("click", () => {
    window.location.href = "/gaze";
  });
  els.toggle.addEventListener("click", () => setEnabled(!state.enabled));
  els.smoothSlider.addEventListener("input", updateSmoothLabel);
  els.modelSelect.addEventListener("focus", refreshModels);

  updateSmoothLabel();
  refreshModels();
})();
