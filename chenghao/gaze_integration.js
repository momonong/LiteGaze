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
    if (state.filterX) state.filterX.alpha = alpha;
    if (state.filterY) state.filterY.alpha = alpha;
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
          if (typeof window.processGazeOnExtractedData === "function") {
            window.processGazeOnExtractedData(point.x, point.y);
            setStatus(`啟用中：${Math.round(point.x)}, ${Math.round(point.y)} (${els.modelSelect.value})`);
          }
        }
      } catch (err) {
        clearTimeout(timeout);
        state._currentAbort = null;
        if (err.name === "AbortError") {
          setStatus("推論超時，重試中...");
        } else {
          setStatus(`推論失敗：${err.message}`);
        }
        await new Promise((resolve) => window.setTimeout(resolve, 300));
        continue;
      }

      await new Promise((resolve) => window.setTimeout(resolve, 90));
    }
  }

  async function setEnabled(enabled) {
    state.enabled = enabled;
    els.toggle.classList.toggle("active", enabled);
    els.toggleLabel.textContent = enabled ? "啟用眼動推論（開啟）" : "啟用眼動推論（關閉）";

    if (enabled) {
      if (typeof gazeMappingToggle !== "undefined" && gazeMappingToggle && typeof gazeMappingOn !== "undefined" && !gazeMappingOn) {
        gazeMappingToggle.click();
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
      setStatus("已停止眼動推論");
    }
  }

  els.openPage.addEventListener("click", () => {
    window.location.href = "/gaze";
  });
  els.toggle.addEventListener("click", () => setEnabled(!state.enabled));
  els.smoothSlider.addEventListener("input", updateSmoothLabel);

  state.filterX = new LowPassFilter(0.08);
  state.filterY = new LowPassFilter(0.08);
  updateSmoothLabel();
  refreshModels();
})();
