(function installLexiGazeCapture(global) {
  "use strict";

  const SCHEMA_VERSION = 1;
  const INTENT_WIDTH_PX = 1280;
  const INTENT_HEIGHT_PX = 720;
  const INTENT_FRAME_RATE_HZ = 30;
  const MIN_WIDTH_PX = 640;
  const MIN_HEIGHT_PX = 480;
  const TRANSPORT_WIDTH_PX = 640;
  const JPEG_QUALITY = 0.8;
  const RESIZE_POLICY = "fit_width_preserve_aspect";

  function mediaConstraints() {
    return {
      video: {
        width: { ideal: INTENT_WIDTH_PX, min: MIN_WIDTH_PX },
        height: { ideal: INTENT_HEIGHT_PX, min: MIN_HEIGHT_PX },
        frameRate: { ideal: INTENT_FRAME_RATE_HZ },
        facingMode: "user",
      },
      audio: false,
    };
  }

  function positiveDimension(value, label) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) {
      throw new Error(`Invalid camera capture dimension: ${label}`);
    }
    return Math.round(number);
  }

  function frameSize(sourceWidth, sourceHeight, outputWidth = TRANSPORT_WIDTH_PX) {
    const width = positiveDimension(sourceWidth, "camera width");
    const height = positiveDimension(sourceHeight, "camera height");
    const transportWidth = positiveDimension(outputWidth, "transport width");
    return {
      width: transportWidth,
      height: Math.max(1, Math.round(transportWidth * height / width)),
    };
  }

  function captureContract(video) {
    const track = video?.srcObject?.getVideoTracks?.()[0];
    const settings = track?.getSettings?.() || {};
    const sourceWidth = positiveDimension(
      video?.videoWidth || settings.width,
      "camera width",
    );
    const sourceHeight = positiveDimension(
      video?.videoHeight || settings.height,
      "camera height",
    );
    const transport = frameSize(sourceWidth, sourceHeight);
    const frameRate = Number(settings.frameRate || 0);
    return {
      schema_version: SCHEMA_VERSION,
      intent_width_px: INTENT_WIDTH_PX,
      intent_height_px: INTENT_HEIGHT_PX,
      intent_frame_rate_hz: INTENT_FRAME_RATE_HZ,
      source_width_px: sourceWidth,
      source_height_px: sourceHeight,
      source_frame_rate_hz: Number.isFinite(frameRate) && frameRate >= 0 ? frameRate : 0,
      transport_width_px: transport.width,
      transport_height_px: transport.height,
      resize_policy: RESIZE_POLICY,
      mime_type: "image/jpeg",
      jpeg_quality: JPEG_QUALITY,
      mirror_applied: false,
      facing_mode: String(settings.facingMode || "user").toLowerCase(),
    };
  }

  function captureSnapshot(video, canvas) {
    const contract = captureContract(video);
    canvas.width = contract.transport_width_px;
    canvas.height = contract.transport_height_px;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("Camera capture canvas is unavailable");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return {
      image_data: canvas.toDataURL("image/jpeg", JPEG_QUALITY),
      capture_contract: contract,
    };
  }

  const api = Object.freeze({
    mediaConstraints,
    frameSize,
    captureContract,
    captureSnapshot,
  });
  global.LexiGazeCapture = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
