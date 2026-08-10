(function installMeasurementCeilingGate(global) {
  "use strict";

  const SCHEMA_VERSION = 1;
  const IMPLEMENTATION_ID = "browser-visible-focus-dwell-render-v1";
  const MINIMUM_DWELL_MS = 900;
  const TARGET_TOLERANCE_PX = 3;

  function finite(value, label) {
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`${label} must be finite`);
    return number;
  }

  function positive(value, label) {
    const number = finite(value, label);
    if (number <= 0) throw new Error(`${label} must be positive`);
    return number;
  }

  function frozenViewport(width, height, devicePixelRatio) {
    return Object.freeze({
      width: positive(width, "viewport width"),
      height: positive(height, "viewport height"),
      device_pixel_ratio: positive(devicePixelRatio, "device pixel ratio"),
    });
  }

  function validateScheduleRow(row) {
    if (!row || typeof row !== "object" || Array.isArray(row)) {
      throw new Error("schedule row must be an object");
    }
    const fractionX = finite(
      row.target_x_viewport_fraction,
      "target x viewport fraction",
    );
    const fractionY = finite(
      row.target_y_viewport_fraction,
      "target y viewport fraction",
    );
    if (fractionX < 0 || fractionX > 1 || fractionY < 0 || fractionY > 1) {
      throw new Error("target viewport fraction is outside [0, 1]");
    }
    const sequenceIndex = finite(row.sequence_index, "sequence index");
    if (!Number.isInteger(sequenceIndex) || sequenceIndex < 0 || sequenceIndex >= 193) {
      throw new Error("sequence index is outside the frozen 193-row schedule");
    }
    for (const field of ["block_id", "block_role", "posture", "distance"]) {
      if (typeof row[field] !== "string" || !row[field]) {
        throw new Error(`schedule row ${field} is invalid`);
      }
    }
    return row;
  }

  function targetCenter(row, viewport) {
    validateScheduleRow(row);
    return {
      x: finite(row.target_x_viewport_fraction, "target x viewport fraction")
        * positive(viewport.width, "viewport width"),
      y: finite(row.target_y_viewport_fraction, "target y viewport fraction")
        * positive(viewport.height, "viewport height"),
    };
  }

  function begin(row, viewport, nowMs = 0) {
    validateScheduleRow(row);
    const frozen = frozenViewport(
      viewport.width,
      viewport.height,
      viewport.device_pixel_ratio,
    );
    return {
      schema_version: SCHEMA_VERSION,
      implementation_id: IMPLEMENTATION_ID,
      sequence_index: Number(row.sequence_index),
      frozen_viewport: frozen,
      stable_since_ms: null,
      stable_frame_count: 0,
      last_observed_ms: finite(nowMs, "gate start time"),
      ready: false,
      reason: "waiting_for_stable_render",
      rendered_target_center_x_px: null,
      rendered_target_center_y_px: null,
      observed_stable_dwell_ms: 0,
    };
  }

  function sameViewport(frozen, observed) {
    return (
      Math.abs(finite(observed.width, "observed viewport width") - frozen.width) <= 0.5
      && Math.abs(finite(observed.height, "observed viewport height") - frozen.height) <= 0.5
      && Math.abs(
        finite(observed.device_pixel_ratio, "observed device pixel ratio")
          - frozen.device_pixel_ratio,
      ) <= 0.001
    );
  }

  function observe(state, row, observation) {
    if (!state || state.implementation_id !== IMPLEMENTATION_ID) {
      throw new Error("gate state is invalid");
    }
    validateScheduleRow(row);
    if (Number(row.sequence_index) !== state.sequence_index) {
      throw new Error("gate state belongs to a different schedule row");
    }
    const nowMs = finite(observation.now_ms, "observation time");
    if (nowMs < state.last_observed_ms) {
      throw new Error("observation time moved backwards");
    }
    const next = { ...state, last_observed_ms: nowMs, ready: false };
    const reset = (reason) => ({
      ...next,
      stable_since_ms: null,
      stable_frame_count: 0,
      observed_stable_dwell_ms: 0,
      reason,
    });
    if (observation.visibility_state !== "visible") return reset("document_hidden");
    if (observation.document_focused !== true) return reset("document_unfocused");
    if (!sameViewport(state.frozen_viewport, observation.viewport || {})) {
      return reset("viewport_changed");
    }
    const rect = observation.target_rect || {};
    const renderedX = finite(rect.left, "target rect left")
      + positive(rect.width, "target rect width") / 2;
    const renderedY = finite(rect.top, "target rect top")
      + positive(rect.height, "target rect height") / 2;
    const expected = targetCenter(row, state.frozen_viewport);
    if (
      Math.abs(renderedX - expected.x) > TARGET_TOLERANCE_PX
      || Math.abs(renderedY - expected.y) > TARGET_TOLERANCE_PX
    ) {
      return reset("target_render_mismatch");
    }
    const stableSince = state.stable_since_ms === null ? nowMs : state.stable_since_ms;
    const stableFrames = state.stable_since_ms === null
      ? 1
      : state.stable_frame_count + 1;
    const dwellMs = Math.max(0, nowMs - stableSince);
    return {
      ...next,
      stable_since_ms: stableSince,
      stable_frame_count: stableFrames,
      rendered_target_center_x_px: renderedX,
      rendered_target_center_y_px: renderedY,
      observed_stable_dwell_ms: dwellMs,
      ready: stableFrames >= 2 && dwellMs >= MINIMUM_DWELL_MS,
      reason: stableFrames >= 2 && dwellMs >= MINIMUM_DWELL_MS
        ? "ready"
        : "dwelling",
    };
  }

  function evidence(state) {
    if (!state || state.ready !== true) {
      throw new Error("browser gate is not ready");
    }
    return {
      schema_version: SCHEMA_VERSION,
      implementation_id: IMPLEMENTATION_ID,
      sequence_index: state.sequence_index,
      visibility_state: "visible",
      document_focused: true,
      viewport_width: state.frozen_viewport.width,
      viewport_height: state.frozen_viewport.height,
      device_pixel_ratio: state.frozen_viewport.device_pixel_ratio,
      rendered_target_center_x_px: state.rendered_target_center_x_px,
      rendered_target_center_y_px: state.rendered_target_center_y_px,
      minimum_dwell_ms: MINIMUM_DWELL_MS,
      observed_stable_dwell_ms: state.observed_stable_dwell_ms,
      stable_render_frame_count: state.stable_frame_count,
      structural_browser_gate_only: true,
      visual_attestation_claimed: false,
    };
  }

  const api = Object.freeze({
    SCHEMA_VERSION,
    IMPLEMENTATION_ID,
    MINIMUM_DWELL_MS,
    TARGET_TOLERANCE_PX,
    frozenViewport,
    validateScheduleRow,
    targetCenter,
    begin,
    observe,
    evidence,
  });
  global.LexiGazeMeasurementGate = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
