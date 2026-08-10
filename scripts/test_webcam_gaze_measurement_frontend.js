"use strict";

const assert = require("node:assert/strict");
const gate = require("../web/static/measurement_ceiling_gate.js");
const policy = require("../web/static/measurement_ceiling_client_policy.js");

const row = {
  sequence_index: 12,
  block_id: "calibration_neutral",
  block_role: "calibration",
  posture: "neutral",
  distance: "nominal",
  target_x_viewport_fraction: 0.25,
  target_y_viewport_fraction: 0.75,
};
const viewport = gate.frozenViewport(1000, 800, 1.25);
const rect = { left: 238, top: 588, width: 24, height: 24 };

assert.deepEqual(gate.targetCenter(row, viewport), { x: 250, y: 600 });
let state = gate.begin(row, viewport, 0);
state = gate.observe(state, row, {
  now_ms: 100,
  visibility_state: "visible",
  document_focused: true,
  viewport,
  target_rect: rect,
});
assert.equal(state.ready, false);
assert.equal(state.stable_frame_count, 1);
state = gate.observe(state, row, {
  now_ms: 1000,
  visibility_state: "visible",
  document_focused: true,
  viewport,
  target_rect: rect,
});
assert.equal(state.ready, true);
const evidence = gate.evidence(state);
assert.equal(evidence.minimum_dwell_ms, 900);
assert.equal(evidence.visual_attestation_claimed, false);
assert.equal(evidence.structural_browser_gate_only, true);

state = gate.observe(state, row, {
  now_ms: 1010,
  visibility_state: "hidden",
  document_focused: true,
  viewport,
  target_rect: rect,
});
assert.equal(state.ready, false);
assert.equal(state.stable_since_ms, null);
assert.equal(state.reason, "document_hidden");

state = gate.observe(state, row, {
  now_ms: 1020,
  visibility_state: "visible",
  document_focused: false,
  viewport,
  target_rect: rect,
});
assert.equal(state.reason, "document_unfocused");

state = gate.observe(state, row, {
  now_ms: 1030,
  visibility_state: "visible",
  document_focused: true,
  viewport: { ...viewport, width: 999 },
  target_rect: rect,
});
assert.equal(state.reason, "viewport_changed");

state = gate.observe(state, row, {
  now_ms: 1040,
  visibility_state: "visible",
  document_focused: true,
  viewport,
  target_rect: { ...rect, left: 250 },
});
assert.equal(state.reason, "target_render_mismatch");

assert.throws(() => gate.evidence(state), /not ready/);
assert.throws(
  () => gate.validateScheduleRow({ ...row, sequence_index: 193 }),
  /outside the frozen 193-row schedule/,
);

assert.equal(policy.blocksNewRun({ run_id: "old" }, null), true);
assert.equal(policy.blocksNewRun(null, { runId: "active" }), true);
assert.equal(policy.blocksNewRun(null, null), false);
assert.equal(policy.cleanupConfirmed(true, { ok: true, cleanup_verified: true }), true);
assert.equal(policy.cleanupConfirmed(true, { ok: true, cleanup_verified: false }), false);
assert.equal(policy.cleanupConfirmed(false, { ok: true, cleanup_verified: true }), false);
assert.equal(
  policy.isCalibrationUnusableNegative({
    classification: "calibration_unusable_negative_result",
  }),
  true,
);
assert.equal(policy.exactFrameRetryRequired({ exact_frame_retry_required: true }), true);
assert.equal(policy.exactFrameRetryRequired({ prepared: true }), true);
assert.equal(policy.exactFrameRetryRequired({ retryable: true }), false);
assert.equal(policy.abortRequired({ abort_required: true }), true);
assert.equal(policy.abortRequired({ classification: "abort_required" }), true);
assert.equal(policy.abortRequired({ retryable: true }), false);
let randomCall = 0;
const authority = policy.newCreateAuthority((bytes) => {
  bytes.fill(randomCall === 0 ? 0xab : 0xcd);
  randomCall += 1;
});
assert.equal(authority.create_request_id, `WGMCREQ-${"ab".repeat(16)}`);
assert.equal(authority.run_token, `wgmr_client_${"cd".repeat(32)}`);
assert.equal(policy.isPendingCreate({ state: "pending_create", ...authority }), true);
assert.equal(policy.isPendingCreate({ state: "active_run", ...authority }), false);
assert.equal(policy.isInvalidStoredContext({ state: "invalid_stored_context" }), true);
assert.equal(
  policy.canReplacePendingPreflight({
    state: "pending_create",
    ...authority,
    preflight_replacement_required: true,
  }),
  true,
);
assert.equal(
  policy.canReplacePendingPreflight({ state: "pending_create", ...authority }),
  false,
);
assert.equal(
  policy.serverSpoolRetryAvailable({ server_spool_retry_available: true }),
  true,
);
assert.equal(
  policy.serverSpoolRetryAvailable({ prepared_observation_pending: true }),
  true,
);
assert.equal(policy.serverSpoolRetryAvailable({ status: "committed" }), true);
assert.equal(policy.serverSpoolRetryAvailable({ retryable: true }), false);
assert.deepEqual(
  policy.calibrationNegativeDisplay({
    calibration_attempt_count: 65,
    usable_calibration_count: 63,
    images_purged: true,
    cleanup_verified: true,
  }),
  { attempts: 65, usable: 63, purge_verified: true },
);

console.log("webcam gaze measurement frontend gate tests passed");
