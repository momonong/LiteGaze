"use strict";

const assert = require("node:assert/strict");
const feedback = require("../web/static/gaze_calibration_feedback.js");

const message = feedback.buildFailureMessage({
  quality: {
    reasons: ["motion_coverage_gate_failed"],
    motion_audit_issues: [
      { code: "INSUFFICIENT_USABLE_SAMPLES" },
      { code: "INSUFFICIENT_DISTANCE_SEPARATION" },
    ],
  },
});
assert.match(message, /暫存校正影像已清除/);
assert.match(message, /額頭、雙眼、鼻子與下巴/);
assert.match(message, /15–20 公分/);
assert.match(message, /motion_coverage_gate_failed/);
assert.match(message, /INSUFFICIENT_USABLE_SAMPLES/);
assert.match(message, /重新開始校正/);

const unknown = feedback.buildFailureMessage({ quality: { reasons: ["new_code"] } });
assert.match(unknown, /new_code/);
assert.match(unknown, /臉完整入鏡/);

const trainingFailure = feedback.buildFailureMessage({
  quality: { reasons: ["personalization_training_failed"] },
});
assert.match(trainingFailure, /CPU 個人化沒有完成/);

const incompleteMetadata = feedback.buildFailureMessage({
  quality: {
    motion_audit_issues: [{ code: "INCOMPLETE_target_repeat_index" }],
  },
});
assert.match(incompleteMetadata, /metadata 不完整/);
assert.match(incompleteMetadata, /不要繼續閱讀/);

const noFace = feedback.noFacePrompt(2);
assert.match(noFace, /第 3 個目標/);
assert.match(noFace, /相機預覽中央/);
assert.match(noFace, /強烈背光/);

assert.deepEqual(
  feedback.qualityCodes({
    quality: {
      reasons: ["a", "a"],
      motion_audit_issues: [{ code: "b" }, null],
    },
  }),
  ["a", "b"],
);

console.log("gaze calibration feedback tests passed");
