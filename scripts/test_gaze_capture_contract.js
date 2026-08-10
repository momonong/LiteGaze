"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const capture = require("../web/static/gaze_capture_contract.js");
const measurement = require("../core/gaze_core/participant_gaze_measurement_contract_v1.json");

assert.deepEqual(capture.frameSize(1280, 720), { width: 640, height: 360 });
assert.deepEqual(capture.frameSize(640, 480), { width: 640, height: 480 });

const drawCalls = [];
const canvas = {
  width: 640,
  height: 480,
  getContext() {
    return {
      drawImage(...args) {
        drawCalls.push(args);
      },
    };
  },
  toDataURL(mimeType, quality) {
    assert.equal(mimeType, "image/jpeg");
    assert.equal(quality, 0.8);
    return "data:image/jpeg;base64,fixture";
  },
};
const video = {
  videoWidth: 1280,
  videoHeight: 720,
  srcObject: {
    getVideoTracks() {
      return [{ getSettings: () => ({ width: 1280, height: 720, frameRate: 29.97 }) }];
    },
  },
};

const snapshot = capture.captureSnapshot(video, canvas);
assert.equal(canvas.width, 640);
assert.equal(canvas.height, 360);
assert.equal(snapshot.capture_contract.source_width_px, 1280);
assert.equal(snapshot.capture_contract.transport_height_px, 360);
assert.deepEqual(drawCalls[0].slice(1), [0, 0, 640, 360]);

const constraints = capture.mediaConstraints();
assert.equal(constraints.video.width.ideal, 1280);
assert.equal(constraints.video.width.min, 640);
assert.equal(constraints.video.height.ideal, 720);
assert.equal(constraints.video.height.min, 480);
assert.equal(constraints.video.frameRate.ideal, 30);

const captureContract = measurement.capture_contract;
assert.equal(captureContract.exact_source_resolution_must_match, false);
assert.deepEqual(
  captureContract.must_match_between_calibration_validation_and_reading,
  [
    "source_aspect_ratio_within_0.02",
    "transport_aspect_ratio_within_0.02",
    "resize_policy",
    "mirror_policy",
    "facing_mode",
  ],
);
const targetContract = measurement.target_independence;
const calibrationContract = measurement.participant_calibration;
assert.equal(calibrationContract.frozen_targets.length, 13);
assert.equal(calibrationContract.motion_blocks.length, 5);
assert.equal(calibrationContract.collection_protocol, "motion-diverse-v1");
assert.equal(calibrationContract.repeats_per_target_per_block, 1);
assert.equal(calibrationContract.target_pixel_role, "client_reported_diagnostic_only");
assert.deepEqual(calibrationContract.server_overwritten_fields, [
  "phase", "point_index", "repeat_index", "target_x_norm", "target_y_norm",
  "collect_mode", "collection_protocol", "motion_block_id", "posture_condition",
  "distance_condition", "lighting_condition", "capture_burst_id",
  "calibration_label_authority", "target_pixel_role",
]);
assert.equal(targetContract.selected_validation_targets.length, 5);
assert.equal(targetContract.frozen_heldout_grid_4x4.length, 16);
assert.equal(targetContract.overlap_boundary_is_failure, false);
for (const target of targetContract.selected_validation_targets) {
  assert.ok(Math.abs(target.target_x_norm - (target.target_x_viewport_fraction * 2 - 1)) < 1e-12);
  assert.ok(Math.abs(target.target_y_norm - (target.target_y_viewport_fraction * 2 - 1)) < 1e-12);
}

const participantCollectionSource = fs.readFileSync(
  path.join(__dirname, "../web/static/participant_collection.js"),
  "utf8",
);
assert.match(participantCollectionSource, /targetIndependenceStatus === "failed"/);
assert.match(participantCollectionSource, /target-independence 驗證失敗/);
assert.match(participantCollectionSource, /fit-target provenance，target independence 無法證明/);
assert.match(participantCollectionSource, /captureContractUnavailable/);
assert.match(participantCollectionSource, /camera capture provenance，capture compatibility 無法證明/);
assert.match(participantCollectionSource, /仍可繼續收集 behavioral word-review/);
assert.match(participantCollectionSource, /function frozenAssessmentViewport\(\)/);
assert.match(participantCollectionSource, /assertAssessmentViewportStable\(\)/);
assert.match(participantCollectionSource, /assessment_viewport:/);
assert.match(
  participantCollectionSource,
  /if \(error\?\.code === "assessment_viewport_changed"\) throw error;/,
);
assert.match(participantCollectionSource, /coarse_failure_code: "viewport_contract_mismatch"/);
assert.match(participantCollectionSource, /integrity\?\.eligible === false/);
assert.match(participantCollectionSource, /gaze 已永久降級為 behavioral-only/);
assert.match(
  participantCollectionSource,
  /collection\.gaze_measurement_contract\?\.contract \|\| null/,
);
assert.match(
  participantCollectionSource,
  /const design = collection\?\.assessment_id[\s\S]*?\? null[\s\S]*?: await api\("\/api\/study\/general-collection\/protocol"\)/,
);
assert.doesNotMatch(
  participantCollectionSource,
  /viewport_width: innerWidth[\s\S]{0,80}viewport_height: innerHeight/,
);
assert.match(participantCollectionSource, /validation_phase: validationContext\.phase/);
assert.match(participantCollectionSource, /validation_target_id: validationContext\.targetId/);
assert.match(participantCollectionSource, /error\.payload = payload/);
const validationSource = participantCollectionSource.match(
  /async function runValidation\(\)[\s\S]*?\r?\n}\r?\n\r?\nfunction formatElapsed/,
)?.[0] || "";
assert.match(validationSource, /prediction_receipts: predictionReceipts/);
assert.match(validationSource, /prediction_receipts: \[\]/);
assert.match(validationSource, /prediction_receipt\?\.token/);
assert.doesNotMatch(validationSource, /predicted_x_px\s*:/);
assert.doesNotMatch(validationSource, /predicted_y_px\s*:/);
assert.doesNotMatch(validationSource, /target_x_px\s*:/);
assert.doesNotMatch(validationSource, /target_y_px\s*:/);
assert.doesNotMatch(validationSource, /capture_contract\s*:/);

const gazePageSource = fs.readFileSync(
  path.join(__dirname, "../web/static/gaze_page.js"),
  "utf8",
);
const saveSampleSource = gazePageSource.match(
  /async function saveSample\([\s\S]*?\r?\n}\r?\n\r?\nasync function collect/,
)?.[0] || "";
assert.match(saveSampleSource, /if \(!res\.ok \|\| data\.ok === false\)/);
assert.match(saveSampleSource, /throw err;/);
const calibrationPointLiteral = gazePageSource.match(
  /const calibrationPoints = (\[[\s\S]*?\]);/,
);
assert.ok(calibrationPointLiteral, "calibration point definitions are missing");
const browserCalibrationPoints = JSON.parse(
  calibrationPointLiteral[1].replace(/,\s*]$/, "]"),
);
assert.deepEqual(
  browserCalibrationPoints,
  calibrationContract.frozen_targets.map((target) => [
    target.target_x_viewport_fraction,
    target.target_y_viewport_fraction,
  ]),
);
const motionBlockLiteral = gazePageSource.match(
  /const motionCalibrationBlocks = \[([\s\S]*?)\n\];/,
);
assert.ok(motionBlockLiteral, "motion block definitions are missing");
const browserMotionBlocks = [...motionBlockLiteral[1].matchAll(
  /id: "([^"]+)",[\s\S]*?posture: "([^"]+)",[\s\S]*?distance: "([^"]+)"/g,
)].map((match) => ({
  motion_block_id: match[1],
  posture_condition: match[2],
  distance_condition: match[3],
}));
assert.deepEqual(browserMotionBlocks, calibrationContract.motion_blocks);

const wordTrackSource = fs.readFileSync(
  path.join(__dirname, "../web/templates/word_track.html"),
  "utf8",
);
const captureScriptIndex = wordTrackSource.indexOf('/static/gaze_capture_contract.js');
const integrationScriptIndex = wordTrackSource.indexOf('/static/gaze_integration.js');
assert.ok(captureScriptIndex >= 0, "word-track capture helper script is missing");
assert.ok(
  captureScriptIndex < integrationScriptIndex,
  "capture helper must load before live gaze integration",
);

const gazeIntegrationSource = fs.readFileSync(
  path.join(__dirname, "../web/static/gaze_integration.js"),
  "utf8",
);
assert.match(gazeIntegrationSource, /const gazeCapture = window\.LexiGazeCapture/);
assert.match(gazeIntegrationSource, /getUserMedia\(\s*gazeCapture\.mediaConstraints\(\)/);
assert.match(
  gazeIntegrationSource,
  /gazeCapture\.captureSnapshot\(state\.video, state\.canvas\)/,
);
assert.match(gazeIntegrationSource, /image_data: snapshot\.image_data/);
assert.match(gazeIntegrationSource, /capture_contract: snapshot\.capture_contract/);
assert.doesNotMatch(gazeIntegrationSource, /const width = 240/);

const participantAssessmentTemplate = fs.readFileSync(
  path.join(__dirname, "../web/templates/participant_assessment.html"),
  "utf8",
);
assert.ok(
  participantAssessmentTemplate.indexOf("gaze_capture_contract.js")
    < participantAssessmentTemplate.indexOf("participant_assessment.js"),
  "capture helper must load before participant assessment",
);
const participantAssessmentSource = fs.readFileSync(
  path.join(__dirname, "../web/static/participant_assessment.js"),
  "utf8",
);
assert.match(participantAssessmentSource, /getUserMedia\(\s*gazeCapture\.mediaConstraints\(\)/);
assert.match(
  participantAssessmentSource,
  /gazeCapture\.captureSnapshot\(\s*ui\.cameraPreview,\s*ui\.captureCanvas/,
);
assert.match(participantAssessmentSource, /image_data: snapshot\.image_data/);
assert.match(participantAssessmentSource, /capture_contract: snapshot\.capture_contract/);

console.log("gaze capture contract tests passed");
