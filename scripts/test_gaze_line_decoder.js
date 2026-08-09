"use strict";

const assert = require("node:assert/strict");
const decoder = require("../web/static/gaze_line_decoder.js");

const tokens = [
  {
    occurrence_id: "page:1:word:0",
    bbox: { left: 0, right: 40, top: 0, bottom: 20 },
    line_id: "line-1",
    reading_order: 0,
    text: "repeat",
  },
  {
    occurrence_id: "page:1:word:1",
    bbox: { left: 50, right: 90, top: 0, bottom: 20 },
    line_id: "line-1",
    reading_order: 1,
    text: "first",
  },
  {
    occurrence_id: "page:1:word:2",
    bbox: { left: 0, right: 40, top: 60, bottom: 80 },
    line_id: "line-2",
    reading_order: 2,
    text: "repeat",
  },
  {
    occurrence_id: "page:1:word:3",
    bbox: { left: 50, right: 90, top: 60, bottom: 80 },
    line_id: "line-2",
    reading_order: 3,
    text: "middle",
  },
  {
    occurrence_id: "page:1:word:4",
    bbox: { left: 100, right: 140, top: 60, bottom: 80 },
    line_id: "line-2",
    reading_order: 4,
    text: "repeat",
  },
];

const observation = { sensor_accepted: true, x_px: 72, y_px: 70 };
const baseline = decoder.decodeLineFirst(observation, tokens, { top_k: 3 });
assert.equal(baseline.shadow_only, true);
assert.equal(baseline.geometry_only, true);
assert.equal(baseline.abstain, false);
assert.equal(baseline.selected_line_id, "line-2");
assert.equal(baseline.line_scores[0].line_id, "line-2");
assert.ok(baseline.line_scores[0].posterior > baseline.line_scores[1].posterior);
assert.ok(Math.abs(
  baseline.line_scores.reduce((sum, line) => sum + line.posterior, 0) - 1,
) < 1e-12);

assert.deepEqual(
  baseline.top_k_occurrences.map((candidate) => candidate.occurrence_id),
  ["page:1:word:3", "page:1:word:4", "page:1:word:2"],
);
assert.equal(baseline.geometry_only_result.best_occurrence_id, "page:1:word:3");
assert.equal(baseline.top_k_occurrences.length, 3);
assert.equal(
  new Set(baseline.top_k_occurrences.map((candidate) => candidate.occurrence_id)).size,
  3,
  "repeated spellings were collapsed instead of preserving occurrence identity",
);
const topTwo = decoder.decodeLineFirst(observation, tokens, { top_k: 2 });
assert.deepEqual(
  topTwo.top_k_occurrences.map((candidate) => candidate.occurrence_id),
  ["page:1:word:3", "page:1:word:4"],
  "top_k did not truncate the geometry-ranked candidates",
);

const mutatedTokens = tokens.map((token, index) => ({
  ...token,
  text: index % 2 ? "entirely-different" : "same-spelling",
  load_score: index / tokens.length,
  cognitive_profile: { weight: 1000 - index },
}));
const uniformTokens = tokens.map((token) => ({
  ...token,
  text: "uniform",
  load_score: 0.5,
  cognitive_profile: { weight: 0.5 },
}));
assert.deepEqual(
  decoder.decodeLineFirst(observation, uniformTokens, { top_k: 3 }),
  baseline,
  "uniform non-geometry fields changed shadow decoding",
);
const mutated = decoder.decodeLineFirst(observation, mutatedTokens, {
  top_k: 3,
  prior: { rescue: true },
});
assert.deepEqual(mutated, baseline, "non-geometry token fields changed shadow decoding");

const rejected = decoder.decodeLineFirst(
  { sensor_accepted: false, x_px: 72, y_px: 70 },
  mutatedTokens,
  { top_k: 5, prior: { rescue: true } },
);
assert.equal(rejected.abstain, true);
assert.equal(rejected.abstain_reason, "sensor_abstained");
assert.deepEqual(rejected.line_scores, []);
assert.deepEqual(rejected.top_k_occurrences, []);

const outsideLine = decoder.decodeLineFirst(
  { sensor_accepted: true, x_px: 72, y_px: 300 },
  tokens,
);
assert.equal(outsideLine.abstain_reason, "outside_line_geometry");
assert.ok(outsideLine.line_scores.length > 0);

const outsideToken = decoder.decodeLineFirst(
  { sensor_accepted: true, x_px: 500, y_px: 70 },
  tokens,
);
assert.equal(outsideToken.abstain_reason, "outside_token_geometry");
assert.equal(outsideToken.selected_line_id, "line-2");
assert.deepEqual(outsideToken.top_k_occurrences, []);

const duplicateOccurrence = decoder.decodeLineFirst(observation, [tokens[0], {
  ...tokens[1],
  occurrence_id: tokens[0].occurrence_id,
}]);
assert.equal(duplicateOccurrence.abstain_reason, "duplicate_occurrence_id");

const nullCoordinate = decoder.decodeLineFirst(
  { sensor_accepted: true, x_px: null, y_px: 70 },
  tokens,
);
assert.equal(nullCoordinate.abstain_reason, "invalid_sensor_coordinates");

const nullReadingOrder = decoder.decodeLineFirst(observation, [{
  ...tokens[0],
  reading_order: null,
}]);
assert.equal(nullReadingOrder.abstain_reason, "invalid_layout_token");

console.log("gaze line-first shadow decoder: ok");
