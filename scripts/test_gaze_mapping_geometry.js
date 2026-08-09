"use strict";

const assert = require("node:assert/strict");
const mapping = require("../web/static/gaze_mapping_core.js");

function identity(result) {
  if (!result) return null;
  return {
    index: result.item.index,
    confidence: result.confidence,
    distance: result.distance,
    mappingMode: result.mappingMode,
  };
}

const baseItems = [
  { index: 0, text: "easy", left: 0, right: 40, top: 0, bottom: 20 },
  { index: 1, text: "difficult", left: 100, right: 160, top: 0, bottom: 20 },
];

const baseline = identity(mapping.findNearestGeometryCandidate(82, 10, baseItems));
const uniformPrior = identity(mapping.findNearestGeometryCandidate(
  82,
  10,
  baseItems.map((item) => ({ ...item, load_score: 0.5 })),
));
const reversedPrior = identity(mapping.findNearestGeometryCandidate(
  82,
  10,
  [
    { ...baseItems[0], load_score: 1, cognitiveMass: 999 },
    { ...baseItems[1], load_score: 0, cognitiveMass: 0.001 },
  ],
));

assert.deepEqual(uniformPrior, baseline, "uniform text prior changed geometry mapping");
assert.deepEqual(reversedPrior, baseline, "text-score mutation changed geometry mapping");
assert.deepEqual(baseline, {
  index: 1,
  confidence: "medium",
  distance: 18,
  mappingMode: "geometry_only_v1",
});

const farItems = [
  { index: 7, text: "extraordinary", left: 0, right: 40, top: 0, bottom: 20, load_score: 1 },
];
const sensorAbstention = mapping.findNearestGeometryCandidate(150, 10, farItems);
assert.equal(sensorAbstention, null, "text prior rescued a geometry abstention");

assert.equal(
  mapping.findNearestGeometryCandidate(0, 0, [
    { index: 8, left: "not-a-number", right: 30, top: 0, bottom: 20 },
  ]),
  null,
  "malformed layout geometry must fail closed",
);
assert.equal(
  mapping.findNearestGeometryCandidate(0, 0, [
    { index: 9, left: null, right: 30, top: 0, bottom: 20 },
  ]),
  null,
  "null layout geometry must not be coerced into a sensor hit",
);
assert.equal(
  mapping.distanceToRect(0, 0, { left: 10, right: 0, top: 0, bottom: 20 }),
  Number.POSITIVE_INFINITY,
  "inverted layout geometry must not create a sensor hit",
);

const rawPoint = { x: 150, y: 10 };
const center = { x: 20, y: 10 };
assert.deepEqual(
  mapping.cognitivePreviewPoint(rawPoint, center, 1),
  { x: 150, y: 10, applied: false },
  "cognitive preview must be disabled by default",
);
const preview = mapping.cognitivePreviewPoint(rawPoint, center, 1, { enabled: true });
assert.equal(preview.applied, true, "explicit downstream preview did not activate");
assert.equal(
  mapping.findNearestGeometryCandidate(150, 10, farItems),
  null,
  "downstream preview mutated sensor abstention",
);

console.log("gaze mapping geometry separation: ok");
