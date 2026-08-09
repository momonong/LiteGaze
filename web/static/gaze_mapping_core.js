(function installLexiGazeMappingCore(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.LexiGazeMappingCore = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function createMappingCore() {
  "use strict";

  const DEFAULT_THRESHOLDS = Object.freeze({
    lineY: 90,
    mediumDistance: 35,
    lowDistance: 90,
  });

  function finiteNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function strictNumber(value) {
    if (
      value === null
      || value === undefined
      || typeof value === "boolean"
      || (typeof value === "string" && value.trim() === "")
    ) {
      return null;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function geometryBox(item) {
    if (!item || typeof item !== "object") return null;
    const left = strictNumber(item.left);
    const top = strictNumber(item.top);
    const explicitRight = strictNumber(item.right);
    const explicitBottom = strictNumber(item.bottom);
    const width = strictNumber(item.width);
    const height = strictNumber(item.height);
    const right = explicitRight !== null
      ? explicitRight
      : width !== null && left !== null ? left + width : null;
    const bottom = explicitBottom !== null
      ? explicitBottom
      : height !== null && top !== null ? top + height : null;
    if ([left, right, top, bottom].some((value) => value === null)) return null;
    if (right < left || bottom < top) return null;
    return { left, right, top, bottom };
  }

  function distanceToRect(x, y, item) {
    const box = geometryBox(item);
    if (!box) return Number.POSITIVE_INFINITY;
    const dx = Math.max(box.left - x, 0, x - box.right);
    const dy = Math.max(box.top - y, 0, y - box.bottom);
    return Math.hypot(dx, dy);
  }

  function findNearestGeometryCandidate(localX, localY, items, thresholds = {}) {
    const active = {
      ...DEFAULT_THRESHOLDS,
      ...thresholds,
    };
    let best = null;

    for (const item of items || []) {
      const box = geometryBox(item);
      if (!box) continue;
      const { left, right, top, bottom } = box;
      const insideActualBox = (
        localX >= left
        && localX <= right
        && localY >= top
        && localY <= bottom
      );

      if (insideActualBox) {
        return { item, confidence: "high", distance: 0, mappingMode: "geometry_only_v1" };
      }

      const wordCenterY = top + (bottom - top) / 2;
      if (Math.abs(localY - wordCenterY) > active.lineY) {
        continue;
      }

      const distance = distanceToRect(localX, localY, item);
      const confidence = distance <= active.mediumDistance
        ? "medium"
        : distance <= active.lowDistance
          ? "low"
          : null;
      if (!confidence) {
        continue;
      }

      const candidate = {
        item,
        confidence,
        distance,
        mappingMode: "geometry_only_v1",
      };
      if (!best || candidate.distance < best.distance) {
        best = candidate;
      }
    }

    return best;
  }

  function cognitivePreviewPoint(gazePoint, wordCenter, loadScore, options = {}) {
    const raw = {
      x: finiteNumber(gazePoint && gazePoint.x),
      y: finiteNumber(gazePoint && gazePoint.y),
      applied: false,
    };
    if (options.enabled !== true) {
      return raw;
    }

    const score = Math.min(1, Math.max(0, finiteNumber(loadScore)));
    const cognitiveMass = 1 + 1.8 * score;
    const pullFactor = 1 - (1 / cognitiveMass);
    return {
      x: raw.x + (finiteNumber(wordCenter && wordCenter.x) - raw.x) * pullFactor,
      y: raw.y + (finiteNumber(wordCenter && wordCenter.y) - raw.y) * pullFactor,
      applied: true,
      cognitiveMass,
    };
  }

  return Object.freeze({
    DEFAULT_THRESHOLDS,
    cognitivePreviewPoint,
    distanceToRect,
    findNearestGeometryCandidate,
  });
});
