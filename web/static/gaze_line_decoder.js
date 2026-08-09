(function installLexiGazeLineDecoder(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.LexiGazeLineDecoder = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function createLineDecoder() {
  "use strict";

  const DECODER_ID = "geometry-layout-line-first-shadow-v1";
  const DEFAULT_OPTIONS = Object.freeze({
    line_sigma_y_px: 45,
    line_sigma_x_px: 240,
    token_sigma_px: 55,
    maximum_line_distance_px: 90,
    maximum_token_distance_px: 90,
    top_k: 3,
  });

  function finiteNumber(value) {
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

  function positiveOption(value, fallback) {
    const parsed = finiteNumber(value);
    return parsed !== null && parsed > 0 ? parsed : fallback;
  }

  function nonnegativeOption(value, fallback) {
    const parsed = finiteNumber(value);
    return parsed !== null && parsed >= 0 ? parsed : fallback;
  }

  function distanceToInterval(value, lower, upper) {
    return Math.max(lower - value, 0, value - upper);
  }

  function distanceToBox(x, y, bbox) {
    return Math.hypot(
      distanceToInterval(x, bbox.left, bbox.right),
      distanceToInterval(y, bbox.top, bbox.bottom),
    );
  }

  function normalizeBox(value) {
    if (!value || typeof value !== "object") return null;
    const left = finiteNumber(value.left);
    const right = finiteNumber(value.right);
    const top = finiteNumber(value.top);
    const bottom = finiteNumber(value.bottom);
    if ([left, right, top, bottom].some((entry) => entry === null)) return null;
    if (right < left || bottom < top) return null;
    return { left, right, top, bottom };
  }

  function normalizeTokens(tokens) {
    if (!Array.isArray(tokens) || tokens.length === 0) {
      return { tokens: [], reason: "missing_layout_tokens" };
    }

    const normalized = [];
    const occurrenceIds = new Set();
    for (const token of tokens) {
      if (!token || typeof token !== "object") {
        return { tokens: [], reason: "invalid_layout_token" };
      }
      const occurrenceId = String(token.occurrence_id ?? "").trim();
      const rawLineId = token.line_id;
      const lineId = rawLineId === null || rawLineId === undefined
        ? ""
        : String(rawLineId).trim();
      const readingOrder = finiteNumber(token.reading_order);
      const bbox = normalizeBox(token.bbox);
      if (!occurrenceId || !lineId || readingOrder === null || !bbox) {
        return { tokens: [], reason: "invalid_layout_token" };
      }
      if (occurrenceIds.has(occurrenceId)) {
        return { tokens: [], reason: "duplicate_occurrence_id" };
      }
      occurrenceIds.add(occurrenceId);
      normalized.push({
        occurrence_id: occurrenceId,
        line_id: lineId,
        reading_order: readingOrder,
        bbox,
      });
    }
    return { tokens: normalized, reason: null };
  }

  function normalizedWeights(records) {
    if (records.length === 0) return [];
    const maximum = Math.max(...records.map((record) => record.log_score));
    const weights = records.map((record) => Math.exp(record.log_score - maximum));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return records.map((record, index) => ({
      ...record,
      posterior: total > 0 ? weights[index] / total : 0,
    }));
  }

  function baseResult(sensorAccepted, x, y) {
    return {
      schema_version: 1,
      decoder_id: DECODER_ID,
      shadow_only: true,
      geometry_only: true,
      sensor_accepted: sensorAccepted,
      abstain: true,
      abstain_reason: null,
      selected_line_id: null,
      line_scores: [],
      top_k_occurrences: [],
      geometry_only_result: {
        sensor_x_px: x,
        sensor_y_px: y,
        best_line_id: null,
        best_occurrence_id: null,
        line_posterior: null,
        occurrence_posterior: null,
      },
    };
  }

  function abstention(reason, sensorAccepted, x, y, details = {}) {
    return {
      ...baseResult(sensorAccepted, x, y),
      ...details,
      abstain_reason: reason,
    };
  }

  function buildLines(tokens, x, y, options) {
    const groups = new Map();
    for (const token of tokens) {
      if (!groups.has(token.line_id)) groups.set(token.line_id, []);
      groups.get(token.line_id).push(token);
    }

    const records = [];
    for (const [lineId, lineTokens] of groups.entries()) {
      const left = Math.min(...lineTokens.map((token) => token.bbox.left));
      const right = Math.max(...lineTokens.map((token) => token.bbox.right));
      const top = Math.min(...lineTokens.map((token) => token.bbox.top));
      const bottom = Math.max(...lineTokens.map((token) => token.bbox.bottom));
      const verticalDistance = distanceToInterval(y, top, bottom);
      const horizontalDistance = distanceToInterval(x, left, right);
      const verticalScaled = verticalDistance / options.line_sigma_y_px;
      const horizontalScaled = horizontalDistance / options.line_sigma_x_px;
      records.push({
        line_id: lineId,
        vertical_distance_px: verticalDistance,
        horizontal_distance_px: horizontalDistance,
        geometry_cost: Math.hypot(verticalScaled, horizontalScaled),
        log_score: -0.5 * (
          verticalScaled * verticalScaled
          + horizontalScaled * horizontalScaled
        ),
        tokens: lineTokens,
      });
    }

    return normalizedWeights(records).sort((left, right) => (
      right.posterior - left.posterior
      || left.geometry_cost - right.geometry_cost
      || left.line_id.localeCompare(right.line_id)
    ));
  }

  function publicLineScores(lines) {
    return lines.map((line) => ({
      line_id: line.line_id,
      posterior: line.posterior,
      geometry_cost: line.geometry_cost,
      vertical_distance_px: line.vertical_distance_px,
      horizontal_distance_px: line.horizontal_distance_px,
    }));
  }

  function buildTokenCandidates(line, x, y, options) {
    const records = line.tokens.map((token) => {
      const distance = distanceToBox(x, y, token.bbox);
      const scaled = distance / options.token_sigma_px;
      return {
        occurrence_id: token.occurrence_id,
        line_id: token.line_id,
        reading_order: token.reading_order,
        distance_px: distance,
        log_score: -0.5 * scaled * scaled,
      };
    });
    return normalizedWeights(records).sort((left, right) => (
      right.posterior - left.posterior
      || left.distance_px - right.distance_px
      || left.reading_order - right.reading_order
      || left.occurrence_id.localeCompare(right.occurrence_id)
    ));
  }

  function decodeLineFirst(sensorObservation, tokens, rawOptions = {}) {
    const accepted = sensorObservation && sensorObservation.sensor_accepted === true;
    const x = finiteNumber(sensorObservation && sensorObservation.x_px);
    const y = finiteNumber(sensorObservation && sensorObservation.y_px);
    if (!sensorObservation || typeof sensorObservation !== "object") {
      return abstention("missing_sensor_observation", false, null, null);
    }
    if (!accepted) {
      return abstention("sensor_abstained", false, x, y);
    }
    if (x === null || y === null) {
      return abstention("invalid_sensor_coordinates", true, x, y);
    }

    const normalized = normalizeTokens(tokens);
    if (normalized.reason) {
      return abstention(normalized.reason, true, x, y);
    }

    const options = {
      line_sigma_y_px: positiveOption(
        rawOptions.line_sigma_y_px,
        DEFAULT_OPTIONS.line_sigma_y_px,
      ),
      line_sigma_x_px: positiveOption(
        rawOptions.line_sigma_x_px,
        DEFAULT_OPTIONS.line_sigma_x_px,
      ),
      token_sigma_px: positiveOption(
        rawOptions.token_sigma_px,
        DEFAULT_OPTIONS.token_sigma_px,
      ),
      maximum_line_distance_px: nonnegativeOption(
        rawOptions.maximum_line_distance_px,
        DEFAULT_OPTIONS.maximum_line_distance_px,
      ),
      maximum_token_distance_px: nonnegativeOption(
        rawOptions.maximum_token_distance_px,
        DEFAULT_OPTIONS.maximum_token_distance_px,
      ),
      top_k: Math.max(
        1,
        Math.floor(positiveOption(rawOptions.top_k, DEFAULT_OPTIONS.top_k)),
      ),
    };

    const lines = buildLines(normalized.tokens, x, y, options);
    const lineScores = publicLineScores(lines);
    const selectedLine = lines[0];
    if (selectedLine.vertical_distance_px > options.maximum_line_distance_px) {
      return abstention("outside_line_geometry", true, x, y, {
        line_scores: lineScores,
      });
    }

    const candidates = buildTokenCandidates(selectedLine, x, y, options);
    const bestCandidate = candidates[0];
    if (bestCandidate.distance_px > options.maximum_token_distance_px) {
      return abstention("outside_token_geometry", true, x, y, {
        selected_line_id: selectedLine.line_id,
        line_scores: lineScores,
      });
    }

    const topCandidates = candidates.slice(0, options.top_k).map((candidate, index) => ({
      occurrence_id: candidate.occurrence_id,
      line_id: candidate.line_id,
      reading_order: candidate.reading_order,
      rank: index + 1,
      posterior: candidate.posterior,
      distance_px: candidate.distance_px,
    }));
    return {
      ...baseResult(true, x, y),
      abstain: false,
      selected_line_id: selectedLine.line_id,
      line_scores: lineScores,
      top_k_occurrences: topCandidates,
      geometry_only_result: {
        sensor_x_px: x,
        sensor_y_px: y,
        best_line_id: selectedLine.line_id,
        best_occurrence_id: bestCandidate.occurrence_id,
        line_posterior: selectedLine.posterior,
        occurrence_posterior: bestCandidate.posterior,
      },
    };
  }

  return Object.freeze({
    DECODER_ID,
    DEFAULT_OPTIONS,
    decodeLineFirst,
  });
});
