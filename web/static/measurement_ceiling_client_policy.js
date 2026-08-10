(function installMeasurementCeilingClientPolicy(global) {
  "use strict";

  function blocksNewRun(recoverableContext, credentials) {
    return Boolean(recoverableContext) || Boolean(credentials);
  }

  function isPendingCreate(context) {
    return Boolean(context)
      && context.state === "pending_create"
      && typeof context.create_request_id === "string"
      && context.create_request_id.length > 0
      && typeof context.run_token === "string"
      && context.run_token.length > 0;
  }

  function isInvalidStoredContext(context) {
    return Boolean(context) && context.state === "invalid_stored_context";
  }

  function canReplacePendingPreflight(context) {
    return isPendingCreate(context)
      && context.preflight_replacement_required === true;
  }

  function randomHex(byteLength, fillRandom) {
    if (!Number.isInteger(byteLength) || byteLength < 16) {
      throw new Error("authority byte length is too small");
    }
    if (typeof fillRandom !== "function") {
      throw new Error("cryptographic random source is unavailable");
    }
    const bytes = new Uint8Array(byteLength);
    fillRandom(bytes);
    return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  }

  function newCreateAuthority(fillRandom) {
    return {
      create_request_id: `WGMCREQ-${randomHex(16, fillRandom)}`,
      run_token: `wgmr_client_${randomHex(32, fillRandom)}`,
    };
  }

  function cleanupConfirmed(responseOk, payload) {
    return responseOk === true
      && Boolean(payload)
      && payload.ok === true
      && payload.cleanup_verified === true;
  }

  function isCalibrationUnusableNegative(payload) {
    return Boolean(payload)
      && payload.classification === "calibration_unusable_negative_result";
  }

  function calibrationNegativeDisplay(payload) {
    const source = payload || {};
    return Object.freeze({
      attempts: Number(source.calibration_attempt_count ?? 0),
      usable: Number(
        source.calibration_usable_count
        ?? source.usable_calibration_count
        ?? 0,
      ),
      purge_verified: (
        source.purge_verified === true
        || (
          source.images_purged === true
          && source.cleanup_verified === true
        )
      ),
    });
  }

  function exactFrameRetryRequired(payload) {
    return Boolean(payload)
      && (
        payload.exact_frame_retry_required === true
        || payload.prepared === true
      );
  }

  function serverSpoolRetryAvailable(payload) {
    return Boolean(payload)
      && (
        payload.server_spool_retry_available === true
        || payload.server_spool_available === true
        || payload.prepared_observation_pending === true
        || payload.status === "committed"
      );
  }

  function abortRequired(payload) {
    return Boolean(payload)
      && (
        payload.abort_required === true
        || payload.classification === "abort_required"
      );
  }

  const api = Object.freeze({
    blocksNewRun,
    isPendingCreate,
    isInvalidStoredContext,
    canReplacePendingPreflight,
    newCreateAuthority,
    cleanupConfirmed,
    isCalibrationUnusableNegative,
    calibrationNegativeDisplay,
    exactFrameRetryRequired,
    serverSpoolRetryAvailable,
    abortRequired,
  });
  global.LexiGazeMeasurementClientPolicy = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
