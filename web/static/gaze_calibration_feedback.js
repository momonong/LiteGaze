(function installLexiGazeCalibrationFeedback(global) {
  "use strict";

  const REASON_GUIDANCE = Object.freeze({
    fewer_than_65_calibration_samples:
      "有效校正影格不足；請確認完整臉部持續在預覽中央且光線均勻。",
    fewer_than_13_unique_targets:
      "校正沒有完成全部 13 個目標；重試時請不要中途停止或切換分頁。",
    motion_diverse_protocol_missing:
      "校正流程版本不符；請停止並聯絡研究者。",
    motion_coverage_gate_failed:
      "姿勢或距離覆蓋不足；請依序完成 neutral、left、right、near、far 五個區塊。",
    malformed_manifest_rows:
      "校正紀錄不完整；請停止並聯絡研究者。",
    personalization_training_error:
      "CPU 個人化沒有完成；可重試一次，若再次出現請停止並聯絡研究者。",
    personalization_training_failed:
      "CPU 個人化沒有完成；可重試一次，若再次出現請停止並聯絡研究者。",
    personalization_model_binding_mismatch:
      "個人化模型與本次校正不相符；請停止並聯絡研究者。",
    calibration_image_purge_error:
      "校正影像清除未能驗證；請停止流程並立即聯絡研究者。",
    calibration_audit_error:
      "伺服器無法完成校正稽核；請停止並聯絡研究者。",
  });

  const MOTION_GUIDANCE = Object.freeze({
    INSUFFICIENT_USABLE_SAMPLES:
      "可用臉部／頭部姿勢影格不足；請讓額頭、雙眼、鼻子與下巴都留在畫面內。",
    INSUFFICIENT_MOTION_BLOCKS:
      "缺少動作區塊；請完成畫面要求的五種姿勢與距離。",
    NO_CROSS_CONDITION_TARGET_REPLICATION:
      "同一批目標沒有跨姿勢完整重複；請不要略過任何區塊。",
    NARROW_HEAD_YAW_COVERAGE:
      "左右轉頭幅度不足；left／right 區塊請各轉約 15 度，眼睛仍看校正點。",
    MISSING_DISTANCE_CONDITIONS:
      "缺少近／中／遠距離區塊；near／far 請移動約 15–20 公分。",
    MISSING_POSTURE_CONDITIONS:
      "缺少左／中／右姿勢區塊；請依指示完成每一區塊。",
    UNVERIFIED_LATERAL_POSE_SEPARATION:
      "左右姿勢無法辨識；請保持臉完整入鏡並增加左右轉頭差異。",
    INSUFFICIENT_LATERAL_POSE_SEPARATION:
      "左右轉頭差異不足；left／right 區塊請各轉約 15 度。",
    UNVERIFIED_DISTANCE_SEPARATION:
      "近／遠距離無法辨識；請保持臉完整入鏡並依指示移動。",
    INSUFFICIENT_DISTANCE_SEPARATION:
      "近／遠距離變化不足；near／far 請各移動約 15–20 公分。",
    MALFORMED_MANIFEST_LINES:
      "校正紀錄損壞；請停止並聯絡研究者。",
  });

  function uniqueStrings(values) {
    return [...new Set(values.filter((value) => typeof value === "string" && value.trim()).map((value) => value.trim()))];
  }

  function qualityCodes(payload) {
    const quality = payload && typeof payload === "object" ? payload.quality : null;
    if (!quality || typeof quality !== "object") return [];
    const reasons = Array.isArray(quality.reasons) ? quality.reasons : [];
    const motionIssues = Array.isArray(quality.motion_audit_issues)
      ? quality.motion_audit_issues.map((issue) => issue && typeof issue === "object" ? issue.code : "")
      : [];
    return uniqueStrings([...reasons, ...motionIssues]);
  }

  function buildFailureMessage(payload) {
    const codes = qualityCodes(payload);
    const guidance = uniqueStrings(codes.map((code) => {
      if (REASON_GUIDANCE[code]) return REASON_GUIDANCE[code];
      if (MOTION_GUIDANCE[code]) return MOTION_GUIDANCE[code];
      if (code.startsWith("INCOMPLETE_")) {
        return "校正 metadata 不完整；請停止、不要繼續閱讀，並聯絡研究者檢查流程版本。";
      }
      return "";
    }));
    if (!guidance.length) {
      guidance.push("請確認臉完整入鏡、光線均勻，並完整完成五個姿勢與全部目標。");
    }
    const codeSummary = codes.length ? ` 原因代碼：${codes.join("、")}。` : "";
    return `校正未通過，暫存校正影像已清除。${guidance.join(" ")}${codeSummary} 調整後可在本頁重新開始校正；若訊息要求聯絡研究者，請不要繼續閱讀。`;
  }

  function noFacePrompt(pointIndex) {
    const point = Number.isInteger(pointIndex) ? `第 ${pointIndex + 1} 個目標` : "目前目標";
    return `${point}偵測不到完整臉部。請確認臉位於相機預覽中央、雙眼與下巴都入鏡、沒有遮擋或強烈背光；調整後按確定繼續。若持續發生，本次校正會被品質閘拒絕並可重新校正。`;
  }

  const api = Object.freeze({ buildFailureMessage, noFacePrompt, qualityCodes });
  global.LexiGazeCalibrationFeedback = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
