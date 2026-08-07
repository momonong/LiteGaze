const STORAGE_KEY = "lexigaze.participantStudy.v1";

const state = {
  protocol: null,
  context: null,
  withdrawalCode: null,
  receiptText: null,
};

const $ = (id) => document.getElementById(id);

function escHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showAlert(message, type = "danger") {
  const node = $("globalAlert");
  node.textContent = message;
  node.className = `alert ${type}`;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function clearAlert() {
  $("globalAlert").className = "alert hidden";
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (state.context?.access_token) {
    headers.Authorization = `Bearer ${state.context.access_token}`;
  }
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function saveContext(context) {
  state.context = context;
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(context));
}

function clearContext() {
  state.context = null;
  state.withdrawalCode = null;
  state.receiptText = null;
  sessionStorage.removeItem(STORAGE_KEY);
}

function restoreContext() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    if (parsed?.study_session_id && parsed?.access_token) state.context = parsed;
  } catch (_) {
    sessionStorage.removeItem(STORAGE_KEY);
  }
}

function renderProtocol(protocol) {
  state.protocol = protocol;
  const activation = protocol.activation;
  const pilotReady = activation.pilot_ready === true;
  $("studyTitle").textContent = protocol.title_zh;
  $("studyPurpose").textContent = protocol.purpose_zh;
  $("protocolVersion").textContent = `Protocol ${protocol.protocol_version}`;
  $("footerProtocol").textContent = protocol.protocol_version;
  $("estimatedTime").textContent = `約 ${protocol.estimated_duration_minutes} 分鐘`;
  $("consentVersion").textContent = `Consent ${protocol.consent_version}`;
  $("modeBadge").textContent = pilotReady ? "正式 pilot 已通過啟動閘門" : "DRY RUN · 正式收案鎖定";
  $("modeBadge").className = `mode-badge ${pilotReady ? "" : "locked"}`;
  $("dryRunNotice").classList.toggle("hidden", pilotReady);
  $("enrollBtn").textContent = pilotReady ? "同意並開始正式流程" : "開始技術 dry run";
  $("inviteField").classList.toggle("hidden", !pilotReady);

  $("proceduresList").innerHTML = protocol.procedures_zh
    .map((item) => `<li>${escHtml(item)}</li>`).join("");
  $("risksList").innerHTML = [...protocol.risks_zh, ...protocol.risk_controls_zh]
    .map((item) => `<li>${escHtml(item)}</li>`).join("");
  $("dataCategories").innerHTML = protocol.data_categories.map((item) => `
    <div class="data-item ${item.required ? "" : "optional"}">
      <strong>${item.required ? "必要" : "獨立選擇"}</strong>${escHtml(item.description_zh)}
    </div>`).join("");

  const governance = protocol.data_governance;
  $("governanceSummary").textContent = pilotReady
    ? `資料地點：${governance.location}；研究資料保存 ${governance.retention_days} 天；校正影格最長 ${governance.raw_frame_retention_hours} 小時。`
    : "正式資料地點、保存期限與加密確認尚未填妥，因此系統不允許真人收案。";
  const contacts = protocol.research_contacts;
  $("contactsSummary").textContent = pilotReady
    ? `研究者：${contacts.investigator}（${contacts.investigator_email}）；受試者權益：${contacts.participant_rights}`
    : "正式研究者與受試者權益聯絡資訊尚未填妥；目前只能進行不收資料的技術演練。";
  $("withdrawalPolicy").textContent = protocol.withdrawal_policy_zh;

  $("requiredStatements").innerHTML = protocol.required_consent_statements.map((item) => `
    <label class="check-row">
      <input type="checkbox" name="consent_statement" value="${escHtml(item.id)}" />
      <span>${escHtml(item.text_zh)}</span>
    </label>`).join("");
  $("comprehensionChecks").innerHTML = protocol.comprehension_checks.map((item) => `
    <div class="question">
      <p>${escHtml(item.question_zh)}</p>
      ${Object.entries(item.options).map(([value, label]) => `
        <label><input type="radio" name="check_${escHtml(item.id)}" value="${escHtml(value)}" /> ${escHtml(label)}</label>`).join("")}
    </div>`).join("");
  $("optionalScopes").innerHTML = protocol.optional_scopes.map((item) => `
    <label class="check-row">
      <input type="checkbox" name="optional_scope" value="${escHtml(item.id)}" />
      <span>${escHtml(item.text_zh)}</span>
    </label>`).join("");
  $("optionalFieldset").classList.toggle("hidden", protocol.optional_scopes.length === 0);
}

function enrollmentPayload() {
  const statements = {};
  document.querySelectorAll('input[name="consent_statement"]').forEach((input) => {
    statements[input.value] = input.checked;
  });
  const checks = {};
  state.protocol.comprehension_checks.forEach((item) => {
    checks[item.id] = document.querySelector(`input[name="check_${item.id}"]:checked`)?.value || "";
  });
  const scopes = {};
  document.querySelectorAll('input[name="optional_scope"]').forEach((input) => {
    scopes[input.value] = input.checked;
  });
  return {
    mode: state.protocol.activation.pilot_ready ? "pilot" : "dry_run",
    invite_code: $("inviteCode").value.trim(),
    adult_confirmed: $("adultConfirmed").checked,
    private_space_confirmed: $("privateSpaceConfirmed").checked,
    consent_statements: statements,
    comprehension_answers: checks,
    optional_scopes: scopes,
  };
}

function sessionStep(stateName) {
  const order = ["consented", "system_check_passed", "calibration_in_progress", "calibration_complete", "assessment_in_progress", "completed"];
  return Math.max(0, order.indexOf(stateName));
}

function renderSession(session) {
  $("consentPanel").classList.add("hidden");
  $("workflowPanel").classList.remove("hidden");
  $("participantId").textContent = session.participant_id;
  $("sessionState").textContent = session.state;
  $("withdrawSessionId").value = session.study_session_id;
  if (state.withdrawalCode) {
    $("withdrawalCode").textContent = state.withdrawalCode;
    $("withdrawCodeInput").value = state.withdrawalCode;
  } else {
    $("receiptBox").classList.add("hidden");
  }

  const step = sessionStep(session.state);
  document.querySelectorAll(".step").forEach((node, index) => {
    node.classList.toggle("done", index < step + 1);
    node.classList.toggle("active", index === Math.min(step + 1, 4));
  });
  $("calibrationBtn").disabled = session.state !== "system_check_passed";
  $("assessmentBtn").disabled = session.state !== "calibration_complete";
  $("completeBtn").disabled = session.state !== "assessment_in_progress";
  $("completionBox").classList.toggle("hidden", session.state !== "completed");

  const dryRun = session.mode === "dry_run";
  $("calibrationBtn").textContent = dryRun ? "模擬校正（不開相機）" : "前往眼動校正";
  $("assessmentBtn").textContent = dryRun ? "模擬閱讀評量（不作答）" : "前往閱讀評量";
  $("completeBtn").textContent = dryRun ? "完成 dry run" : "等待評量完成";
}

function participantReceiptText(result) {
  const receipt = result.consent_receipt;
  return [
    "LexiGaze 受試者同意憑證",
    `Protocol: ${receipt.protocol_id} / ${receipt.protocol_version}`,
    `Consent: ${receipt.consent_version}`,
    `Consent SHA-256: ${receipt.consent_digest_sha256}`,
    `Study session: ${receipt.study_session_id}`,
    `Participant: ${receipt.participant_id}`,
    `Accepted at: ${receipt.accepted_at_utc}`,
    `Mode: ${receipt.mode}`,
    `Withdrawal code: ${result.withdrawal_code}`,
    "",
    "請保存此檔案。正式研究中可使用 study session ID 與撤回碼提出撤回。",
  ].join("\n");
}

function downloadText(text, filename) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function enroll(event) {
  event.preventDefault();
  clearAlert();
  $("enrollBtn").disabled = true;
  try {
    const result = await api("/api/study/enroll", {
      method: "POST",
      body: JSON.stringify(enrollmentPayload()),
    });
    state.withdrawalCode = result.withdrawal_code;
    saveContext({
      study_session_id: result.study_session_id,
      participant_id: result.participant_id,
      access_token: result.access_token,
      mode: result.mode,
      optional_scopes: result.consent_receipt.optional_scopes,
    });
    state.receiptText = participantReceiptText(result);
    renderSession({ ...result, study_session_id: result.study_session_id });
  } catch (error) {
    showAlert(error.message);
  } finally {
    $("enrollBtn").disabled = false;
  }
}

async function runSystemCheck() {
  const secureContext = window.isSecureContext || ["localhost", "127.0.0.1"].includes(location.hostname);
  const results = {
    secure_context: secureContext,
    camera_api: Boolean(navigator.mediaDevices?.getUserMedia),
    screen_size: window.innerWidth >= 760 && window.innerHeight >= 540,
    network: navigator.onLine,
  };
  $("systemCheckResults").innerHTML = Object.entries(results).map(([name, passed]) =>
    `<li class="${passed ? "pass" : "fail"}">${passed ? "✓" : "✕"} ${escHtml(name)}</li>`
  ).join("");
  if (Object.values(results).some((passed) => !passed)) {
    showAlert("系統檢查未通過，請處理紅色項目後再試一次。", "danger");
    return;
  }
  try {
    const result = await api(`/api/study/sessions/${state.context.study_session_id}/system-check`, {
      method: "POST",
      body: JSON.stringify({ checks: results }),
    });
    renderSession(result.session);
  } catch (error) {
    showAlert(error.message);
  }
}

async function dryRunAction(action) {
  const result = await api(`/api/study/sessions/${state.context.study_session_id}/dry-run`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });
  renderSession(result.session);
  return result.session;
}

async function calibrationAction() {
  if (state.context.mode === "dry_run") {
    await dryRunAction("calibration_start");
    await dryRunAction("calibration_complete");
    return;
  }
  location.assign("/gaze?study=1");
}

async function assessmentAction() {
  if (state.context.mode === "dry_run") {
    await dryRunAction("assessment_start");
    return;
  }
  location.assign("/study/assessment?study=1");
}

async function completeAction() {
  if (state.context.mode === "dry_run") await dryRunAction("assessment_complete");
}

async function withdraw() {
  clearAlert();
  const sessionId = $("withdrawSessionId").value.trim();
  const code = $("withdrawCodeInput").value.trim();
  if (!sessionId || (!code && state.context?.study_session_id !== sessionId)) {
    showAlert("請提供 study session ID 與撤回碼。", "danger");
    return;
  }
  if (!window.confirm("確定要停止流程並刪除仍可定位的資料嗎？此動作無法復原。")) return;
  try {
    const payload = { study_session_id: sessionId, withdrawal_code: code };
    if (state.context?.study_session_id === sessionId) payload.access_token = state.context.access_token;
    const result = await api("/api/study/withdraw", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    $("withdrawResult").textContent = JSON.stringify({
      withdrawal_receipt_id: result.withdrawal_receipt_id,
      withdrawn_at_utc: result.withdrawn_at_utc,
      deleted_scopes: result.deleted_scopes,
    }, null, 2);
    $("withdrawResult").classList.remove("hidden");
    if (state.context?.study_session_id === sessionId) {
      clearContext();
      $("workflowPanel").classList.add("hidden");
      $("consentPanel").classList.remove("hidden");
    }
  } catch (error) {
    showAlert(error.message);
  }
}

async function restoreSession() {
  if (!state.context) return;
  try {
    const result = await api(`/api/study/sessions/${state.context.study_session_id}`);
    renderSession(result.session);
  } catch (_) {
    clearContext();
  }
}

async function init() {
  restoreContext();
  try {
    const result = await api("/api/study/protocol");
    renderProtocol(result.protocol);
    await restoreSession();
  } catch (error) {
    showAlert(`研究流程無法啟動：${error.message}`);
  }
}

$("consentForm").addEventListener("submit", enroll);
$("declineBtn").addEventListener("click", () => {
  clearContext();
  showAlert("你已選擇不參與。系統沒有建立研究 session 或保存研究資料。", "success");
});
$("runSystemCheckBtn").addEventListener("click", runSystemCheck);
$("calibrationBtn").addEventListener("click", () => calibrationAction().catch((error) => showAlert(error.message)));
$("assessmentBtn").addEventListener("click", () => assessmentAction().catch((error) => showAlert(error.message)));
$("completeBtn").addEventListener("click", () => completeAction().catch((error) => showAlert(error.message)));
$("downloadReceiptBtn").addEventListener("click", () => {
  if (state.receiptText) {
    downloadText(state.receiptText, `${state.context.participant_id}_consent_receipt.txt`);
  }
});
$("withdrawBtn").addEventListener("click", withdraw);

init();
