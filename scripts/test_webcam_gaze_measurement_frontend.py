"""Offline source contracts for the dedicated measurement browser surface."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "templates" / "measurement_ceiling.html"
PAGE = ROOT / "web" / "static" / "measurement_ceiling.js"
GATE = ROOT / "web" / "static" / "measurement_ceiling_gate.js"
POLICY = ROOT / "web" / "static" / "measurement_ceiling_client_policy.js"
CSS = ROOT / "web" / "static" / "measurement_ceiling.css"
NODE_TEST = ROOT / "scripts" / "test_webcam_gaze_measurement_frontend.js"


class MeasurementCeilingFrontendContractTests(unittest.TestCase):
    def test_template_uses_shared_capture_and_pure_gate_before_page(self) -> None:
        source = TEMPLATE.read_text(encoding="utf-8")
        capture = source.index("/static/gaze_capture_contract.js")
        gate = source.index("/static/measurement_ceiling_gate.js")
        policy = source.index("/static/measurement_ceiling_client_policy.js")
        page = source.index("/static/measurement_ceiling.js")
        self.assertLess(capture, gate)
        self.assertLess(gate, policy)
        self.assertLess(policy, page)
        self.assertIn("0 / 193", source)
        self.assertIn("不衡量認知能力", source)
        self.assertIn("不是視覺 attestation", source)
        self.assertNotIn("<script>", source)
        self.assertNotIn("<style>", source)
        self.assertTrue(CSS.is_file())
        self.assertIn("dedicated session 暫存 raw/crop/normalized", source)
        self.assertIn("crash-recovery spool 也可能短暫存在", source)
        self.assertIn("Ledger 與 final artifact 不含影像", source)
        page_source = PAGE.read_text(encoding="utf-8")
        css_source = CSS.read_text(encoding="utf-8")
        self.assertNotIn(".style.", page_source)
        self.assertIn("function applyTargetPosition(row)", page_source)
        self.assertIn(".target-x-008 { left: 8%; }", css_source)
        self.assertIn(".target-y-090 { top: 90%; }", css_source)

    def test_tokens_stay_in_headers_and_session_storage(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        self.assertIn('const RUN_TOKEN_HEADER = "X-Lexigaze-Measurement-Run-Token"', source)
        self.assertIn(
            'const CHALLENGE_TOKEN_HEADER = "X-Lexigaze-Measurement-Challenge-Token"',
            source,
        )
        self.assertIn("sessionStorage.setItem", source)
        self.assertNotIn("localStorage", source)
        self.assertNotIn("URLSearchParams", source)
        self.assertNotIn("location.search", source)
        self.assertNotIn("console.", source)
        self.assertNotIn("payload.run_token", source)
        self.assertNotIn("payload.challenge_token", source)
        self.assertNotIn("body.run_token", source)
        self.assertNotIn("body.challenge_token", source)

    def test_readiness_preflight_is_target_free_and_required_before_run(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        start = source.index("async function runReadinessPreflight()")
        end = source.index("function updateProgress", start)
        preflight = source[start:end]
        self.assertIn("attempt <= 3", preflight)
        self.assertIn("consecutive_successes", preflight)
        self.assertIn("snapshot.image_data", preflight)
        self.assertIn("snapshot.capture_contract", preflight)
        self.assertNotIn("target_", preflight)
        self.assertNotIn("model_name", preflight)
        self.assertNotIn("cursor", preflight)
        self.assertNotIn("cognitive", preflight)
        create = source[source.index("async function createRun()") :]
        self.assertIn("!state.preflightReady || !state.preflightToken", create)
        self.assertIn(
            "headers[PREFLIGHT_TOKEN_HEADER] = state.pendingCreate.preflightToken",
            create,
        )

    def test_capture_boundary_omits_labels_and_non_sensor_priors(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        capture_start = source.index("state.pendingCapture = {")
        capture_end = source.index("};", capture_start)
        capture_payload = source[capture_start:capture_end]
        self.assertIn("image_data:", capture_payload)
        self.assertIn("capture_contract:", capture_payload)
        self.assertIn("client_gate:", capture_payload)
        for forbidden in (
            "target_x",
            "target_y",
            "cursor",
            "word",
            "text",
            "cognitive",
            "model_name",
        ):
            self.assertNotIn(forbidden, capture_payload)
        self.assertIn("payload.consumed === true", source)
        self.assertIn('payload.retryable === true', source)
        self.assertIn('payload.classification === "attributable_sensor_failure"', source)

    def test_gate_is_explicitly_structural_and_resets_on_context_change(self) -> None:
        source = GATE.read_text(encoding="utf-8")
        self.assertIn("const MINIMUM_DWELL_MS = 900", source)
        self.assertIn("stableFrames >= 2", source)
        self.assertIn('return reset("document_hidden")', source)
        self.assertIn('return reset("document_unfocused")', source)
        self.assertIn('return reset("viewport_changed")', source)
        self.assertIn('return reset("target_render_mismatch")', source)
        self.assertIn("structural_browser_gate_only: true", source)
        self.assertIn("visual_attestation_claimed: false", source)

    def test_node_behavior_contract_is_independently_runnable(self) -> None:
        source = NODE_TEST.read_text(encoding="utf-8")
        self.assertIn('require("node:assert/strict")', source)
        self.assertIn(
            'require("../web/static/measurement_ceiling_gate.js")',
            source,
        )
        self.assertIn(
            'require("../web/static/measurement_ceiling_client_policy.js")',
            source,
        )
        self.assertIn(
            "webcam gaze measurement frontend gate tests passed",
            source,
        )

    def test_verified_analysis_is_authenticated_and_never_promotes_claims(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn('id="analyzeRunButton"', template)
        self.assertIn('id="analysisSummary"', template)
        analysis = source[
            source.index("async function analyzeRun()") :
            source.index("async function abortRun()")
        ]
        self.assertIn('requestJson("/analysis"', analysis)
        self.assertIn("headers: authHeaders()", analysis)
        self.assertIn('body: "{}"', analysis)
        self.assertIn('"measurement_claim_authorized=false"', analysis)
        self.assertIn('"threshold_selected=false"', analysis)
        self.assertNotIn("run_token", analysis)

    def test_recoverable_run_blocks_creation_and_abort_keeps_token_until_cleanup(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        policy = POLICY.read_text(encoding="utf-8")
        self.assertIn("Boolean(recoverableContext) || Boolean(credentials)", policy)
        self.assertIn("payload.cleanup_verified === true", policy)
        self.assertIn("state.recoverableContext = stored", source)
        self.assertIn("state.recoverableContext || state.credentials", source)
        abort = source[source.index("async function abortRun()") : source.index("function showError")]
        confirmation = abort.index("cleanupConfirmed")
        clear = abort.index("state.credentials = null")
        reload = abort.index("globalThis.location.reload()")
        self.assertLess(confirmation, clear)
        self.assertLess(clear, reload)
        finally_body = abort[abort.index("finally") :]
        self.assertNotIn("state.credentials = null", finally_body)
        self.assertNotIn("location.reload", finally_body)

    def test_exact_65_unusable_is_rendered_as_terminal_negative(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        self.assertIn('"calibration_unusable_negative_result"', POLICY.read_text(encoding="utf-8"))
        self.assertIn("function showCalibrationNegative(payload)", source)
        self.assertIn("calibrationNegativeDisplay(payload)", source)
        policy = POLICY.read_text(encoding="utf-8")
        self.assertIn("source.usable_calibration_count", policy)
        self.assertIn("source.images_purged === true", policy)
        self.assertIn("source.cleanup_verified === true", policy)
        self.assertIn("不進入 evaluation、不補抽直到成功，也不自動建立新 run", source)

    def test_prepared_retry_keeps_frame_and_reload_uses_server_spool(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        policy = POLICY.read_text(encoding="utf-8")
        self.assertIn("payload.exact_frame_retry_required === true", policy)
        self.assertIn("payload.prepared === true", policy)
        self.assertIn("payload.server_spool_retry_available === true", policy)
        capture_start = source.index("async function captureCurrentTarget")
        retry_start = source.index("if (payload.retryable === true)", capture_start)
        retry_end = source.index("throw new Error", retry_start)
        retry = source[retry_start:retry_end]
        self.assertIn(
            "if (!exactRetry && !serverSpoolRetry) state.pendingCapture = null",
            retry,
        )
        self.assertIn('state.retryMode = serverSpoolRetry ? "server_spool"', retry)
        resume_start = source.index("async function resumeRun()")
        resume_end = source.index("async function issueChallenge()", resume_start)
        resume = source[resume_start:resume_end]
        self.assertIn("current.challenge_recovery", resume)
        self.assertIn("serverSpoolRetryAvailable", resume)
        self.assertIn("await resumeServerSpool()", resume)
        self.assertNotIn("不可用新 frame 冒充 exact retry", resume)
        spool = source[
            source.index("async function resumeServerSpool()") :
            source.index("async function captureCurrentTarget")
        ]
        self.assertIn("resume_server_spool: true", spool)
        self.assertNotIn("captureSnapshot()", spool)
        self.assertNotIn("image_data", spool)

    def test_abort_required_is_terminal_until_verified_cleanup(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        policy = POLICY.read_text(encoding="utf-8")
        self.assertIn("payload.abort_required === true", policy)
        self.assertIn('payload.classification === "abort_required"', policy)
        self.assertIn("function showAbortRequired(payload)", source)
        self.assertIn("不可繼續擷取或自動建立新 run", source)
        self.assertIn("cleanup_verified=true", source)

    def test_lost_challenge_response_can_rotate_via_status_recovery(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        self.assertIn('state.retryMode = "recover"', source)
        self.assertIn('ui.retryButton.textContent = "核對狀態並恢復"', source)
        self.assertIn("await resumeRun()", source)
        resume = source[
            source.index("async function resumeRun()") :
            source.index("async function issueChallenge()")
        ]
        self.assertIn("current.challenge_outstanding === true", resume)
        self.assertNotIn("current.active_challenge", resume)
        self.assertNotIn("current.has_active_challenge", resume)
        self.assertIn("await rotateChallenge()", resume)

    def test_pending_create_authority_is_persisted_before_request_and_replayed(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        policy = POLICY.read_text(encoding="utf-8")
        self.assertIn("newCreateAuthority", policy)
        self.assertIn("randomHex(32, fillRandom)", policy)
        self.assertNotIn("randomUUID", source)
        create_start = source.index("async function createRun()")
        submit_start = source.index("async function submitPendingCreate()")
        create = source[create_start:submit_start]
        persist = create.index("persistContext()")
        submit = create.index("submitPendingCreate()")
        self.assertLess(persist, submit)
        pending_submit = source[
            submit_start : source.index("async function activateCreatedRun", submit_start)
        ]
        self.assertIn("[CREATE_REQUEST_ID_HEADER]", pending_submit)
        self.assertIn("[RUN_TOKEN_HEADER]", pending_submit)
        self.assertIn("[PREFLIGHT_TOKEN_HEADER]", pending_submit)
        body = pending_submit[pending_submit.index("body: JSON.stringify") :]
        self.assertNotIn("runToken", body)
        self.assertNotIn("createRequestId", body)
        resume = source[
            source.index("async function resumeRun()") :
            source.index("async function issueChallenge()")
        ]
        self.assertIn("isPendingCreate(stored)", resume)
        self.assertIn("await submitPendingCreate()", resume)
        self.assertIn("invalid_stored_context", source)
        self.assertNotIn("sessionStorage.removeItem(STORAGE_KEY);\n      return null;", source)

    def test_pending_create_can_replace_only_preflight_after_server_proof(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        policy = POLICY.read_text(encoding="utf-8")
        self.assertIn("function canReplacePendingPreflight(context)", policy)
        replacement = source[
            source.index("function markPendingCreatePreflightReplacement") :
            source.index("async function createRun()")
        ]
        self.assertIn(
            'payload?.classification !== "pending_create_preflight_required"',
            replacement,
        )
        self.assertIn("payload?.existing_run !== false", replacement)
        self.assertIn("payload?.authority_retained !== true", replacement)
        self.assertIn("payload?.replace_preflight_allowed !== true", replacement)
        self.assertIn("state.pendingCreate.preflightToken = null", replacement)
        self.assertNotIn("state.pendingCreate.createRequestId =", replacement)
        self.assertNotIn("state.pendingCreate.runToken =", replacement)
        preflight = source[
            source.index("async function runReadinessPreflight()") :
            source.index("function updateProgress")
        ]
        self.assertIn("replacingPendingPreflight", preflight)
        self.assertIn(
            "state.pendingCreate.preflightToken = state.preflightToken",
            preflight,
        )


if __name__ == "__main__":
    unittest.main()
