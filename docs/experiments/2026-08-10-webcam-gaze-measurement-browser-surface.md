# Webcam gaze measurement browser surface

Date: 2026-08-10
Status: implementation and offline verification complete; no physical capture claim

## Purpose

This is the dedicated local browser/server surface for the frozen 193-row
webcam-gaze measurement-ceiling protocol. It is not a participant Visit flow,
does not consume an invite, and does not write to the participant-study store.

The surface runs only on the exact authority `127.0.0.1:8099` by default. It
uses the existing shared capture contract and the measurement runner's
server-owned schedule. Cursor position, reading text, cognitive profile, model
selection, and client-provided target labels are forbidden at the capture API.

## Runtime

From the branch checkout, use the external Python 3.11 runtime with the
project packages and explicit CPU/offline guards:

```powershell
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONPATH = 'D:\projects\lexigaze\.venv\Lib\site-packages'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$python = 'C:\Users\morris\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe'
& $python -X utf8 -m scripts.run_webcam_gaze_measurement_server
```

Use `--no-browser` to start the listener without opening a browser. The script
forces `CUDA_VISIBLE_DEVICES=-1`, Hugging Face/Transformers offline mode, and
serves with Waitress on `127.0.0.1`. The default data root resolves from a
temporary branch workspace back to `D:\projects\lexigaze`, so calibration
images do not move to C:.

## Browser-to-run contract

1. Start the self-facing camera with no audio.
2. Complete a target-free, non-persistent readiness preflight: three distinct,
   consecutive HTTP-200 predictions from the fixed `before` baseline.
3. Generate a client create-request ID and 256-bit run secret, persist both in
   `sessionStorage` before POST, and transmit them only in headers.
4. The route performs authenticated persistent lookup before consuming the
   ephemeral preflight receipt. A lost response replays the same authority.
5. Targets come only from a one-time server challenge. The client proves page
   visibility, focus, frozen viewport/DPR, two stable render frames, target
   center, and at least 900 ms dwell. This is a structural browser gate, not
   visual attestation; the gate result is discarded before inference.
6. No-face is an attributable consumed row. It is not silently replaced.
7. A pre-inference hard error may capture a new frame under the same challenge.
   Durable prepared/save work must resume the encrypted server spool; it may
   not capture a different frame. A committed exact retry returns the receipt
   without replaying inference.
8. A calibration with fewer than 65 usable rows is a terminal negative result.
   The UI reports attempts, usable count, and verified image purge, and it does
   not enter evaluation or silently create another run.
9. Abort credentials remain in `sessionStorage` until the server returns both
   `ok=true` and `cleanup_verified=true`.
10. After the artifact is verified, the authenticated analysis endpoint can
    re-read the frame-free artifact, attempt sidecar, training provenance,
    bound model, and purge evidence through the canonical live runner. The UI
    reports descriptive geometry and uncertainty diagnostics while keeping
    both claim flags and threshold selection false.

If the server proves that no run exists and the original readiness receipt is
expired or consumed, the browser keeps the same create-request ID/run secret
and permits replacing only the three-frame preflight receipt.

## Storage and claim boundary

- Readiness frames and evaluation frames are ephemeral.
- Successful calibration rows temporarily use the dedicated D-drive session's
  `raw`, `crop`, and `normalized_face` files until training, terminal failure,
  or authenticated abort completes verified purge.
- An encrypted crash-recovery spool can briefly persist one frame.
- The derived ledgers and final acquisition artifact contain no images.
- Completion of this browser flow alone does not authorize physical-capture,
  accuracy, generalization, reading, cognitive, holdout, uncertainty, or
  abstention claims.
- The frozen measurement protocol canonical SHA-256 is
  `be4dfb0956ce3594546336fe7a54da6ba878f2d6fcd457d36cbaf0159132fced`;
  the frozen analysis definition canonical SHA-256 is
  `d3118fb8a1cb4eff437ea45e2b9b4619ce78e856d2bfbf84a4acef80f278755a`.

## Security boundary

- Exact Host and loopback remote checks reject DNS rebinding, `localhost`
  aliases, other ports, non-loopback clients, query strings, cross-origin/site
  requests, and CORS preflight.
- Only the measurement page, its exact static allowlist, and its exact API
  allowlist are reachable. Legacy `/`, `/gaze`, `/api/sessions`, participant
  routes, and unrelated static assets are denied in measurement mode.
- Mutation endpoints require `Content-Type: application/json`, an object body,
  exact fields, and a 12 MiB request ceiling.
- Responses use `no-store`, strict CSP without inline script/style, camera-only
  Permissions Policy, no CORS headers, and a minimal `Server-Timing` duration.
- Run, challenge, preflight, and create authority plaintext is absent from URL,
  JSON response bodies, application logs, and persistent server state.

## Verification evidence

- 56 focused Python tests passed across preflight, browser gate, security
  policy, frontend source contracts, and Flask routes.
- The final measurement-specific adversarial slice passed 120/120 tests across
  schedule/protocol/preflight/browser gate, store, runner, analysis, frontend,
  route, and security boundaries, with no remaining P0/P1 finding.
- The latest complete offline quality gate passed 475/475 tests with zero
  failures, errors, or skips. The unittest body took 248.935 seconds, the
  recorded gate duration was 250.383 seconds, and the supervisor took 250.703
  seconds. Network/process probes were blocked, credentials were cleared, and
  `artifact_changes=[]` and `network_attempts=[]`.
- Node behavior tests and syntax checks passed for all measurement JavaScript.
- 11 existing app-factory and gaze-provenance route tests passed unchanged.
- A real Python 3.11 offline app build returned HTTP 200 from the dedicated
  health endpoint with `cpu_only=true`, `offline=true`, and
  `measurement_claim_authorized=false`.
- Python compilation and `git diff --check` passed for the touched surface.
- No GPU path or network research was used. `CUDA_VISIBLE_DEVICES=-1` was set
  for every Python verification command. The final gate reported
  `torch_imported=false` and 0% GPU utilization before and after; NVIDIA memory
  snapshots were 2782 MiB before and 2913 MiB after. The external ambient
  131 MiB drift is not attributed to the CUDA-hidden worker.
- The tracked final gate result is
  [`results/2026-08-10-webcam-gaze-measurement-ceiling-v1-final-quality-gate.json`](results/2026-08-10-webcam-gaze-measurement-ceiling-v1-final-quality-gate.json),
  byte SHA-256
  `9a2043172e996a7599bac56281d6b068fde7ef66ea55593ce4ea1aea3c7d54c8`.

The in-app browser did not grant permission to navigate to the local test URL,
so visual/camera interaction was not forced or bypassed. A real camera
permission check and a full 193-row human capture remain operational validation
steps. The authenticated analysis route has not yet been run on human 193-row
evidence, so no measurement result is claimed here.
