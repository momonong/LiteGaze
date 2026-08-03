# Tunnel security guardrails — 2026-08-04

## Objective

Make temporary public collection fail closed without changing local developer
behavior or spending GPU/model budget. Completion requires authenticated browser
and API access, bounded request/resource pressure, a non-bypassable startup
path, offline regression coverage, and one real public-tunnel verification.

Branch: `codex/feat/tunnel-security-guardrails`

## Baseline findings

- `run.py --tunnel` exposed every page, saved session, report, dataset, model,
  upload, training, and inference endpoint without authentication.
- ngrok download/start failures silently continued as a local-only server,
  obscuring whether the requested public mode was actually active.
- The calibration URL was sent to a third-party QR-code service.
- Public mode inherited the local 500 MB body limit and had no aggregate rate or
  high-cost concurrency budget.
- Running `scripts/setup_remote_collection.py` directly bypassed any app-level
  tunnel profile.
- ngrok archives used `extractall`, and URL discovery queried the shared,
  unauthenticated local Agent API on port 4040. A different ngrok process could
  therefore be mistaken for the process LexiGaze just started.

## Decisions and implementation

1. `create_app(tunnel_mode=True)` requires a token of at least 32 UTF-8 bytes.
   The CLI generates an ephemeral 256-bit token when a stable environment token
   is absent. Invalid settings raise before ngrok starts.
2. Browser links carry credentials in the URL fragment, not the query string.
   A minimal no-store bootstrap exchanges the secret for an HMAC-signed,
   server-expiring `Secure`, `HttpOnly`, `SameSite=Strict`, `__Host-` cookie and
   removes the fragment. API clients use Bearer or `X-LexiGaze-Token` headers.
3. Public defaults are 64 MB/request, 300 mutations/minute, 900 real-time gaze
   predictions/minute, 12 low-frequency expensive operations/minute, one shared
   high-cost execution slot, and 10 invalid login attempts/minute. Valid tokens
   are checked before the invalid-attempt budget so an attacker cannot consume
   the counter and lock out a participant.
4. Real-time prediction has its own 15 Hz class because the production reading
   loop polls around every 90 ms. Treating it like training/LLM work at 12/minute
   would pass synthetic tests but break actual reading after seconds.
5. Every route is authenticated in tunnel mode. Same-origin navigation is
   validated, query tokens are rejected, security headers are added, and rate
   failures return `429` plus `Retry-After`.
6. The standalone tunnel entry point now refuses to run. ngrok failure exits
   instead of falling back. Download size is capped at 100 MB, only the expected
   executable is atomically extracted, HTTP request inspection is disabled, and
   the HTTPS URL is read from and continuously drained from the exact child
   process's JSON log.
7. Tunnel mode binds and starts the already-protected WSGI app before spawning
   ngrok. A pre-existing process on port 8080 therefore causes an immediate exit
   and can never be forwarded through a newly created LexiGaze tunnel.
8. The adaptive regression suite explicitly blanks `GEMINI_API_KEY`, preventing
   a developer's `.env` from turning an offline test into paid external calls.

The ngrok design was checked against the official Agent documentation: the
[local Agent API has no authentication](https://ngrok.com/docs/agent/api), while
the [CLI supports stdout logging and JSON log format](https://ngrok.com/docs/agent/cli).

## Verification results

All commands were run with model execution disabled; the relevant regression
commands also set `CUDA_VISIBLE_DEVICES` to an empty value.

| Verification | Result |
| --- | --- |
| `scripts.test_tunnel_security` | 16/16 passed |
| `scripts.test_adaptive_stepper` | 2/2 passed offline |
| `scripts.test_fusion_routes` | 1/1 passed |
| `scripts.test_cognitive_inspector` | 11/11 passed |
| Python `compileall` for changed runtime/tests | passed |
| Ruff on all changed Python files | passed |
| `git diff --check` | passed |
| Actual public ngrok HTTPS smoke test (final startup order) | passed; unauthenticated `401`, Bearer `200`, login `200`, signed-cookie API `200` |
| Raw token present in bootstrap HTML | false |
| Nonce-based CSP active, without `unsafe-inline` | true |
| Tunnel/server cleanup after public smoke test | completed |
| GPU after tests | 0% utilization, 88 MiB display/app allocation; no Python model workload |

The live smoke test used a random token and an ephemeral local port, asserted
the complete result map, then stopped both ngrok and Werkzeug in a `finally`
block. No public URL or token was persisted in this record.

## Dataset and overfitting check

No model was trained, fine-tuned, selected, or threshold-tuned. No GECO, PROVO,
question-answer, or other evaluation corpus was loaded. Route budgets were based
on production call cadence and resource semantics, not benchmark answers, so
this work cannot overfit a QA dataset.

## Residual risks

- An access link is a bearer credential. Anyone who receives it can authenticate
  until the process/token is rotated; links must be shared privately.
- Rate counters and semaphores are process-local and reset on restart. A future
  multi-worker production deployment needs a shared limiter.
- The ngrok archive is fetched over TLS and extraction is constrained, but the
  project still lacks a pinned vendor checksum or platform signature gate.
- Flask's development server remains appropriate only for temporary research
  collection, not an unattended internet service.
- The 64 MB public upload limit may need an explicit, bounded override for long
  videos.

## Delivery

The commit identifier and Draft PR URL live in Git/GitHub metadata; they are not
embedded here because a commit cannot stably contain its own hash.
