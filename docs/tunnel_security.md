# Authenticated remote tunnel

LexiGaze local mode remains convenient for development. Public ngrok mode is a
separate, fail-closed runtime profile: every application route requires a
secret, large requests are capped, and model-heavy operations receive a global
rate and concurrency budget.

## Start a temporary tunnel

```powershell
uv run python -X utf8 run.py --tunnel
```

If `LEXIGAZE_TUNNEL_TOKEN` is absent, the process creates a cryptographically
random ephemeral secret. The terminal prints separate Reading and Calibration
links. The secret lives after `#` in those links, so browsers do not send it in
HTTP request lines, Flask/ngrok logs, or referrer headers. The bootstrap page
exchanges it for an eight-hour `Secure`, `HttpOnly`, `SameSite=Strict` cookie and
immediately removes the fragment from browser history.

Share an access link only with the participant who needs it. LexiGaze does not
send the link to an external QR-code service. If a QR code is needed, generate
it locally from the printed link.

Tunnel setup fails closed: invalid configuration, an ngrok download failure, or
an ngrok startup failure exits without falling back to an unprotected server.
The standalone `scripts/setup_remote_collection.py` entry point is disabled so
it cannot expose a local-mode Flask process by accident.

## Stable token and API clients

Set a stable token only when a participant must reconnect across process
restarts. It must be at least 32 UTF-8 bytes and should be randomly generated:

```powershell
$env:LEXIGAZE_TUNNEL_TOKEN = uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
uv run python -X utf8 run.py --tunnel
```

Non-browser clients can send the raw token without using the bootstrap page:

```text
Authorization: Bearer <LEXIGAZE_TUNNEL_TOKEN>
```

`X-LexiGaze-Token` is also accepted for clients that cannot set an Authorization
header. Tokens in query parameters are intentionally rejected.

## Default public-mode budgets

| Control | Default | Environment override |
| --- | ---: | --- |
| Request body | 64 MB | `LEXIGAZE_TUNNEL_MAX_UPLOAD_MB` (1–256) |
| Mutating requests | 300/minute | `LEXIGAZE_TUNNEL_MUTATIONS_PER_MINUTE` |
| Real-time gaze predictions | 900/minute (15 Hz) | `LEXIGAZE_TUNNEL_PREDICTIONS_PER_MINUTE` |
| Expensive requests | 12/minute | `LEXIGAZE_TUNNEL_EXPENSIVE_PER_MINUTE` |
| Expensive concurrency | 1 | `LEXIGAZE_TUNNEL_EXPENSIVE_CONCURRENCY` (1–8) |
| Invalid login attempts | 10/minute | `LEXIGAZE_TUNNEL_AUTH_ATTEMPTS_PER_MINUTE` |
| Browser session | 8 hours | `LEXIGAZE_TUNNEL_SESSION_TTL_SECONDS` |

Real-time gaze prediction has a separate 15 Hz budget because the reading UI
normally polls around 11 Hz. It bypasses the generic mutation counter but still
shares the single high-cost execution slot. The low-frequency expensive class
covers cognitive model warmup/analysis, gaze training, video analysis, dataset
reprocessing, and Inspector operations that invoke an external generative
model. Limits are global per process on purpose: they cannot be bypassed by
forging proxy headers and they bound aggregate GPU, CPU, memory, and paid-API
pressure.

Local mode is unchanged and retains the existing 500 MB request limit. These
guards are defense in depth for temporary research collection; they do not turn
the Flask development server into a general-purpose production deployment.

## CPU-only verification

The dedicated suite stubs its high-cost handlers and does not load or run model
weights:

```powershell
uv run python -X utf8 -m unittest scripts.test_tunnel_security -v
```
