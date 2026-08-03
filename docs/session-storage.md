# Session Storage

LexiGaze stores extracted document-layout sessions as UUID-named JSON files. The Flask application owns a `SessionStore` instance at `app.extensions["lexigaze_session_store"]`.

## Configuration

| Flask setting | Default | Purpose |
| --- | ---: | --- |
| `LEXIGAZE_DATA_DIR` | `<repository>/data` | Session storage directory; use a disposable path in tests. |
| `LEXIGAZE_SESSION_MAX_BYTES` | 64 MiB | Maximum request and encoded session size. |
| `LEXIGAZE_SESSION_MAX_ITEMS` | 250,000 | Maximum number of layout items in one session. |

`filename` is limited to 255 characters, `filetype` to 64 characters, and every `items` entry must be a JSON object. Non-finite numbers and other non-JSON values are rejected.

## Integrity behavior

- Route identifiers are parsed as UUIDs and converted to canonical form before a path is constructed.
- New files are serialized and validated before publication.
- Each write uses a unique temporary file in the destination directory, flushes and synchronizes its contents, and publishes it with `os.replace`.
- A process-local re-entrant lock coordinates operations. Atomic replacement keeps completed files visible to readers in other processes as well.
- Stored records are checked for required fields, UUID/filename agreement, item shape, and `item_count` consistency.
- Symbolic-link session files are rejected.
- The list endpoint omits corrupt entries but emits a warning containing the affected filename. `/api/ping` exposes an additive `corrupt_sessions` count.

The successful API response shapes are unchanged. Error responses are JSON objects with stable `error` and `message` fields. Invalid IDs and payloads return `400`, unsupported media types return `415`, configured size violations return `413`, missing sessions return `404`, and corrupt or failed storage operations return `500`.

## Operational limits

This store provides atomic single-file persistence; it is not a replacement for backups, cross-file transactions, authorization, retention policy, or a multi-node database. A corrupt file is never modified automatically. Preserve it for diagnosis, use the logged UUID filename to inspect it, and restore or remove it through an explicit operator action.

## Verification

Run the offline CPU quality gate from the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_offline_quality_gate
```

The session tests use only disposable directories and cover compatible CRUD response shapes, malformed JSON, traversal-like IDs, resource limits, corrupt records, failed publication cleanup, and concurrent access. See `docs/experiments/2026-08-04-session-data-integrity.md` for the compatibility audit and measured latency.
