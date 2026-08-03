# Session Data Integrity Hardening

## Status

- Date: 2026-08-04 (Asia/Taipei)
- Branch: `feat/session-data-integrity`
- Stacked base: `codex/test/offline-cpu-quality-gate` at `2ee6fea`
- State: implementation and local/Ubuntu validation complete; Draft PR #9 open
- Compute policy: CPU-only, offline quality gate; no Torch import and no LexiGaze GPU allocation
- Evaluation policy: storage and API contract tests only; no model, prompt, or question-answer dataset is used

## Goal

Make the file-backed session API safe under invalid input, interrupted writes, corrupt files, and concurrent requests without changing the successful response shapes consumed by existing clients or modifying production session data during tests.

## Baseline

The pre-change implementation in `web/__init__.py`:

- interpolates an unvalidated route parameter into the session file path;
- accepts forced JSON parsing without requiring an object or validating `items`;
- writes the destination file directly, so an interruption can expose partial JSON;
- catches every list/read failure and silently drops the affected entry;
- uses a process-global data directory and import-time directory creation, which makes isolated tests difficult;
- uses a naive local timestamp and has no concurrency boundary around file operations.

Production-data compatibility was sampled read-only before implementation:

| Metric | Baseline |
| --- | ---: |
| Root-level session JSON files | 60 |
| Total bytes | 1,528,573 |
| Minimum / maximum file size | 22,456 / 40,886 bytes |
| Valid JSON files | 60 / 60 |
| Missing required fields | 0 |
| `item_count` mismatches | 0 |

No existing file was changed by the inspection.

## Compatibility contract

- `GET /api/sessions` remains a JSON array of summary objects.
- Successful `POST /api/sessions` remains `201` with `id` and `created_at`.
- Successful `GET /api/sessions/<id>` returns the full stored object.
- Successful `DELETE /api/sessions/<id>` remains `{ "ok": true }`.
- Existing UUID-named JSON records and their naive ISO timestamps remain readable.
- Validation and corruption failures return stable JSON errors instead of HTML tracebacks.

## Planned design

1. Move persistence into a standard-library-only `SessionStore` with an app-configurable root.
2. Canonicalize UUID identifiers before constructing paths.
3. Validate the request envelope, metadata lengths, item count, and encoded JSON size.
4. Serialize to a unique temporary file in the destination directory, flush and sync it, then publish with `os.replace`.
5. Guard list/read/write/delete operations with a re-entrant lock and clean up temporary files after failures.
6. Log corrupt records with their path while preserving the list endpoint's response shape.
7. Exercise the store and Flask API in disposable directories through the offline CPU quality gate.

## Completion criteria

- All 60 existing session files pass the new reader and summary validation without mutation.
- Traversal-like and malformed IDs cannot escape the configured root and return `400`.
- Invalid bodies fail before any session or temporary file is created.
- A simulated publish failure leaves no partial destination and no temporary artifact.
- Concurrent create/list/get/delete stress completes without corrupt JSON or leaked temporary files.
- The full offline gate passes with network/process denial, `torch_imported=false`, no repository artifact changes, and unchanged GPU memory attributable to this test lane.

## Results

### Implementation

- Added a standard-library-only `SessionStore` and moved all session routes onto it.
- Added configurable storage root, 64 MiB encoded/request limit, and 250,000-item limit.
- Added UUID canonicalization, symlink rejection, UTC timestamps, strict JSON validation, atomic publication, temporary-file cleanup, and process-local locking.
- Preserved every successful route response shape and added stable JSON errors.
- Added integrity visibility through warning logs and the additive `corrupt_sessions` ping field.

### Existing-data compatibility

The new reader scanned the production `data` directory 25 times. A SHA-256, size, and nanosecond-mtime snapshot was compared before and after the experiment.

| Result | Value |
| --- | ---: |
| Existing files accepted | 60 / 60 |
| Corrupt files | 0 |
| Content/metadata snapshot unchanged | yes |
| Scan median | 16.369 ms |
| Scan p95 | 21.497 ms |

### Synthetic write benchmark

The benchmark used 250 writes per mode with 51 synthetic layout objects per session. Both modes used UTF-8 and indented JSON; the hardened mode additionally validates, flushes, calls `fsync`, and atomically replaces the destination.

| Mode | Median | p95 |
| --- | ---: | ---: |
| Previous direct write | 0.688 ms | 1.186 ms |
| Validated atomic write | 1.971 ms | 3.252 ms |

The hardened path adds about 1.28 ms at the median. Session persistence is user-event frequency rather than frame frequency, so the absolute cost is acceptable for the integrity guarantee.

### Concurrency soak

Eight independent `SessionStore` instances and 16 worker threads created, read, and deleted 512 sessions in a disposable directory:

- 512 unique IDs and 512 valid records after creation;
- 512 / 512 records read back successfully;
- zero corrupt records;
- zero records and zero temporary files remaining after deletion.

### Offline quality gate

Command:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_offline_quality_gate --timeout-seconds 180
```

Result on Windows / Python 3.11.9:

- 27 tests run in 1.927 seconds; zero failures and zero errors;
- one symlink test skipped because the local Windows account cannot create symlinks;
- supervisor completed in 2.803 seconds without timeout;
- network probe and process-spawn probe blocked;
- credentials cleared and repository artifact changes empty;
- `CUDA_VISIBLE_DEVICES=-1` and `torch_imported=false`;
- RTX 5090 Laptop GPU memory was 5,544 MiB before and after. Utilization changed from
  25% to 23% because of the pre-existing external workload; this worker had CUDA hidden
  and never imported Torch.

No model inference, training, prompt tuning, question-answer dataset, or external benchmark was used. These results therefore cannot overfit a question-answer set; they measure only deterministic storage and API behavior.

GitHub Actions then validated the pushed commit on Linux / Python 3.12.13:

- 27 tests passed in 0.787 seconds with zero skips; the symbolic-link rejection test executed successfully;
- worker completed in 1.233 seconds and the supervisor in 1.468 seconds;
- network/process probes were blocked, credentials were cleared, artifact changes were empty, and Torch was not imported;
- the runner exposed no NVIDIA GPU;
- GitGuardian Security Checks also passed.

Draft PR: <https://github.com/momonong/lexigaze/pull/9>
