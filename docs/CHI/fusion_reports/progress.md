# LexiGaze Fusion Module — Progress Report

> **Location:** `docs/CHI/fusion_reports/progress.md`  
> **Last updated:** 2026-06-15  
> **Branch:** `feat/module-fusion`  
> **Status:** 🟡 Infrastructure Complete — UI trigger pending

---

## Table of Contents

1. [What LexiGaze Is](#1-what-lexigaze-is)
2. [The Core Problem We Are Solving](#2-the-core-problem-we-are-solving)
3. [What Was Already Built (Before This Session)](#3-what-was-already-built-before-this-session)
4. [What We Built This Session](#4-what-we-built-this-session)
5. [How the Fusion Algorithm Works](#5-how-the-fusion-algorithm-works)
6. [File Map — Where Everything Lives](#6-file-map--where-everything-lives)
7. [What Is NOT Done Yet](#7-what-is-not-done-yet)
8. [Recommended Next Steps](#8-recommended-next-steps)
9. [How to Test Right Now](#9-how-to-test-right-now)

---

## 1. What LexiGaze Is

LexiGaze is a reading research platform that combines two signals to understand how difficult a person finds a piece of text:

| Signal | Question it answers | Technology |
|---|---|---|
| **Eye tracking** | *Where* is the reader looking? | UniGaze-B16 (ViT neural net) + webcam |
| **Text modeling** | *How cognitively hard* is each word? | BERT / GPT-2 surprisal + Ridge Regression |

On their own, each signal is incomplete. Eye tracking tells you coordinates — it doesn't know what word is there or whether it's hard. The text model tells you word difficulty — but it has no idea whether the reader actually looked at it or skipped it.

**Fusion is the bridge that makes both signals useful together.**

---

## 2. The Core Problem We Are Solving

```
Eye Tracker Output:           Text Model Output:
  (x=556, y=410)                "neuro-symbolic" → load_score: 0.92
                                "the"            → load_score: 0.05
  ↑                             "calibration"    → load_score: 0.75
  Just pixels.                  ↑
  No word info.                 Just linguistic scores.
                                No spatial info.
```

**The fusion layer answers:** *"For each word the reader actually looked at, how difficult was it — and did the reader's eye behaviour (dwell time, re-reading) confirm that difficulty?"*

---

## 3. What Was Already Built (Before This Session)

These modules were fully working before we started:

### `shengwen/` + `chenghao/gaze_core/` — Eye Tracker
- Webcam → MediaPipe face detection → UniGaze-B16 → `(pitch, yaw)` angles
- Polynomial calibration: maps raw gaze angles to screen pixel coordinates
- Runs at ~8 Hz (one prediction every 120 ms)
- Flask endpoint: `POST /api/gaze/predict` → returns `screen_xy_px`

### `weichi/` — Text Model (Cognitive Load Pipeline)
- Takes raw text → BERT/GPT-2 → per-word scores
- Features per word: `surprisal`, `entropy`, `dependency_load`, `zipf_score`, `word_length`, `aoa_score`, `pos_score`
- Final output: `load_score ∈ [0, 1]` per word (normalised cognitive difficulty)
- Flask endpoint: `POST /api/cognitive/analyze/text` → returns `word_analysis[]`

### `chenghao/` — Integration Hub (Frontend)
- `word_track.html` — PDF viewer, renders word bounding boxes extracted by PDF.js
- `gaze_integration.js` — runs the 8 Hz gaze prediction loop
- `mapping.js` — `findNearestExtractedWord(x, y)` maps a gaze coordinate to the nearest word bounding box, with confidence tiers (`high` / `medium` / `low`)
- `cognitive_routes.py` — Flask Blueprint wrapping the weichi pipeline
- `server.py` — main Flask server, port 8080

### What the system could do before fusion:
1. Render a PDF with word-level bounding boxes
2. Show a live gaze cursor moving on the page
3. Highlight the word the user is looking at (visual only)
4. Colour words by cognitive load (after running text analysis)

**But:** There was no connection between "this word was looked at for 600ms" and "this word has load_score 0.92". Those two facts lived in separate parts of the code and were never combined.

---

## 4. What We Built This Session

### 4.1 Documentation & Analysis

**`docs/CHI/system_architecture.md` — Section 7 (Fusion Algorithm)**
- Defined the fusion as a 3-stage pipeline: Gaze-to-Word Anchoring → Load Score Lookup → RDS calculation
- Documented the RDS formula with weight rationale
- Compared current (frontend-only) vs proposed (backend) architecture

**`docs/CHI/fusion_blueprint.md` — Data Format Investigation**
- Investigated actual field names and value ranges in both modules' real code
- Identified 4 concrete alignment bugs (missing timestamps, no `word` field in gaze data, case mismatch, hyphenated word mismatch)
- Specified the exact Flask integration point (line numbers in `server.py`)

---

### 4.2 Backend: Fusion Orchestrator

**`scripts/fusion/orchestrator.py`** — the core fusion engine

This is a standalone Python script (also importable as a library) that:

1. **Reads** a `gaze_log.jsonl` file (per-word dwell + fixation events from a reading session)
2. **Reads** a `cognitive_result.json` (output of `CognitiveLoadPipeline.run()`)
3. **Aligns** them using case-insensitive lookup + hyphenated word fallback
4. **Computes** RDS per word using the formula:
   ```
   RDS(w) = 0.35 × dwell_norm + 0.25 × fixation_norm + 0.40 × load_score
   ```
5. **Outputs** a complete JSON report to `docs/fusion_reports/<session_id>.json`

The report includes all raw linguistic features alongside the RDS score for every word that was looked at.

**Verified working** — smoke test confirmed:
```
neuro-symbolic  → RDS=0.9680  (difficulty)
calibration     → RDS=0.4750  (attention)
the             → RDS=0.0200  (fluent)
```

---

### 4.3 Backend: Flask Blueprint

**`chenghao/fusion_routes.py`** — HTTP interface to the orchestrator

Added 3 new API endpoints (no existing endpoints were changed):

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/fuse/` | Run fusion on a batch of gaze events + cognitive result |
| `GET` | `/api/fuse/reports` | List all saved fusion reports |
| `GET` | `/api/fuse/reports/<session_id>` | Fetch a specific report |

**`chenghao/server.py`** — registered the new Blueprint (2 lines added, nothing changed):
```python
from fusion_routes import fusion_bp
app.register_blueprint(fusion_bp)
```

---

### 4.4 Frontend: Gaze Buffer

**`chenghao/gaze_integration.js`** (+74 lines, all additive)

Added a silent in-memory buffer that accumulates data as the user reads:

```
Every 120ms:
  gaze (x,y) → screen → [existing code, unchanged] → word highlight

NEW (runs alongside, does not interrupt existing flow):
  if word found → recordGazeHit("neuro-symbolic", "high")
                    → gazeBuffer["neuro-symbolic"].dwell_count++
```

New functions exposed on `window`:
- `window.recordGazeHit(word, confidence)` — called automatically by mapping.js
- `window.flushGazeBuffer()` — returns the accumulated buffer as a JSON-ready array
- `window.clearGazeBuffer()` — resets for a new session
- `window.exportFusion(cognitiveResult, sessionId, persist)` — fires `POST /api/fuse`

**`chenghao/mapping.js`** (+23 lines, one guarded hook added)

When a word is matched, now also records it to the buffer:
```javascript
if (gazeMatch && typeof window.recordGazeHit === "function") {
  window.recordGazeHit(gazeMatch.item.text, gazeMatch.confidence);
}
```
Guard means: if `gaze_integration.js` hasn't loaded, this silently skips. **No impact on existing behaviour.**

Also added `window.lookupCognitive(text)` — a helper that does case-insensitive + hyphen-fallback lookup into the cognitive score table.

---

### 4.5 Does Any of This Break Existing Code?

**No.** Here is the proof:

| Change | Breaks anything? | Why not |
|---|---|---|
| `server.py` +2 lines | No | Only adds new routes under `/api/fuse/*` |
| `gaze_integration.js` +74 lines | No | New code is inert until `recordGazeHit` is called; loop unchanged |
| `mapping.js` hook | No | Wrapped in `typeof === "function"` guard |
| `fusion_routes.py` (new file) | No | New file, doesn't touch existing files |
| `orchestrator.py` (new file) | No | New file in `scripts/` |

---

## 5. How the Fusion Algorithm Works

```
┌─────────────────────────────────────────────────────────┐
│                    READING SESSION                       │
│                                                         │
│  User uploads PDF → word bounding boxes extracted       │
│  User runs text analysis → load_score per word          │
│  User enables gaze tracking → 8 Hz loop starts         │
│                                                         │
│  Every 120ms:                                           │
│    camera frame → UniGaze-B16 → (x, y) on screen       │
│    (x, y) → findNearestExtractedWord() → "neuro-symbolic"│
│    → recordGazeHit("neuro-symbolic", "high")            │
│    → gazeBuffer["neuro-symbolic"].dwell_count += 1      │
└─────────────────────────────────────────────────────────┘
                          │
              [session ends / user clicks Export]
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│               POST /api/fuse                            │
│                                                         │
│  Input A: gaze_events (from gazeBuffer)                 │
│    { word: "neuro-symbolic", dwell_count: 5,            │
│      fixation_count: 2, confidence: "high" }            │
│                                                         │
│  Input B: cognitive_result (from last text analysis)    │
│    { word: "neuro-symbolic", load_score: 0.92,          │
│      surprisal: 14.2, entropy: 0.83, ... }              │
│                                                         │
│  Algorithm:                                             │
│    1. Aggregate dwell_count × 120ms = dwell_ms per word │
│    2. Min-Max normalise dwell_ms and fixation_count     │
│       across all words in session                       │
│    3. Look up load_score from cognitive result          │
│    4. RDS = 0.35×dwell_norm + 0.25×fix_norm            │
│             + 0.40×load_score                           │
│    5. Classify: ≥0.70 → difficulty                      │
│                 0.40–0.69 → attention                   │
│                 <0.40 → fluent                          │
│                                                         │
│  Output: per-word RDS list + full linguistic features   │
└─────────────────────────────────────────────────────────┘
                          │
              [if persist=true]
                          ▼
         docs/fusion_reports/<session_id>.json
```

### RDS Weight Rationale

| Weight | Signal | Why this weight |
|---|---|---|
| 0.40 | `load_score` | Most reliable — validated against GECO eye-tracking corpus |
| 0.35 | `dwell_norm` | Strong behavioural signal; long dwell = difficulty |
| 0.25 | `fix_norm` | Re-reading behaviour (secondary evidence) |

---

## 6. File Map — Where Everything Lives

```
lexigaze/
│
├── chenghao/                          ← Flask integration hub
│   ├── server.py                      ← MODIFIED: +2 lines (fusion_bp registered)
│   ├── fusion_routes.py               ← NEW: POST /api/fuse Blueprint
│   ├── gaze_integration.js            ← MODIFIED: +74 lines (gazeBuffer added)
│   ├── mapping.js                     ← MODIFIED: +23 lines (recordGazeHit hook)
│   ├── cognitive_routes.py            ← unchanged
│   ├── gaze_routes.py                 ← unchanged
│   └── gaze_core/                     ← unchanged
│       ├── inference.py
│       ├── training.py
│       └── sample_store.py
│
├── scripts/
│   └── fusion/
│       └── orchestrator.py            ← NEW: standalone CLI + importable library
│
├── weichi/
│   ├── cognitive_load_pipeline.py     ← unchanged (text model)
│   └── ridge_model.json               ← unchanged (pre-trained weights)
│
├── shengwen/
│   └── src/unigaze_personalization/   ← unchanged (gaze model)
│
└── docs/
    ├── fusion_reports/                ← NEW dir: JSON output from /api/fuse
    └── CHI/
        ├── system_architecture.md     ← MODIFIED: Section 7 (Fusion) added
        ├── fusion_blueprint.md        ← NEW: data format investigation report
        └── fusion_reports/
            └── progress.md            ← THIS FILE
```

---

## 7. What Is NOT Done Yet

### 🔴 Critical — Must do before the fusion is usable

**1. UI trigger button in `word_track.html`**

The gaze buffer accumulates data silently, but there is **no button** to send it to `/api/fuse`. The user cannot currently trigger a fusion analysis from the UI.

What needs to be added to `word_track.html`:
```javascript
// When user clicks "Run Fusion Analysis":
const result = await window.exportFusion(
  lastCognitiveResult,   // ← the response from the last /api/cognitive/analyze call
  currentSessionId,      // ← the session ID from /api/sessions
  true                   // ← persist report to disk
);
// Then render result.rds as a sorted list
```

**2. `lastCognitiveResult` variable in `word_track.html`**

The cognitive analysis result is currently consumed and displayed (word colours) but **not stored** in a variable accessible to `exportFusion()`. The existing analyze handler needs to save it:
```javascript
// In the existing cognitive analysis success handler:
window.lastCognitiveResult = result;   // ← add this one line
```

---

### 🟡 Important — Should do for a complete experiment

**3. `gaze_log.jsonl` persistence (for offline CLI use)**

The CLI orchestrator (`scripts/fusion/orchestrator.py`) reads from a `gaze_log.jsonl` file. But currently the frontend doesn't write this file — it only holds data in memory. To support offline analysis and reproducibility, the gaze buffer should also be periodically written to `POST /api/sessions` or a dedicated endpoint.

**4. RDS visualisation in `word_track.html`**

After fusion runs, the RDS results should be shown visually. The simplest version: overlay words in a separate colour scheme based on `rds_level` (red = difficulty, orange = attention, green = fluent), distinct from the existing cognitive load colours.

**5. Chinese language support in RDS**

`aoa_score` is `0.0` for all Chinese words (no Kuperman word table for Chinese). The RDS weights should be adjusted for Chinese mode — redistribute the 0.40 weight across the remaining features.

---

### 🟢 Nice to Have — Future work

**6. Cross-session aggregation**

The current system computes RDS for one session at a time. A future `GET /api/fuse/aggregate?participant=alice` could average RDS across multiple sessions for the same reader and document.

**7. Real-time RDS overlay**

Instead of computing RDS at session end, compute it incrementally every N seconds and update the word colours live. This requires a sliding window over the gaze buffer.

**8. Export to CSV for statistical analysis**

Add `GET /api/fuse/reports/<session_id>/csv` that returns the RDS results as a CSV, ready for R or Python analysis.

---

## 8. Recommended Next Steps

Do these in order:

### Step 1 — Wire up the UI (30 min, one file)

In `chenghao/word_track.html`, find the block where the cognitive analysis response is handled (search for `word_analysis`) and add:

```javascript
window.lastCognitiveResult = result;  // save for fusion
```

Then add a "Run Fusion" button somewhere in the gaze control panel that calls:
```javascript
document.getElementById("runFusionBtn").addEventListener("click", async () => {
  if (!window.lastCognitiveResult) {
    alert("請先執行文字分析");
    return;
  }
  const data = await window.exportFusion(
    window.lastCognitiveResult,
    currentSessionId || "session_" + Date.now(),
    true   // persist to disk
  );
  if (data && data.rds) {
    displayFusionResults(data.rds);
  }
});
```

### Step 2 — Display results (1–2 hrs)

Write `displayFusionResults(rds)` to show a ranked list of words by difficulty, or overlay a new colour layer on the PDF.

### Step 3 — Test with a real reading session (1 session)

Run a real reading session:
1. Upload a PDF
2. Run cognitive analysis
3. Enable gaze tracking, read for ~2 minutes
4. Click "Run Fusion"
5. Check `docs/fusion_reports/<session_id>.json`

### Step 4 — Iterate on weights

After seeing real data, the default weights `(0.35, 0.25, 0.40)` may need tuning. The orchestrator makes this easy — weights are constants at the top of `orchestrator.py`.

---

## 9. How to Test Right Now

### Test A — Orchestrator CLI (no server needed)

Create a test gaze log:
```bash
cat > /tmp/test_gaze.jsonl << 'EOF'
{"word": "neuro-symbolic", "confidence": "high", "dwell_count": 8, "fixation_count": 3, "timestamp_ms": 1000}
{"word": "the", "confidence": "low", "dwell_count": 1, "fixation_count": 1, "timestamp_ms": 1120}
{"word": "calibration", "confidence": "medium", "dwell_count": 4, "fixation_count": 2, "timestamp_ms": 1360}
{"word": "reading", "confidence": "high", "dwell_count": 2, "fixation_count": 1, "timestamp_ms": 1600}
EOF
```

Save a cognitive result:
```bash
cat > /tmp/test_cognitive.json << 'EOF'
{
  "model": "gpt2", "lang": "en", "domain": "general",
  "word_analysis": [
    {"word": "neuro-symbolic", "load_score": 0.92, "load_level": "high",   "pos": "NOUN", "surprisal": 14.2, "entropy": 0.83, "renyi_entropy": 0.61, "dependency_load": 0.45, "zipf_score": 1.1, "word_length": 14, "aoa_score": 0.71, "pos_score": 1.0},
    {"word": "the",            "load_score": 0.05, "load_level": "low",    "pos": "DET",  "surprisal": 2.1,  "entropy": 0.12, "renyi_entropy": 0.1,  "dependency_load": 0.0,  "zipf_score": 7.5, "word_length": 3,  "aoa_score": 0.0,  "pos_score": 0.1},
    {"word": "calibration",    "load_score": 0.75, "load_level": "high",   "pos": "NOUN", "surprisal": 11.3, "entropy": 0.70, "renyi_entropy": 0.55, "dependency_load": 0.30, "zipf_score": 2.8, "word_length": 11, "aoa_score": 0.60, "pos_score": 1.0},
    {"word": "reading",        "load_score": 0.30, "load_level": "low",    "pos": "VERB", "surprisal": 5.8,  "entropy": 0.40, "renyi_entropy": 0.35, "dependency_load": 0.10, "zipf_score": 5.2, "word_length": 7,  "aoa_score": 0.20, "pos_score": 1.0}
  ]
}
EOF
```

Run the orchestrator:
```bash
cd /home/ubuntu/projects/lexigaze
python scripts/fusion/orchestrator.py \
  --gaze-log /tmp/test_gaze.jsonl \
  --cognitive /tmp/test_cognitive.json \
  --session-id test_session_001
```

Check output:
```bash
cat docs/fusion_reports/test_session_001.json | python -m json.tool | head -50
```

### Test B — HTTP endpoint (server running)

```bash
curl -s -X POST http://localhost:8080/api/fuse/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "curl_test",
    "persist": true,
    "cognitive_result": {
      "word_analysis": [
        {"word": "calibration", "load_score": 0.75, "load_level": "high", "pos": "NOUN",
         "surprisal": 11.3, "entropy": 0.7, "renyi_entropy": 0.55,
         "dependency_load": 0.3, "zipf_score": 2.8, "word_length": 11,
         "aoa_score": 0.6, "pos_score": 1.0}
      ]
    },
    "gaze_events": [
      {"word": "calibration", "confidence": "high", "dwell_count": 6, "fixation_count": 2}
    ]
  }' | python -m json.tool
```

### Test C — Browser console (server running, with PDF loaded)

Open DevTools console on `http://localhost:8080` and run:
```javascript
// Simulate some gaze hits
window.recordGazeHit("calibration", "high");
window.recordGazeHit("calibration", "high");
window.recordGazeHit("the", "low");

// See what's accumulated
console.log(window.gazeBuffer);

// See what would be sent
console.log(window.flushGazeBuffer());
```

---

*Document written by Antigravity — 2026-06-15*  
*Branch: `feat/module-fusion`*
