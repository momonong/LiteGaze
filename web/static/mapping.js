console.log("mapping.js loaded");
let gazeMappingOn = false;
window.gazeMappingOn = false;
const gazeMappingToggle = document.getElementById("gazeMappingToggle");
const gazeMappingLabel = document.getElementById("gazeMappingLabel");

let mouseMatch = null;
let gazeMatch = null;

const gazeOverlayMap = new Map();

gazeMappingToggle.addEventListener("click", () => {
  gazeMappingOn = !gazeMappingOn;
  window.gazeMappingOn = gazeMappingOn;
  mouseMatch = null;
  gazeMatch = null;

  gazeMappingToggle.classList.toggle("active", gazeMappingOn);
  gazeMappingLabel.textContent = gazeMappingOn
    ? "啟用 Gaze Mapping(開啟）"
    : "啟用 Gaze Mapping(關閉）";

  if (!gazeMappingOn) {
    clearAllGazeHighlights();
  }
});

let _mouseRaf = null;
document.addEventListener("mousemove", (e) => {
  if (!gazeMappingOn) return;
  cancelAnimationFrame(_mouseRaf);
  _mouseRaf = requestAnimationFrame(() => {
    processMouseGroundTruth(e.clientX, e.clientY);
  });
}, { passive: true });

function processMouseGroundTruth(mouseX, mouseY) {
  if (!gazeMappingOn) return;
  mouseMatch = findNearestExtractedWord(mouseX, mouseY);
  drawHighlights();
}

function processGazeOnExtractedData(gazeX, gazeY) {
  if (!gazeMappingOn) return;
  gazeMatch = findNearestExtractedWord(gazeX, gazeY);

  // ── Feed the fusion gaze buffer ──────────────────────────────────
  // recordGazeHit is exposed by gaze_integration.js via window.recordGazeHit
  if (gazeMatch && typeof window.recordGazeHit === "function") {
    window.recordGazeHit(gazeMatch.item.text, gazeMatch.confidence, gazeMatch.item.index);
  }

  drawHighlights(gazeX, gazeY);
}

function distanceToExtractedRect(x, y, item) {
  const dx = Math.max(item.left - x, 0, x - item.right);
  const dy = Math.max(item.top - y, 0, y - item.bottom);
  return Math.sqrt(dx * dx + dy * dy);
}

function findNearestExtractedWord(gazeX, gazeY) {
  let best = null;

  const LINE_Y_THRESHOLD = 90;
  const MEDIUM_DISTANCE = 35;
  const LOW_DISTANCE = 90;

  const pageWraps = document.querySelectorAll(".page-wrap");

  const overlays = window.pageOverlayMap || (typeof pageOverlayMap !== "undefined" ? pageOverlayMap : null);
  if (!overlays) return null;

  for (const [pageNum, pageData] of overlays.entries()) {

    const { items } = pageData;

    const pageWrap = pageWraps[pageNum - 1];

    if (!pageWrap) continue;

    const pageRect =
      pageWrap.getBoundingClientRect();

    const localX =
      gazeX - pageRect.left;

    const localY =
      gazeY - pageRect.top;

    for (const item of items) {

      const insideActualBox =
        localX >= item.left &&
        localX <= item.right &&
        localY >= item.top &&
        localY <= item.bottom;

      const distance =
        distanceToExtractedRect(localX, localY, item);

      // Apply Cognitive Mass attraction (rare/difficult words reduce effective distance)
      const cogData = (typeof window.lookupCognitive === 'function') ? window.lookupCognitive(item.text) : null;
      const loadScore = cogData ? (cogData.score || cogData.load_score || 0.0) : 0.0;
      const cognitiveMass = 1.0 + 1.8 * loadScore; // More difficult = higher mass = larger pull
      const effectiveDistance = distance / cognitiveMass;

      const wordCenterY =
        item.top + item.height / 2;

      const yDistance =
        Math.abs(localY - wordCenterY);

      // ===== HIGH =====

      if (insideActualBox) {

        return {
          pageNum,
          item,
          confidence: "high",
          distance: 0
        };
      }

      // ===== SAME LINE =====

      if (yDistance <= LINE_Y_THRESHOLD) {

        let confidence = null;

        // ===== MEDIUM =====

        if (effectiveDistance <= MEDIUM_DISTANCE) {
          confidence = "medium";
        }

        // ===== LOW =====

        else if (effectiveDistance <= LOW_DISTANCE) {
          confidence = "low";
        }

        if (confidence) {

          const candidate = {
            pageNum,
            item,
            confidence,
            distance: effectiveDistance
          };

          if (!best ||
              candidate.distance < best.distance) {

            best = candidate;
          }
        }
      }
    }
  }

  return best;
}

function drawHighlights(gazeX, gazeY) {

  clearAllGazeHighlights();

  if (mouseMatch && gazeMatch && mouseMatch.pageNum === gazeMatch.pageNum && mouseMatch.item === gazeMatch.item) {
    highlightExtractedWord(mouseMatch,"correct");
    return;
  }

  if (mouseMatch) {
    highlightExtractedWord(mouseMatch,"mouse");
  }

  if (gazeMatch) {
    highlightExtractedWord(gazeMatch, "gaze");
    if (gazeX !== undefined && gazeY !== undefined) {
      drawGazeWordFusionAttractor(gazeMatch, gazeX, gazeY);
    }
  }
}

function drawGazeWordFusionAttractor(match, gazeX, gazeY) {
  const canvas = getGazeCanvas(match.pageNum);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const item = match.item;

  const pageWraps = document.querySelectorAll(".page-wrap");
  const pageWrap = pageWraps[match.pageNum - 1];
  if (!pageWrap) return;
  const pageRect = pageWrap.getBoundingClientRect();

  const localGazeX = gazeX - pageRect.left;
  const localGazeY = gazeY - pageRect.top;

  const wordCenterX = item.left + item.width / 2;
  const wordCenterY = item.top + item.height / 2;

  // Retrieve cognitive mass pull
  const cogData = (typeof window.lookupCognitive === 'function') ? window.lookupCognitive(item.text) : null;
  const loadScore = cogData ? (cogData.score || cogData.load_score || 0.0) : 0.0;
  const cognitiveMass = 1.0 + 1.8 * loadScore; 
  const pullFactor = 1.0 - (1.0 / cognitiveMass);

  // Fused position pulled towards the word center
  const fusedX = localGazeX + (wordCenterX - localGazeX) * pullFactor;
  const fusedY = localGazeY + (wordCenterY - localGazeY) * pullFactor;

  // 1. Draw gravity pull line (Feather-like clean styling)
  ctx.beginPath();
  ctx.moveTo(localGazeX, localGazeY);
  ctx.lineTo(wordCenterX, wordCenterY);
  ctx.strokeStyle = "rgba(124, 58, 237, 0.5)"; // Deep Purple
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.setLineDash([]); // reset

  // 2. Draw raw gaze dot (noisy position)
  ctx.beginPath();
  ctx.arc(localGazeX, localGazeY, 5, 0, 2 * Math.PI);
  ctx.fillStyle = "rgba(239, 68, 68, 0.6)"; // Coral Red
  ctx.fill();
  ctx.strokeStyle = "rgba(220, 38, 38, 0.8)";
  ctx.lineWidth = 1;
  ctx.stroke();

  // 3. Draw fused gaze dot (corrected position)
  ctx.beginPath();
  ctx.arc(fusedX, fusedY, 7, 0, 2 * Math.PI);
  ctx.fillStyle = "rgba(16, 185, 129, 0.75)"; // Emerald Green
  ctx.fill();
  ctx.strokeStyle = "rgba(4, 120, 87, 0.9)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // 4. Draw labels
  ctx.fillStyle = "rgba(220, 38, 38, 0.9)";
  ctx.font = "8px 'JetBrains Mono', monospace";
  ctx.fillText("Raw 雜訊", localGazeX + 10, localGazeY + 3);

  ctx.fillStyle = "rgba(4, 120, 87, 0.95)";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.fillText(`Fused CM=${cognitiveMass.toFixed(2)}`, fusedX + 12, fusedY + 3);

  // 5. Draw gravity field concentric rings around the word center if cognitiveMass is high
  if (loadScore > 0) {
    const numRings = Math.min(3, Math.ceil(loadScore * 3));
    for (let r = 1; r <= numRings; r++) {
      ctx.beginPath();
      ctx.arc(wordCenterX, wordCenterY, 15 + r * 10 * loadScore, 0, 2 * Math.PI);
      ctx.strokeStyle = `rgba(124, 58, 237, ${0.15 / r})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}

function clearAllGazeHighlights() {
  gazeOverlayMap.forEach((canvas) => {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  });
}

function getGazeCanvas(pageNum) {
  const pageWraps = document.querySelectorAll(".page-wrap");
  const wrap = pageWraps[pageNum - 1];
  if (!wrap) return null;

  let canvas = gazeOverlayMap.get(pageNum);
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.className = "gaze-overlay-canvas";
    canvas.style.position = "absolute";
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.pointerEvents = "none";
    canvas.style.zIndex = "5";
    wrap.appendChild(canvas);
    gazeOverlayMap.set(pageNum, canvas);
  }

  const rect = wrap.getBoundingClientRect();
  const w = Math.round(rect.width);
  const h = Math.round(rect.height);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }
  return canvas;
}

function highlightExtractedWord(match, type) {
  const canvas = getGazeCanvas(match.pageNum);
  if (!canvas) return;

  canvas.style.display = "block";
  const ctx = canvas.getContext("2d");

  if (type === "correct") {
    ctx.fillStyle = "rgba(0,255,0,0.45)";
    ctx.strokeStyle = "rgba(0,200,0,1)";
  } else if (type === "mouse") {
    ctx.fillStyle = "rgba(80, 180, 255, 0.45)";
    ctx.strokeStyle = "rgba(0, 120, 255, 0.9)";
  } else {
    ctx.fillStyle = "rgba(255, 220, 80, 0.45)";
    ctx.strokeStyle = "rgba(255, 180, 0, 0.9)";
  }

  ctx.lineWidth = 2;
  ctx.fillRect(match.item.left, match.item.top, match.item.width, match.item.height);
  ctx.strokeRect(match.item.left, match.item.top, match.item.width, match.item.height);
}

window.processGazeOnExtractedData = processGazeOnExtractedData;

// ── Cognitive lookup helper (case-normalised + hyphen fallback + baseline heuristic) ────────────
// cognitiveLookup is expected to be set by word_track.html after a
// /api/cognitive/analyze/* response arrives.
//   { "word".toLowerCase(): WordResult }
window.lookupCognitive = function lookupCognitive(text) {
  let score = 0.0;
  let level = 'Medium';
  let ent_type = '';

  if (window.cognitiveLookup) {
    const key = text.toLowerCase();
    let found = null;
    if (window.cognitiveLookup[key]) found = window.cognitiveLookup[key];
    else {
      for (const [k, v] of Object.entries(window.cognitiveLookup)) {
        if (k.includes("-") && k.split("-").includes(key)) {
          found = v;
          break;
        }
      }
    }
    if (found) {
      return {
        ...found,
        score: found.load_score !== undefined ? found.load_score : (found.score || 0.0)
      };
    }
  }

  // Fallback: heuristic cognitive mass based on word features
  // This ensures that even before running the LLM analysis, the "cognitive mass" attraction is functional and visible!
  const clean = text.toLowerCase().replace(/[^a-z0-9]/g, '');
  const stopwords = new Set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me']);
  if (clean.length > 0 && !stopwords.has(clean)) {
    // Longer words and rarer words get higher mass
    // e.g., length 3 -> 0.0, length 8 -> 0.5, length 12 -> 0.9
    score = Math.min(0.95, Math.max(0.0, (clean.length - 4) * 0.12));
    if (score > 0.6) level = 'High';
    else if (score < 0.3) level = 'Low';
  }
  return { score, level, ent_type };
};

