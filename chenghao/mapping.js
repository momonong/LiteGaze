console.log("mapping.js loaded");
let gazeMappingOn = false;
const gazeMappingToggle = document.getElementById("gazeMappingToggle");
const gazeMappingLabel = document.getElementById("gazeMappingLabel");

gazeMappingToggle.addEventListener("click", () => {
  gazeMappingOn = !gazeMappingOn;

  gazeMappingToggle.classList.toggle("active", gazeMappingOn);
  gazeMappingLabel.textContent = gazeMappingOn
    ? "啟用 Gaze Mapping（開啟）"
    : "啟用 Gaze Mapping（關閉）";

  if (!gazeMappingOn) {
    clearAllGazeHighlights();
  }
});

document.addEventListener("mousemove", (e) => {
  processGazeOnExtractedData(e.clientX, e.clientY);
});

function processGazeOnExtractedData(gazeX, gazeY) {
  if (!gazeMappingOn) return;

  const match = findNearestExtractedWord(gazeX, gazeY);

  console.log(match);

  if (!match) {
    clearAllGazeHighlights();
    return;
  }

  highlightExtractedWord(match);
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

  for (const [pageNum, pageData]
    of pageOverlayMap.entries()) {

    const { items } = pageData;

    const pageWrap =
      document.querySelectorAll(".page-wrap")[pageNum - 1];

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

        if (distance <= MEDIUM_DISTANCE) {
          confidence = "medium";
        }

        // ===== LOW =====

        else if (distance <= LOW_DISTANCE) {
          confidence = "low";
        }

        if (confidence) {

          const candidate = {
            pageNum,
            item,
            confidence,
            distance
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

function clearAllGazeHighlights() {
  pageOverlayMap.forEach(({ overlayCanvas }) => {
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  });
}

function highlightExtractedWord(match) {
  clearAllGazeHighlights();
  const pageData =
    pageOverlayMap.get(match.pageNum);

  if (!pageData) return;

  const { overlayCanvas, items } = pageData;

  overlayCanvas.style.display = "block";
  const ctx =
    overlayCanvas.getContext("2d");

  ctx.clearRect(
    0,
    0,
    overlayCanvas.width,
    overlayCanvas.height
  );

  //highlight

  ctx.fillStyle =
    "rgba(255, 220, 80, 0.45)";

  ctx.strokeStyle =
    "rgba(255, 180, 0, 0.9)";

  ctx.lineWidth = 2;

  ctx.fillRect(
    match.item.left,
    match.item.top,
    match.item.width,
    match.item.height
  );

  ctx.strokeRect(
    match.item.left,
    match.item.top,
    match.item.width,
    match.item.height
  );

  console.log(
    "Matched:",
    match.item.text
  );
}

window.processGazeOnExtractedData = processGazeOnExtractedData;

