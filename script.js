/* =====================================================================
   PatchTrack — Feature Vision
   Shi-Tomasi + Lucas-Kanade Optical Flow via OpenCV.js
   ===================================================================== */

'use strict';

// ── STATE ─────────────────────────────────────────────────────────────
let cvReady      = false;
let mode         = null;     // 'image' | 'video'
let animId       = null;
let frameCount   = 0;
let features     = [];       // [{x, y, dx, dy, score, id}]
let prevGray     = null;
let prevPts      = null;

// OpenCV Mats kept persistent (video mode)
let matPrevGray  = null;
let matCurrGray  = null;

// Settings (driven by sliders)
let MAX_FEATURES = 50;
let PATCH_SIZE   = 40;
let QUALITY_LVL  = 0.03;

// Redetect every N frames (video)
const REDETECT_INTERVAL = 45;

// Motion simulation (image mode)
const DRIFT_SCALE = 12;

// ── DOM REFS ──────────────────────────────────────────────────────────
const uploadZone    = document.getElementById('uploadZone');
const fileInput     = document.getElementById('fileInput');
const viewer        = document.getElementById('viewer');
const srcVideo      = document.getElementById('srcVideo');
const srcImage      = document.getElementById('srcImage');
const canvas        = document.getElementById('outputCanvas');
const ctx           = canvas.getContext('2d');
const cvStatusDot   = document.getElementById('cvStatus');
const cvStatusText  = document.getElementById('cvStatusText');
const frameCounter  = document.getElementById('frameCounter');
const featureCounter= document.getElementById('featureCounter');
const hudLabel      = document.getElementById('hudLabel');
const resetBtn      = document.getElementById('resetBtn');
const maxFeaturesEl = document.getElementById('maxFeatures');
const maxFeaturesVal= document.getElementById('maxFeaturesVal');
const patchSizeEl   = document.getElementById('patchSize');
const patchSizeVal  = document.getElementById('patchSizeVal');
const qualityEl     = document.getElementById('quality');
const qualityVal    = document.getElementById('qualityVal');

// ── OPENCV READY CALLBACK ─────────────────────────────────────────────
function onOpenCvReady() {
  cv['onRuntimeInitialized'] = () => {
    cvReady = true;
    cvStatusDot.classList.add('ready');
    cvStatusText.textContent = 'CV ENGINE READY';
  };
  // Sometimes already initialised
  if (cv.Mat) {
    cvReady = true;
    cvStatusDot.classList.add('ready');
    cvStatusText.textContent = 'CV ENGINE READY';
  }
}

// ── SLIDER WIRING ─────────────────────────────────────────────────────
maxFeaturesEl.addEventListener('input', () => {
  MAX_FEATURES = +maxFeaturesEl.value;
  maxFeaturesVal.textContent = MAX_FEATURES;
  if (mode === 'image') restartImage();
});
patchSizeEl.addEventListener('input', () => {
  PATCH_SIZE = +patchSizeEl.value;
  patchSizeVal.textContent = PATCH_SIZE;
});
qualityEl.addEventListener('input', () => {
  QUALITY_LVL = +qualityEl.value / 100;
  qualityVal.textContent = QUALITY_LVL.toFixed(2);
  if (mode === 'image') restartImage();
});

// ── FILE UPLOAD ───────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

resetBtn.addEventListener('click', () => {
  stopAll();
  uploadZone.style.display = '';
  viewer.style.display = 'none';
  mode = null;
  fileInput.value = '';
});

// ── HANDLE FILE ───────────────────────────────────────────────────────
function handleFile(file) {
  stopAll();
  const url = URL.createObjectURL(file);
  if (file.type.startsWith('image/')) {
    mode = 'image';
    loadImage(url);
  } else if (file.type.startsWith('video/')) {
    mode = 'video';
    loadVideo(url);
  } else {
    alert('Unsupported file type.');
    return;
  }
  uploadZone.style.display = 'none';
  viewer.style.display = '';
}

// ── IMAGE MODE ────────────────────────────────────────────────────────
function loadImage(url) {
  srcImage.onload = () => {
    setCanvasSize(srcImage.naturalWidth, srcImage.naturalHeight);
    if (!cvReady) {
      waitForCv(startImageMode);
    } else {
      startImageMode();
    }
  };
  srcImage.src = url;
}

function waitForCv(cb) {
  if (cvReady) { cb(); return; }
  cvStatusText.textContent = 'WAITING FOR CV…';
  const t = setInterval(() => { if (cvReady) { clearInterval(t); cb(); } }, 200);
}

function startImageMode() {
  hudLabel.textContent = 'IMAGE MODE — FEATURE DETECTION';
  features = detectImageFeatures(srcImage);
  animateImage();
}

function restartImage() {
  cancelAnimationFrame(animId);
  features = detectImageFeatures(srcImage);
  animateImage();
}

function detectImageFeatures(img) {
  const tmp = document.createElement('canvas');
  tmp.width  = img.naturalWidth;
  tmp.height = img.naturalHeight;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(img, 0, 0);

  const src  = cv.imread(tmp);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  const corners = new cv.Mat();
  const mask    = new cv.Mat();
  cv.goodFeaturesToTrack(gray, corners, MAX_FEATURES, QUALITY_LVL, 10, mask, 3, false, 0.04);

  const result = [];
  for (let i = 0; i < corners.rows; i++) {
    const x     = corners.data32F[i * 2];
    const y     = corners.data32F[i * 2 + 1];
    const score = computeLocalVariance(gray, Math.round(x), Math.round(y), 5);
    result.push({
      x, y,
      ox: x, oy: y,
      dx: (Math.random() - 0.5) * DRIFT_SCALE,
      dy: (Math.random() - 0.5) * DRIFT_SCALE,
      score: score.toFixed(1),
      id: i,
      phase: Math.random() * Math.PI * 2
    });
  }
  src.delete(); gray.delete(); corners.delete(); mask.delete();
  featureCounter.textContent = `FEATURES: ${result.length}`;
  return result;
}

function computeLocalVariance(gray, cx, cy, r) {
  const W = gray.cols, H = gray.rows;
  let sum = 0, sum2 = 0, count = 0;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      const v = gray.ucharAt(ny, nx);
      sum += v; sum2 += v * v; count++;
    }
  }
  if (count === 0) return 0;
  const mean = sum / count;
  return Math.sqrt(Math.max(0, sum2 / count - mean * mean));
}

// Image animation loop — simulate drift
let imgT = 0;
function animateImage() {
  imgT += 0.018;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(srcImage, 0, 0, canvas.width, canvas.height);

  for (const f of features) {
    const ox = f.ox + Math.sin(imgT + f.phase) * f.dx;
    const oy = f.oy + Math.cos(imgT * 0.9 + f.phase) * f.dy;
    f.cx = ox; f.cy = oy;
    drawPatch(f.ox, f.oy, ox, oy, f.score, f.id, canvas.width / srcImage.naturalWidth);
  }

  frameCount++;
  frameCounter.textContent = `FRAME: ${frameCount}`;
  animId = requestAnimationFrame(animateImage);
}

// ── VIDEO MODE ────────────────────────────────────────────────────────
function loadVideo(url) {
  srcVideo.src = url;
  srcVideo.onloadeddata = () => {
    setCanvasSize(srcVideo.videoWidth, srcVideo.videoHeight);
    srcVideo.play();
    if (!cvReady) {
      waitForCv(startVideoMode);
    } else {
      startVideoMode();
    }
  };
}

function startVideoMode() {
  hudLabel.textContent = 'VIDEO MODE — OPTICAL FLOW';
  frameCount = 0;
  features   = [];
  matPrevGray = new cv.Mat();
  matCurrGray = new cv.Mat();
  processVideoFrame();
}

function processVideoFrame() {
  if (srcVideo.paused || srcVideo.ended) {
    animId = requestAnimationFrame(processVideoFrame);
    return;
  }

  // Draw source frame to canvas
  ctx.drawImage(srcVideo, 0, 0, canvas.width, canvas.height);

  // Grab pixel data
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const src = cv.matFromImageData(imgData);
  cv.cvtColor(src, matCurrGray, cv.COLOR_RGBA2GRAY);

  const scale = canvas.width / srcVideo.videoWidth;

  if (frameCount === 0 || features.length < 5 || frameCount % REDETECT_INTERVAL === 0) {
    // Detect fresh features
    detectVideoFeatures(matCurrGray, scale);
  } else {
    // Track with Lucas-Kanade
    trackFeatures(matCurrGray, scale);
  }

  // Draw patches over the canvas frame
  ctx.drawImage(srcVideo, 0, 0, canvas.width, canvas.height);
  for (const f of features) {
    drawPatch(f.px, f.py, f.x, f.y, f.score, f.id, 1);
  }

  // Swap
  matCurrGray.copyTo(matPrevGray);
  src.delete();

  frameCount++;
  frameCounter.textContent  = `FRAME: ${frameCount}`;
  featureCounter.textContent = `FEATURES: ${features.length}`;

  animId = requestAnimationFrame(processVideoFrame);
}

function detectVideoFeatures(gray, scale) {
  const corners = new cv.Mat();
  const mask    = new cv.Mat();
  cv.goodFeaturesToTrack(gray, corners, MAX_FEATURES, QUALITY_LVL, 10, mask, 3, false, 0.04);

  features = [];
  for (let i = 0; i < corners.rows; i++) {
    const rx = corners.data32F[i * 2];
    const ry = corners.data32F[i * 2 + 1];
    const cx = rx * scale, cy = ry * scale;
    features.push({ x: cx, y: cy, px: cx, py: cy, score: '0.0', id: i, rx, ry });
  }
  corners.delete(); mask.delete();
}

function trackFeatures(currGray, scale) {
  if (features.length === 0 || matPrevGray.empty()) return;

  // Build previous points Mat
  const prevPtsMat = new cv.Mat(features.length, 1, cv.CV_32FC2);
  for (let i = 0; i < features.length; i++) {
    prevPtsMat.data32F[i * 2]     = features[i].rx;
    prevPtsMat.data32F[i * 2 + 1] = features[i].ry;
  }

  const nextPtsMat = new cv.Mat();
  const status     = new cv.Mat();
  const err        = new cv.Mat();

  try {
    cv.calcOpticalFlowPyrLK(
      matPrevGray, currGray,
      prevPtsMat, nextPtsMat,
      status, err,
      new cv.Size(21, 21), 3,
      new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
      0, 1e-4
    );

    const kept = [];
    for (let i = 0; i < features.length; i++) {
      if (status.data[i] !== 1) continue;
      const rx  = nextPtsMat.data32F[i * 2];
      const ry  = nextPtsMat.data32F[i * 2 + 1];
      const cx  = rx * scale, cy = ry * scale;
      const ddx = cx - features[i].x;
      const ddy = cy - features[i].y;
      const mag = Math.sqrt(ddx * ddx + ddy * ddy).toFixed(1);
      kept.push({
        px: features[i].x, py: features[i].y,
        x: cx, y: cy,
        score: mag,
        id: features[i].id,
        rx, ry
      });
    }
    features = kept;
  } catch(e) {
    // If LK fails silently re-detect next frame
    features = [];
  }

  prevPtsMat.delete(); nextPtsMat.delete(); status.delete(); err.delete();
}

// ── DRAWING ───────────────────────────────────────────────────────────
const PATCH_ALPHA   = 0.28;
const BORDER_ALPHA  = 0.75;
const LINE_ALPHA    = 0.85;

function drawPatch(fromX, fromY, toX, toY, score, id, scale) {
  const half = (PATCH_SIZE * scale) / 2;

  // Semi-transparent fill
  ctx.save();
  ctx.globalAlpha = PATCH_ALPHA;
  ctx.fillStyle   = '#0088cc';
  ctx.fillRect(toX - half, toY - half, PATCH_SIZE * scale, PATCH_SIZE * scale);
  ctx.restore();

  // Border
  ctx.save();
  ctx.globalAlpha   = BORDER_ALPHA;
  ctx.strokeStyle   = '#00e5ff';
  ctx.lineWidth     = 1.2 * scale;
  ctx.setLineDash([4 * scale, 3 * scale]);
  ctx.strokeRect(toX - half, toY - half, PATCH_SIZE * scale, PATCH_SIZE * scale);
  ctx.restore();

  // Motion line
  const dx = toX - fromX, dy = toY - fromY;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist > 0.5) {
    ctx.save();
    ctx.globalAlpha   = LINE_ALPHA;
    ctx.strokeStyle   = '#00ff88';
    ctx.lineWidth     = 1.5 * scale;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(dy, dx);
    const aLen  = 7 * scale;
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(
      toX - aLen * Math.cos(angle - 0.45),
      toY - aLen * Math.sin(angle - 0.45)
    );
    ctx.moveTo(toX, toY);
    ctx.lineTo(
      toX - aLen * Math.cos(angle + 0.45),
      toY - aLen * Math.sin(angle + 0.45)
    );
    ctx.stroke();
    ctx.restore();
  }

  // Feature point dot
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.fillStyle   = '#00e5ff';
  ctx.beginPath();
  ctx.arc(toX, toY, 2.5 * scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Score label
  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.font        = `${Math.round(10 * scale)}px 'Share Tech Mono', monospace`;
  ctx.fillStyle   = '#00ff88';
  ctx.fillText(`${score}`, toX - half + 3 * scale, toY - half + 12 * scale);
  ctx.restore();

  // ID label (tiny, corner)
  ctx.save();
  ctx.globalAlpha = 0.5;
  ctx.font        = `${Math.round(8 * scale)}px 'Share Tech Mono', monospace`;
  ctx.fillStyle   = '#ffffff';
  ctx.fillText(`#${id}`, toX + half - 22 * scale, toY + half - 4 * scale);
  ctx.restore();
}

// ── UTILITIES ─────────────────────────────────────────────────────────
function setCanvasSize(w, h) {
  canvas.width  = w;
  canvas.height = h;
}

function stopAll() {
  cancelAnimationFrame(animId);
  animId = null;
  frameCount = 0;
  features   = [];
  imgT       = 0;
  if (matPrevGray && !matPrevGray.isDeleted && !matPrevGray.isDeleted()) {
    try { matPrevGray.delete(); } catch(e) {}
    matPrevGray = null;
  }
  if (matCurrGray && !matCurrGray.isDeleted && !matCurrGray.isDeleted()) {
    try { matCurrGray.delete(); } catch(e) {}
    matCurrGray = null;
  }
  srcVideo.pause();
  srcVideo.src = '';
  srcImage.src = '';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  frameCounter.textContent   = 'FRAME: —';
  featureCounter.textContent = 'FEATURES: —';
}
