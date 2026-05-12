/**
 * MilkCoffee – client-side face anonymisation
 *
 * Memory design (zero-copy, single buffer):
 *   1. Allocate pixel buffer in WASM memory via alloc()
 *   2. Create a Uint8ClampedArray VIEW into wasm.memory.buffer (no copy)
 *   3. Copy decoded image pixels into that view ONCE (the one allowed copy)
 *   4. Call WASM process() – modifies buffer in-place
 *   5. Re-read the SAME buffer position to render (no additional copy)
 */

import initWasm, { alloc, dealloc, process as wasmProcess }
  from './pkg/milkcoffee_wasm.js';
import { detectFaces, loadFaceDetectors } from './face_detection.js';
import { DETECTION_PRESETS, validateScales, validateNumber } from './detection_settings.js';

// ─── DOM refs ────────────────────────────────────────────────────────────────
const fileInput    = document.getElementById('file-input');
const methodSel    = document.getElementById('method');
const strengthSldr = document.getElementById('strength');
const strengthVal  = document.getElementById('strength-val');
const btnProcess   = document.getElementById('btn-process');
const btnDownload  = document.getElementById('btn-download');
const statusEl     = document.getElementById('status');
const canvas       = document.getElementById('preview');
const placeholder  = document.getElementById('canvas-placeholder');
const overlay      = document.getElementById('loading-overlay');
const loadingText  = document.getElementById('loading-text');

const ctx = canvas.getContext('2d');
const METHOD_NAMES = ['mosaic', 'blur', 'solid', 'cyber veil', 'neon blocks'];

// ─── Detection settings DOM refs ─────────────────────────────────────────────
const detScalesInput     = document.getElementById('det-scales');
const detScoreThreshInput = document.getElementById('det-score-threshold');
const detPaddingInput    = document.getElementById('det-padding');
const detTileSizeInput   = document.getElementById('det-tile-size');
const detTileOverlapInput = document.getElementById('det-tile-overlap');
const detTileThreshInput = document.getElementById('det-tile-threshold');

// IDs of all detection setting inputs – used for bulk error clearing.
const DETECTION_INPUT_IDS = [
  'det-scales', 'det-score-threshold', 'det-padding',
  'det-tile-size', 'det-tile-overlap', 'det-tile-threshold',
];

// ─── Detection presets ────────────────────────────────────────────────────────

function applyPreset(presetName) {
  const preset = DETECTION_PRESETS[presetName];
  if (!preset) return;
  detScalesInput.value       = preset.scales.join(', ');
  detScoreThreshInput.value  = preset.scoreThreshold;
  detPaddingInput.value      = preset.padding;
  detTileSizeInput.value     = preset.tileSize;
  detTileOverlapInput.value  = preset.tileOverlap;
  detTileThreshInput.value   = preset.tileThreshold;
  // Clear any previous validation errors.
  for (const id of DETECTION_INPUT_IDS) {
    document.getElementById(id).classList.remove('input-error');
    document.getElementById(id + '-err').textContent = '';
  }
}

document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
});

// ─── Detection settings validation (DOM wrappers around pure validators) ─────
function validateScalesField() {
  const result = validateScales(detScalesInput.value);
  const errEl = document.getElementById('det-scales-err');
  if (!result.ok) {
    errEl.textContent = result.error;
    detScalesInput.classList.add('input-error');
    return null;
  }
  errEl.textContent = '';
  detScalesInput.classList.remove('input-error');
  return result.values;
}

function validateNumberField(inputEl, errId, opts) {
  const result = validateNumber(inputEl.value, opts);
  const errEl = document.getElementById(errId);
  if (!result.ok) {
    errEl.textContent = result.error;
    inputEl.classList.add('input-error');
    return null;
  }
  errEl.textContent = '';
  inputEl.classList.remove('input-error');
  return result.value;
}

function readDetectionSettings() {
  const scales        = validateScalesField();
  const scoreThreshold = validateNumberField(detScoreThreshInput, 'det-score-threshold-err',
    { min: 0, max: 1, label: 'Score threshold' });
  const padding       = validateNumberField(detPaddingInput, 'det-padding-err',
    { min: 0, max: 1, label: 'Padding' });
  const tileSize      = validateNumberField(detTileSizeInput, 'det-tile-size-err',
    { min: 0, label: 'Tile size', integer: true });
  const tileOverlap   = validateNumberField(detTileOverlapInput, 'det-tile-overlap-err',
    { min: 0, max: 0.9, label: 'Tile overlap' });
  const tileThreshold = validateNumberField(detTileThreshInput, 'det-tile-threshold-err',
    { min: 0, label: 'Tile threshold', integer: true });

  if (scales === null || scoreThreshold === null || padding === null ||
      tileSize === null || tileOverlap === null || tileThreshold === null) {
    return null;
  }
  return { scales, scoreThreshold, padding, tileSize, tileOverlap, tileThreshold };
}

// Validate on change so users get immediate feedback.
detScalesInput.addEventListener('change', validateScalesField);
[
  [detScoreThreshInput, 'det-score-threshold-err', { min: 0, max: 1, label: 'Score threshold' }],
  [detPaddingInput,     'det-padding-err',          { min: 0, max: 1, label: 'Padding' }],
  [detTileSizeInput,    'det-tile-size-err',         { min: 0, label: 'Tile size', integer: true }],
  [detTileOverlapInput, 'det-tile-overlap-err',      { min: 0, max: 0.9, label: 'Tile overlap' }],
  [detTileThreshInput,  'det-tile-threshold-err',    { min: 0, label: 'Tile threshold', integer: true }],
].forEach(([el, errId, opts]) => {
  el.addEventListener('change', () => validateNumberField(el, errId, opts));
});

// ─── App state ───────────────────────────────────────────────────────────────
let wasmMemory   = null;   // WebAssembly.Memory (exported from WASM module)
let faceDetectors = [];    // MediaPipe FaceDetector instances
let currentImage = null;   // { bitmap, width, height } of the uploaded image
let wasmPtr      = 0;      // current allocation pointer
let wasmSize     = 0;      // current allocation byte size

// ─── Utility helpers ─────────────────────────────────────────────────────────
function setStatus(msg, kind = '') {
  statusEl.textContent = msg;
  statusEl.className = kind;
}

function showOverlay(msg) {
  loadingText.textContent = msg;
  overlay.classList.add('visible');
}

function hideOverlay() {
  overlay.classList.remove('visible');
}

// ─── Strength slider ─────────────────────────────────────────────────────────
strengthSldr.addEventListener('input', () => {
  strengthVal.textContent = (strengthSldr.value / 100).toFixed(2);
});

// ─── Initialisation ───────────────────────────────────────────────────────────
async function init() {
  try {
    // 1. Load WASM module.
    const wasmExports = await initWasm('./pkg/milkcoffee_wasm_bg.wasm');
    wasmMemory = wasmExports.memory;
    setStatus('WASM loaded. Loading face-detection models…');

    // 2. Load MediaPipe Face Detection (CDN).
    await loadFaceDetector();

    setStatus('Ready. Upload an image to begin.', 'ok');
    btnProcess.disabled = false;
  } catch (err) {
    setStatus('Initialisation failed: ' + err.message, 'error');
    console.error(err);
  }
}

// ─── MediaPipe Face Detection ─────────────────────────────────────────────────
// The MediaPipe Tasks Vision library is loaded via dynamic ES-module import.
// Note: the ES module `import()` API does not support the `integrity` attribute
// that <script> tags use for Subresource Integrity checking, so CDN-loaded code
// cannot be SRI-verified through this mechanism. As a mitigation the pinned
// version in the URL (0.10.14) is used to reduce supply-chain drift.
async function loadFaceDetector() {
  const { FaceDetector, FilesetResolver } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
  );

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  faceDetectors = await loadFaceDetectors(FaceDetector, vision, 'IMAGE');
}

// ─── File upload / drop ───────────────────────────────────────────────────────
document.getElementById('upload-btn').addEventListener('click', () => fileInput.click());
let dragDepth = 0;
document.addEventListener('dragenter', e => { e.preventDefault(); dragDepth++; document.body.classList.add('drag-over'); });
document.addEventListener('dragover',  e => { e.preventDefault(); });
document.addEventListener('dragleave', () => {
  if (dragDepth > 0) dragDepth--;
  if (dragDepth === 0) document.body.classList.remove('drag-over');
});
document.addEventListener('drop', e => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('drag-over');
  const file = e.dataTransfer?.files[0];
  if (file) loadFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

async function loadFile(file) {
  if (!file.type.match(/^image\/(jpeg|png)$/)) {
    setStatus('Please upload a JPG or PNG file.', 'error');
    return;
  }

  try {
    showOverlay('Loading image…');
    const bitmap = await createImageBitmap(file);

    const w = bitmap.width;
    const h = bitmap.height;

    currentImage = { bitmap, width: w, height: h };

    // Display original image on canvas.
    canvas.width  = w;
    canvas.height = h;
    ctx.drawImage(bitmap, 0, 0, w, h);
    placeholder.style.display = 'none';
    btnDownload.disabled = true;
    setStatus(`Image loaded (${w}×${h} px). Click "Detect & Anonymise" to process.`);
  } catch (err) {
    setStatus('Failed to load image: ' + err.message, 'error');
  } finally {
    hideOverlay();
  }
}

// ─── Process ──────────────────────────────────────────────────────────────────
btnProcess.addEventListener('click', processImage);

async function processImage() {
  if (!currentImage) {
    setStatus('Upload an image first.', 'error');
    return;
  }

  btnProcess.disabled = true;
  btnDownload.disabled = true;
  showOverlay('Detecting faces…');
  setStatus('Detecting faces…');

  try {
    const { bitmap, width: w, height: h } = currentImage;

    // ── Step 1: Decode image pixels ──────────────────────────────────────────
    // Draw to an offscreen canvas to obtain raw RGBA pixels.
    const offscreen = new OffscreenCanvas(w, h);
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(bitmap, 0, 0, w, h);
    const imageData = offCtx.getImageData(0, 0, w, h);

    // ── Step 2: Face detection (JS side) ─────────────────────────────────────
    const detSettings = readDetectionSettings();
    if (!detSettings) {
      setStatus('Invalid detection settings. Please correct the highlighted fields.', 'error');
      btnProcess.disabled = false;
      hideOverlay();
      return;
    }
    const boxes = await detectFaces({
      bitmap,
      imgW: w,
      imgH: h,
      faceDetectors,
      scales: detSettings.scales,
      scoreThreshold: detSettings.scoreThreshold,
      padding: detSettings.padding,
      tileSize: detSettings.tileSize,
      tileOverlap: detSettings.tileOverlap,
      tileThreshold: detSettings.tileThreshold,
    });
    const boxCount = boxes.length;

    loadingText.textContent = `Found ${boxCount} face(s). Anonymising…`;
    setStatus(`Found ${boxCount} face(s). Applying anonymisation…`);

    // ── Step 3: WASM zero-copy processing ────────────────────────────────────
    const byteSize = w * h * 4;

    // Free any previous allocation.
    if (wasmPtr !== 0) {
      dealloc(wasmPtr, wasmSize);
      wasmPtr  = 0;
      wasmSize = 0;
    }

    // Allocate buffer inside WASM memory.
    wasmPtr  = alloc(byteSize);
    wasmSize = byteSize;

    // ONE allowed copy: decoded pixels → WASM buffer.
    // The view is created immediately before the copy so it always reflects the
    // current wasm.memory.buffer (alloc() may have grown WASM memory).
    new Uint8ClampedArray(wasmMemory.buffer, wasmPtr, byteSize).set(imageData.data);

    // Build face-box JSON for WASM (no DOM/serde dependency in Rust).
    const boxesJson = JSON.stringify(
      boxes.map(b => ({ x: b.x, y: b.y, width: b.width, height: b.height }))
    );

    const method   = parseInt(methodSel.value, 10);
    const strength = strengthSldr.value / 100;

    // In-place processing inside WASM (modifies wasmMemory.buffer at wasmPtr).
    wasmProcess(wasmPtr, w, h, boxesJson, method, strength);

    // ── Step 4: Render from WASM buffer (no additional copy) ─────────────────
    // Always read wasmMemory.buffer fresh: wasmProcess() may have grown WASM
    // memory, which replaces the underlying ArrayBuffer object.
    const resultView = new Uint8ClampedArray(wasmMemory.buffer, wasmPtr, byteSize);
    const result = new ImageData(resultView, w, h);
    canvas.width  = w;
    canvas.height = h;
    ctx.putImageData(result, 0, 0);

    if (boxCount === 0) {
      setStatus('No faces detected – original image displayed unchanged.', 'ok');
    } else {
      setStatus(
        `Done! ${boxCount} face(s) anonymised with ${METHOD_NAMES[method] || 'custom'} method.`,
        'ok'
      );
    }

    btnDownload.disabled = false;
  } catch (err) {
    setStatus('Processing failed: ' + err.message, 'error');
    console.error(err);
  } finally {
    btnProcess.disabled = false;
    hideOverlay();
  }
}

// ─── Download ─────────────────────────────────────────────────────────────────
btnDownload.addEventListener('click', () => {
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'anonymised.png';
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
});

// ─── Start ────────────────────────────────────────────────────────────────────
init();
