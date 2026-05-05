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

// ─── Constants ───────────────────────────────────────────────────────────────
const MAX_WIDTH = 1280;

// ─── DOM refs ────────────────────────────────────────────────────────────────
const dropZone     = document.getElementById('drop-zone');
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

// ─── App state ───────────────────────────────────────────────────────────────
let wasmMemory   = null;   // WebAssembly.Memory (exported from WASM module)
let faceDetector = null;   // MediaPipe FaceDetector instance
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
    setStatus('WASM loaded. Loading face-detection model…');

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

  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_detector/' +
        'blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
      delegate: 'CPU',
    },
    runningMode: 'IMAGE',
  });
}

/**
 * Run face detection on an ImageBitmap and return bounding boxes as
 * [{x, y, width, height}] in absolute pixel coordinates.
 */
async function detectFaces(bitmap, imgW, imgH) {
  if (!faceDetector) return [];

  // MediaPipe needs an HTMLCanvasElement or HTMLImageElement.
  const offscreen = new OffscreenCanvas(imgW, imgH);
  const offCtx = offscreen.getContext('2d');
  offCtx.drawImage(bitmap, 0, 0, imgW, imgH);

  // Convert OffscreenCanvas → ImageData → HTMLCanvasElement
  // (MediaPipe Tasks Vision accepts HTMLCanvasElement directly).
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width  = imgW;
  tmpCanvas.height = imgH;
  const tmpCtx = tmpCanvas.getContext('2d');
  tmpCtx.drawImage(offscreen, 0, 0);

  const result = faceDetector.detect(tmpCanvas);

  return (result.detections || []).map(det => {
    const bb = det.boundingBox;
    return {
      x:      Math.round(bb.originX),
      y:      Math.round(bb.originY),
      width:  Math.round(bb.width),
      height: Math.round(bb.height),
    };
  });
}

// ─── File upload / drop ───────────────────────────────────────────────────────
dropZone.addEventListener('click',    () => fileInput.click());
dropZone.addEventListener('keydown',  e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave',() => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
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

    // Resize if wider than MAX_WIDTH.
    let w = bitmap.width;
    let h = bitmap.height;
    if (w > MAX_WIDTH) {
      h = Math.round(h * MAX_WIDTH / w);
      w = MAX_WIDTH;
    }

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
    const boxes = await detectFaces(bitmap, w, h);
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
        `Done! ${boxCount} face(s) anonymised with ${['mosaic', 'blur', 'solid'][method]} method.`,
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
