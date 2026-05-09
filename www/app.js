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
const DETECTION_MODELS = [
  {
    label: 'short-range',
    modelAssetPath:
      'https://storage.googleapis.com/mediapipe-models/face_detector/' +
      'blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
  },
  {
    label: 'full-range',
    modelAssetPath:
      'https://storage.googleapis.com/mediapipe-models/face_detector/' +
      'blaze_face_full_range/float16/1/blaze_face_full_range.tflite',
  },
];
const DETECTION_SCALES = [1, 1.5, 2];
const DETECTION_SCORE_THRESHOLD = 0.35;
const DETECTION_PADDING = 0.18;

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

  faceDetectors = await Promise.all(
    DETECTION_MODELS.map(model =>
      FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: model.modelAssetPath,
          delegate: 'CPU',
        },
        runningMode: 'IMAGE',
      })
    )
  );
}

function createDetectionCanvas(bitmap, width, height, scale = 1) {
  const scaledW = Math.max(1, Math.round(width * scale));
  const scaledH = Math.max(1, Math.round(height * scale));
  const detectionCanvas = document.createElement('canvas');
  detectionCanvas.width = scaledW;
  detectionCanvas.height = scaledH;
  detectionCanvas.getContext('2d').drawImage(bitmap, 0, 0, scaledW, scaledH);
  return detectionCanvas;
}

function expandBox(box, imgW, imgH, padding = DETECTION_PADDING) {
  const padX = Math.round(box.width * padding);
  const padY = Math.round(box.height * padding);
  const x = Math.max(0, box.x - padX);
  const y = Math.max(0, box.y - padY);
  const right = Math.min(imgW, box.x + box.width + padX);
  const bottom = Math.min(imgH, box.y + box.height + padY);

  return {
    x,
    y,
    width: Math.max(1, right - x),
    height: Math.max(1, bottom - y),
  };
}

function intersectionOverUnion(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const overlapW = Math.max(0, x2 - x1);
  const overlapH = Math.max(0, y2 - y1);
  const intersection = overlapW * overlapH;

  if (intersection === 0) return 0;

  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function mergeBoxes(boxes, imgW, imgH) {
  const merged = [];

  for (const box of boxes) {
    let current = box;
    let didMerge = true;

    while (didMerge) {
      didMerge = false;
      for (let i = 0; i < merged.length; i++) {
        const other = merged[i];
        if (intersectionOverUnion(current, other) >= 0.3) {
          const x = Math.min(current.x, other.x);
          const y = Math.min(current.y, other.y);
          const right = Math.max(current.x + current.width, other.x + other.width);
          const bottom = Math.max(current.y + current.height, other.y + other.height);
          current = {
            x,
            y,
            width: right - x,
            height: bottom - y,
          };
          merged.splice(i, 1);
          didMerge = true;
          break;
        }
      }
    }

    merged.push(expandBox(current, imgW, imgH));
  }

  return merged;
}

/**
 * Run face detection on an ImageBitmap and return bounding boxes as
 * [{x, y, width, height}] in absolute pixel coordinates.
 */
async function detectFaces(bitmap, imgW, imgH) {
  if (faceDetectors.length === 0) return [];

  const detections = [];

  for (const detector of faceDetectors) {
    for (const scale of DETECTION_SCALES) {
      const result = detector.detect(createDetectionCanvas(bitmap, imgW, imgH, scale));
      for (const det of result.detections || []) {
        const score = det.categories?.[0]?.score ?? 1;
        const bb = det.boundingBox;
        if (!bb || score < DETECTION_SCORE_THRESHOLD) continue;
        detections.push({
          x: Math.round(bb.originX / scale),
          y: Math.round(bb.originY / scale),
          width: Math.round(bb.width / scale),
          height: Math.round(bb.height / scale),
        });
      }
    }
  }

  return mergeBoxes(detections, imgW, imgH).filter(box => box.width > 0 && box.height > 0);
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
