// Absolute URL for the self-hosted model downloaded during CI build.
// Using import.meta.url ensures correct resolution regardless of how
// the MediaPipe library internally resolves relative paths.
const _selfHostedModel = new URL('./models/face_detector.tflite', import.meta.url).href;
const _selfHostedScrfdModel = new URL('./models/scrfd_500m_bnkps.onnx', import.meta.url).href;

export const DETECTION_MODEL_ASSET_PATHS = [
  _selfHostedModel,
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector_short_range/float16/1/face_detector_short_range.task',
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector_short_range/float16/latest/face_detector_short_range.task',
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task',
];
export const DETECTION_MODEL_ASSET_PATH = DETECTION_MODEL_ASSET_PATHS[0];
export const DEFAULT_DETECTION_MODEL_ID = 'mediapipe';
export const DETECTION_MODEL_OPTIONS = [
  {
    id: 'mediapipe',
    label: 'MediaPipe (BlazeFace)',
    runtime: 'mediapipe',
    assetPaths: DETECTION_MODEL_ASSET_PATHS,
  },
  {
    id: 'scrfd500',
    label: 'SCRFD-500M (ONNX)',
    runtime: 'onnx',
    // Only the locally-hosted model is supported; no publicly-accessible
    // remote URL is currently available for this model.
    assetPaths: [
      _selfHostedScrfdModel,
    ],
  },
];
// Extra scales (3.5, 4) help detect very small faces in small-to-medium images.
export const DETECTION_SCALES = [1, 1.5, 2, 2.5, 3, 3.5, 4];
// Lower threshold catches faded, partially occluded, or distant tiny faces.
export const DETECTION_SCORE_THRESHOLD = 0.2;
export const DETECTION_PADDING = 0;
// Tile-based detection constants – used for large images (crowds, high-res photos).
export const DETECTION_TILE_SIZE = 512;
export const DETECTION_TILE_OVERLAP = 0.3;
// Images with any dimension above this threshold also receive a tiled detection pass.
export const DETECTION_TILE_THRESHOLD = 640;
const DETECTION_YIELD_EVERY = 4;

export function createFaceDetectorOptions(
  runningMode = 'IMAGE',
  modelAssetPath = DETECTION_MODEL_ASSET_PATH
) {
  return {
    baseOptions: {
      modelAssetPath,
      delegate: 'CPU',
    },
    runningMode,
  };
}

function getDetectionModelOption(modelId = DEFAULT_DETECTION_MODEL_ID) {
  return DETECTION_MODEL_OPTIONS.find(option => option.id === modelId) ?? DETECTION_MODEL_OPTIONS[0];
}

// ONNX Runtime Web runs inside a Web Worker so model compilation and
// inference never block the browser main thread.

const ONNX_MODEL_FETCH_TIMEOUT_MS = 15000;
const ONNX_INIT_TIMEOUT_MS = 30000;
const ONNX_LOAD_TIMEOUT_MS = 30000;
const ONNX_DETECT_TIMEOUT_MS = 15000;

async function fetchWithTimeout(url, timeoutMs = ONNX_MODEL_FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ── ONNX Runtime Web Worker ───────────────────────────────────────────────────
let _ortWorker = null;
let _ortWorkerReady = null;  // Promise that resolves when worker is initialised
let _ortWorkerOpId = 0;
const _ortWorkerPending = new Map(); // id → { resolve, reject, timer }
const _onnxModelPromiseCache = new Map();

function _emitOnnxProgress(phase, detail = {}) {
  try {
    if (typeof window === 'undefined') return;
    window.dispatchEvent(new CustomEvent('onnx-progress', {
      detail: { phase, ...detail, ts: Date.now() },
    }));
  } catch {}
}

function _ensureOrtWorker() {
  if (_ortWorkerReady) return _ortWorkerReady;

  _ortWorker = new Worker(
    new URL('./onnx-detection-worker.js', import.meta.url),
    { type: 'classic' }
  );

  _ortWorker.addEventListener('message', e => {
    if (e.data?.type === 'progress') {
      _emitOnnxProgress(e.data.phase, e.data);
      return;
    }
    const entry = _ortWorkerPending.get(e.data.id);
    if (!entry) return;
    _ortWorkerPending.delete(e.data.id);
    if (entry.timer) clearTimeout(entry.timer);
    if (e.data.type === 'error') entry.reject(new Error(e.data.error));
    else entry.resolve(e.data);
  });

  _ortWorker.addEventListener('error', e => {
    const err = new Error('ONNX Worker crashed: ' + (e.message || 'unknown'));
    for (const { reject, timer } of _ortWorkerPending.values()) {
      if (timer) clearTimeout(timer);
      reject(err);
    }
    _ortWorkerPending.clear();
    _ortWorker = null;
    _ortWorkerReady = null;
  });

  const initId = ++_ortWorkerOpId;
  _ortWorkerReady = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      _ortWorkerPending.delete(initId);
      reject(new Error('ONNX worker initialisation timed out.'));
    }, ONNX_INIT_TIMEOUT_MS);
    _ortWorkerPending.set(initId, { resolve, reject, timer });
    _ortWorker.postMessage({ type: 'init', id: initId });
  }).catch(err => {
    _ortWorker?.terminate();
    _ortWorker = null;
    _ortWorkerReady = null;
    throw err;
  });

  return _ortWorkerReady;
}

async function _ortWorkerCall(msg, transfer = [], timeoutMs = ONNX_LOAD_TIMEOUT_MS) {
  await _ensureOrtWorker();
  return new Promise((resolve, reject) => {
    const id = ++_ortWorkerOpId;
    const timer = setTimeout(() => {
      _ortWorkerPending.delete(id);
      reject(new Error(`ONNX worker ${msg.type} timed out.`));
    }, timeoutMs);
    _ortWorkerPending.set(id, { resolve, reject, timer });
    _ortWorker.postMessage({ ...msg, id }, transfer);
  });
}

function _createOnnxDetectorWrapper(modelKey) {
  return {
    async detect(input) {
      const { width, height } = input;
      let msg;
      if (typeof createImageBitmap === 'function') {
        // An ImageBitmap snapshot stays GPU-side and is transferred to the
        // worker, which draws it straight into the model-sized canvas. This
        // avoids a full-resolution getImageData() readback + copy on the
        // main thread (up to ~37 MB for a 3072px detection canvas).
        const imageBitmap = await createImageBitmap(input);
        msg = [{ type: 'detect', modelKey, imageBitmap, width, height }, [imageBitmap]];
      } else {
        const buf = input.getContext('2d').getImageData(0, 0, width, height).data.buffer;
        msg = [{ type: 'detect', modelKey, imageDataBuffer: buf, width, height }, [buf]];
      }
      const result = await _ortWorkerCall(msg[0], msg[1], ONNX_DETECT_TIMEOUT_MS);
      return { detections: result.detections };
    },
    close() {
      // Worker is a shared singleton – do not terminate it here.
    },
  };
}

async function _loadOnnxFaceDetectorViaWorker(modelAssetPaths) {
  let lastError = null;
  for (const modelAssetPath of modelAssetPaths) {
    try {
      const response = await fetchWithTimeout(modelAssetPath);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const modelData = await response.arrayBuffer();

      // Guard: reject Git LFS pointer files.
      if (modelData.byteLength < 1024) {
        const head = new TextDecoder().decode(modelData);
        if (head.includes('git-lfs.github.com/spec/v1')) {
          throw new Error('Received Git LFS pointer instead of ONNX binary.');
        }
      }

      const modelFileName = modelAssetPath.split('/').pop().split('?')[0];
      const modelKey = modelFileName;
      const transferBuf = modelData.slice(0);
      await _ortWorkerCall(
        // Keep worker-side threshold near-zero so the UI detection threshold
        // (applied in detectFaces) remains the single source of truth.
        { type: 'load-model', modelKey, modelData: transferBuf, modelFileName, scoreThreshold: 0.01 },
        [transferBuf],
        ONNX_LOAD_TIMEOUT_MS
      );
      // Mirror the loaded model filename to the main thread so integration
      // tests can inspect which model was loaded from the page context
      // (the Worker uses its own isolated global scope).
      if (typeof globalThis !== 'undefined') {
        (globalThis.__onnxModelLoads ??= []).push(modelFileName);
      }
      return [_createOnnxDetectorWrapper(modelKey)];
    } catch (err) {
      _emitOnnxProgress('model-load-error', {
        modelAssetPath,
        error: err?.message || String(err),
      });
      lastError = err;
    }
  }
  throw new Error(
    `Failed to load ONNX detector. Last error: ${lastError?.message ?? 'Unknown error'}`
  );
}

function _modelCacheKey(modelAssetPaths) {
  return modelAssetPaths.join('|');
}

// On localhost (development), external URLs are rewritten to go through the
// local static-server proxy so that CORS-restricted hosts (e.g. GitHub raw)
// can still be fetched.  In production users are expected to host model files
// under www/models/ and avoid CORS-restricted remote URLs.
function _resolveModelPaths(paths) {
  try {
    if (typeof window === 'undefined' || !window.location) return paths;
    const { hostname, origin } = window.location;
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') return paths;
    return paths.map(url => {
      try {
        const parsed = new URL(url);
        // Only rewrite external (off-origin) HTTPS URLs.
        if (parsed.protocol === 'https:' &&
            parsed.hostname !== hostname) {
          return `${origin}/proxy-model?url=${encodeURIComponent(url)}`;
        }
      } catch { /* not a URL, leave as-is */ }
      return url;
    });
  } catch {
    return paths;
  }
}

async function loadOnnxFaceDetector(modelAssetPaths) {
  const effectivePaths = _resolveModelPaths(modelAssetPaths);
  const cacheKey = _modelCacheKey(effectivePaths);
  const cached = _onnxModelPromiseCache.get(cacheKey);
  if (cached) return cached;

  const loadPromise = _loadOnnxFaceDetectorViaWorker(effectivePaths);
  const tracked = loadPromise.catch(err => {
    _onnxModelPromiseCache.delete(cacheKey);
    throw err;
  });
  _onnxModelPromiseCache.set(cacheKey, tracked);
  return tracked;
}

export function preloadOnnxFaceDetectors() {
  for (const option of DETECTION_MODEL_OPTIONS) {
    if (option.runtime !== 'onnx') continue;
    void loadOnnxFaceDetector(option.assetPaths).catch(() => {});
  }
}

export async function loadFaceDetectors(
  FaceDetector,
  vision,
  runningMode = 'IMAGE',
  modelId = DEFAULT_DETECTION_MODEL_ID
) {
  const selectedModel = getDetectionModelOption(modelId);
  if (selectedModel.runtime === 'onnx') {
    return loadOnnxFaceDetector(selectedModel.assetPaths);
  }

  let lastError = null;

  for (const modelAssetPath of selectedModel.assetPaths) {
    try {
      return [
        await FaceDetector.createFromOptions(
          vision,
          createFaceDetectorOptions(runningMode, modelAssetPath)
        ),
      ];
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(
    `Failed to fetch model from known URLs. Last error: ${lastError?.message ?? 'Unknown error'}`
  );
}

// Maximum canvas dimension for scale-based detection passes.  Keeping this
// at 3072 prevents allocating enormous canvases for high-resolution source
// images while still providing stronger upscaling for tiny-face detection.
export const DETECTION_MAX_CANVAS_DIM = 3072;

// Final detection-canvas size for a scale pass, after the DETECTION_MAX_CANVAS_DIM cap.
export function detectionCanvasSize(width, height, scale = 1) {
  const scaledW = Math.max(1, Math.round(width * scale));
  const scaledH = Math.max(1, Math.round(height * scale));
  const factor = Math.min(1, DETECTION_MAX_CANVAS_DIM / Math.max(scaledW, scaledH));
  return {
    width: Math.max(1, Math.round(scaledW * factor)),
    height: Math.max(1, Math.round(scaledH * factor)),
  };
}

export function createDetectionCanvas(bitmap, width, height, scale = 1) {
  const { width: finalW, height: finalH } = detectionCanvasSize(width, height, scale);
  const detectionCanvas = document.createElement('canvas');
  detectionCanvas.width = finalW;
  detectionCanvas.height = finalH;
  detectionCanvas.getContext('2d').drawImage(bitmap, 0, 0, finalW, finalH);
  return detectionCanvas;
}

// Creates a canvas for a single tile extracted from the source bitmap.
// tx/ty are the tile origin in the original image coordinate space.
export function createDetectionTileCanvas(bitmap, tx, ty, tw, th) {
  const tileCanvas = document.createElement('canvas');
  tileCanvas.width = tw;
  tileCanvas.height = th;
  tileCanvas.getContext('2d').drawImage(bitmap, tx, ty, tw, th, 0, 0, tw, th);
  return tileCanvas;
}
// Tile origins along one axis. Tiles advance by `step`; the final tile is
// aligned to the far edge so every tile is full-size and no thin sliver tile
// (fully covered by its neighbour, and too small to detect anything) is run.
export function tileOrigins(length, tileSize, step) {
  if (length <= tileSize) return [0];
  const origins = [];
  for (let pos = 0; ; pos += step) {
    if (pos + tileSize >= length) {
      const last = length - tileSize;
      if (origins[origins.length - 1] !== last) origins.push(last);
      return origins;
    }
    origins.push(pos);
  }
}

export function expandBox(box, imgW, imgH, padding = DETECTION_PADDING) {
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

function clampBox(box, imgW, imgH) {
  const x = Math.max(0, Math.min(imgW, box.x));
  const y = Math.max(0, Math.min(imgH, box.y));
  const right = Math.max(x, Math.min(imgW, box.x + box.width));
  const bottom = Math.max(y, Math.min(imgH, box.y + box.height));
  return {
    x,
    y,
    width: right - x,
    height: bottom - y,
  };
}

function yieldToEventLoop() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

function intersectionArea(a, b) {
  const overlapW = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const overlapH = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return overlapW * overlapH;
}

export function intersectionOverUnion(a, b) {
  const intersection = intersectionArea(a, b);
  if (intersection === 0) return 0;

  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

// Fraction of the smaller box that lies inside the other one: 1 when one box
// is fully nested in the other, however different their sizes are.
export function intersectionOverSmaller(a, b) {
  const intersection = intersectionArea(a, b);
  if (intersection === 0) return 0;

  const smaller = Math.min(a.width * a.height, b.width * b.height);
  return smaller > 0 ? intersection / smaller : 0;
}

export const MERGE_IOU_THRESHOLD = 0.3;
// Detections of one face from different scales/tiles are often nested (a tight
// box inside a looser one) and then have a low IoU, so containment is checked
// as well.
export const MERGE_CONTAINMENT_THRESHOLD = 0.6;

// Whether two detections should be treated as the same face.
export function isSameFace(a, b) {
  return intersectionOverUnion(a, b) >= MERGE_IOU_THRESHOLD
    || intersectionOverSmaller(a, b) >= MERGE_CONTAINMENT_THRESHOLD;
}

export function mergeBoxes(boxes, imgW, imgH, padding = DETECTION_PADDING) {
  const merged = [];

  for (const box of boxes) {
    let current = box;
    let didMerge = true;

    while (didMerge) {
      didMerge = false;
      for (let i = 0; i < merged.length; i++) {
        const other = merged[i];
        if (isSameFace(current, other)) {
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

    merged.push(expandBox(current, imgW, imgH, padding));
  }

  return merged;
}

export async function detectFaces({
  bitmap,
  imgW,
  imgH,
  faceDetectors,
  scales = DETECTION_SCALES,
  scoreThreshold = DETECTION_SCORE_THRESHOLD,
  padding = DETECTION_PADDING,
  createCanvas = createDetectionCanvas,
  createTileCanvas = createDetectionTileCanvas,
  tileSize = DETECTION_TILE_SIZE,
  tileOverlap = DETECTION_TILE_OVERLAP,
  tileThreshold = DETECTION_TILE_THRESHOLD,
}) {
  if (!faceDetectors || faceDetectors.length === 0) return [];

  const detections = [];
  let workUnits = 0;

  // ── Scale-based full-image detection ─────────────────────────────────────
  // Scales that end up at the same canvas size after the max-dimension cap
  // (e.g. 2.5×, 3×, 4× of a 1500px photo all become 3072px) would repeat the
  // exact same inference, so only the first of each size is run.
  const seenSizes = new Set();
  const uniqueScales = scales.filter(scale => {
    const { width, height } = detectionCanvasSize(imgW, imgH, scale);
    const key = `${width}x${height}`;
    if (seenSizes.has(key)) return false;
    seenSizes.add(key);
    return true;
  });
  for (const detector of faceDetectors) {
    for (const scale of uniqueScales) {
      const input = createCanvas(bitmap, imgW, imgH, scale);
      const result = await detector.detect(input);
      workUnits++;
      if (workUnits % DETECTION_YIELD_EVERY === 0) {
        // Detection calls can be computationally heavy; yield regularly so the
        // browser can paint and process input while detection runs.
        await yieldToEventLoop();
      }
      for (const det of result.detections || []) {
        const score = det.categories?.[0]?.score ?? 1;
        const bb = det.boundingBox;
        if (!bb || score < scoreThreshold) continue;
        // Coordinates are in the (possibly capped) canvas space; to recover
        // original-image coordinates we divide by the effective scale, which
        // is (canvas.width / imgW) rather than the nominal `scale` value when
        // the canvas was capped.
        const effectiveScaleX = input.width / imgW;
        const effectiveScaleY = input.height / imgH;
        const clamped = clampBox({
          x: Math.round(bb.originX / effectiveScaleX),
          y: Math.round(bb.originY / effectiveScaleY),
          width: Math.round(bb.width / effectiveScaleX),
          height: Math.round(bb.height / effectiveScaleY),
        }, imgW, imgH);
        if (clamped.width <= 0 || clamped.height <= 0) continue;
        detections.push(clamped);
      }
    }
  }

  // ── Tile-based detection for large images (crowds / small distant faces) ──
  // Divides the image into overlapping tiles so that faces which are tiny
  // relative to the full image appear at a useful resolution for the model.
  if (tileSize > 0 && (imgW > tileThreshold || imgH > tileThreshold)) {
    const step = Math.max(1, Math.round(tileSize * (1 - tileOverlap)));
    const xs = tileOrigins(imgW, tileSize, step);
    const ys = tileOrigins(imgH, tileSize, step);
    for (const ty of ys) {
      for (const tx of xs) {
        const tw = Math.min(tileSize, imgW - tx);
        const th = Math.min(tileSize, imgH - ty);
        const input = createTileCanvas(bitmap, tx, ty, tw, th);
        for (const detector of faceDetectors) {
          const result = await detector.detect(input);
          workUnits++;
          if (workUnits % DETECTION_YIELD_EVERY === 0) {
            await yieldToEventLoop();
          }
          for (const det of result.detections || []) {
            const score = det.categories?.[0]?.score ?? 1;
            const bb = det.boundingBox;
            if (!bb || score < scoreThreshold) continue;
            const clamped = clampBox({
              x: Math.round(tx + bb.originX),
              y: Math.round(ty + bb.originY),
              width: Math.round(bb.width),
              height: Math.round(bb.height),
            }, imgW, imgH);
            if (clamped.width <= 0 || clamped.height <= 0) continue;
            detections.push(clamped);
          }
        }
      }
    }
  }

  return mergeBoxes(detections, imgW, imgH, padding).filter(box => box.width > 0 && box.height > 0);
}
