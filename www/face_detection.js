// Absolute URL for the self-hosted model downloaded during CI build.
// Using import.meta.url ensures correct resolution regardless of how
// the MediaPipe library internally resolves relative paths.
const _selfHostedModel = new URL('./models/face_detector.tflite', import.meta.url).href;
const _selfHostedYuNetModel = new URL('./models/face_detection_yunet_2023mar.onnx', import.meta.url).href;
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
    id: 'yunet',
    label: 'YuNet (OpenCV)',
    runtime: 'opencv',
    assetPaths: [
      _selfHostedYuNetModel,
      'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    ],
  },
  {
    id: 'scrfd500',
    label: 'SCRFD-500M (OpenCV, experimental)',
    runtime: 'opencv',
    assetPaths: [
      _selfHostedScrfdModel,
      'https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx',
    ],
  },
];
// Extra scales (3.5, 4) help detect very small faces in small-to-medium images.
export const DETECTION_SCALES = [1, 1.5, 2, 2.5, 3, 3.5, 4];
// Lower threshold catches faded, partially occluded, or distant tiny faces.
export const DETECTION_SCORE_THRESHOLD = 0.2;
export const DETECTION_PADDING = 0.18;
// Tile-based detection constants – used for large images (crowds, high-res photos).
export const DETECTION_TILE_SIZE = 512;
export const DETECTION_TILE_OVERLAP = 0.3;
// Images with any dimension above this threshold also receive a tiled detection pass.
export const DETECTION_TILE_THRESHOLD = 640;

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

let _openCvPromise = null;
async function loadOpenCv() {
  if (typeof globalThis.window === 'undefined') {
    throw new Error('OpenCV runtime is only available in browser environments.');
  }
  if (globalThis.cv?.FaceDetectorYN) return globalThis.cv;
  if (_openCvPromise) return _openCvPromise;

  _openCvPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.async = true;
    script.src = 'https://docs.opencv.org/4.10.0/opencv.js';
    script.onload = () => {
      if (globalThis.cv?.FaceDetectorYN) {
        resolve(globalThis.cv);
        return;
      }
      if (!globalThis.cv) {
        reject(new Error('OpenCV script loaded but `cv` is unavailable.'));
        return;
      }
      const previous = globalThis.cv.onRuntimeInitialized;
      globalThis.cv.onRuntimeInitialized = () => {
        previous?.();
        resolve(globalThis.cv);
      };
    };
    script.onerror = () => reject(new Error('Failed to load OpenCV runtime script.'));
    document.head.appendChild(script);
  }).catch(error => {
    _openCvPromise = null;
    throw error;
  });

  return _openCvPromise;
}

function createOpenCvDetectorWrapper(cv, detector) {
  return {
    async detect(input) {
      const inputCtx = input.getContext('2d');
      const imageData = inputCtx.getImageData(0, 0, input.width, input.height);
      const rgba = cv.matFromImageData(imageData);
      const bgr = new cv.Mat();
      const faces = new cv.Mat();
      try {
        cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
        detector.setInputSize(new cv.Size(input.width, input.height));
        detector.detect(bgr, faces);

        const detections = [];
        if (faces.rows > 0 && faces.cols >= 4) {
          const data = faces.data32F;
          const stride = faces.cols;
          for (let row = 0; row < faces.rows; row++) {
            const base = row * stride;
            detections.push({
              categories: [{ score: data[base + 14] ?? 1 }],
              boundingBox: {
                originX: data[base],
                originY: data[base + 1],
                width: data[base + 2],
                height: data[base + 3],
              },
            });
          }
        }
        return { detections };
      } finally {
        faces.delete();
        bgr.delete();
        rgba.delete();
      }
    },
    close() {
      detector.delete();
    },
  };
}

async function loadOpenCvFaceDetector(modelAssetPaths) {
  const cv = await loadOpenCv();
  let lastError = null;
  for (const modelAssetPath of modelAssetPaths) {
    try {
      const detector = cv.FaceDetectorYN.create(
        modelAssetPath,
        '',
        new cv.Size(320, 320),
        DETECTION_SCORE_THRESHOLD,
        0.3,
        5000
      );
      return [createOpenCvDetectorWrapper(cv, detector)];
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(
    `Failed to load OpenCV detector model from known URLs. Last error: ${lastError?.message ?? 'Unknown error'}`
  );
}

export async function loadFaceDetectors(
  FaceDetector,
  vision,
  runningMode = 'IMAGE',
  modelId = DEFAULT_DETECTION_MODEL_ID
) {
  const selectedModel = getDetectionModelOption(modelId);
  if (selectedModel.runtime === 'opencv') {
    return loadOpenCvFaceDetector(selectedModel.assetPaths);
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

export function createDetectionCanvas(bitmap, width, height, scale = 1) {
  const scaledW = Math.max(1, Math.round(width * scale));
  const scaledH = Math.max(1, Math.round(height * scale));
  const factor = Math.min(1, DETECTION_MAX_CANVAS_DIM / Math.max(scaledW, scaledH));
  const finalW = Math.max(1, Math.round(scaledW * factor));
  const finalH = Math.max(1, Math.round(scaledH * factor));
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

export function intersectionOverUnion(a, b) {
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

export function mergeBoxes(boxes, imgW, imgH, padding = DETECTION_PADDING) {
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

  // ── Scale-based full-image detection ─────────────────────────────────────
  for (const detector of faceDetectors) {
    for (const scale of scales) {
      const input = createCanvas(bitmap, imgW, imgH, scale);
      const result = await detector.detect(input);
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
    for (let ty = 0; ty < imgH; ty += step) {
      for (let tx = 0; tx < imgW; tx += step) {
        const tw = Math.min(tileSize, imgW - tx);
        const th = Math.min(tileSize, imgH - ty);
        const input = createTileCanvas(bitmap, tx, ty, tw, th);
        for (const detector of faceDetectors) {
          const result = await detector.detect(input);
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
