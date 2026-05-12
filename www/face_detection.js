// Absolute URL for the self-hosted model downloaded during CI build.
// Using import.meta.url ensures correct resolution regardless of how
// the MediaPipe library internally resolves relative paths.
const _selfHostedModel = new URL('./models/face_detector.tflite', import.meta.url).href;

export const DETECTION_MODEL_ASSET_PATHS = [
  _selfHostedModel,
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector_short_range/float16/1/face_detector_short_range.task',
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector_short_range/float16/latest/face_detector_short_range.task',
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task',
];
export const DETECTION_MODEL_ASSET_PATH = DETECTION_MODEL_ASSET_PATHS[0];
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

export async function loadFaceDetectors(FaceDetector, vision, runningMode = 'IMAGE') {
  let lastError = null;

  for (const modelAssetPath of DETECTION_MODEL_ASSET_PATHS) {
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

export function mergeBoxes(boxes, imgW, imgH) {
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

export async function detectFaces({
  bitmap,
  imgW,
  imgH,
  faceDetectors,
  scales = DETECTION_SCALES,
  scoreThreshold = DETECTION_SCORE_THRESHOLD,
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

  return mergeBoxes(detections, imgW, imgH).filter(box => box.width > 0 && box.height > 0);
}
