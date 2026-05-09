export const DETECTION_MODEL_ASSET_PATHS = [
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector_short_range/float16/1/face_detector_short_range.task',
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector_short_range/float16/latest/face_detector_short_range.task',
  'https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task',
];
export const DETECTION_MODEL_ASSET_PATH = DETECTION_MODEL_ASSET_PATHS[0];
export const DETECTION_SCALES = [1, 1.5, 2];
export const DETECTION_SCORE_THRESHOLD = 0.35;
export const DETECTION_PADDING = 0.18;

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

export function createDetectionCanvas(bitmap, width, height, scale = 1) {
  const scaledW = Math.max(1, Math.round(width * scale));
  const scaledH = Math.max(1, Math.round(height * scale));
  const detectionCanvas = document.createElement('canvas');
  detectionCanvas.width = scaledW;
  detectionCanvas.height = scaledH;
  detectionCanvas.getContext('2d').drawImage(bitmap, 0, 0, scaledW, scaledH);
  return detectionCanvas;
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
}) {
  if (!faceDetectors || faceDetectors.length === 0) return [];

  const detections = [];

  for (const detector of faceDetectors) {
    for (const scale of scales) {
      const input = createCanvas(bitmap, imgW, imgH, scale);
      const result = await detector.detect(input);
      for (const det of result.detections || []) {
        const score = det.categories?.[0]?.score ?? 1;
        const bb = det.boundingBox;
        if (!bb || score < scoreThreshold) continue;
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
