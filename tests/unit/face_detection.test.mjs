import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DETECTION_MODEL_ASSET_PATH,
  DETECTION_MODEL_ASSET_PATHS,
  DETECTION_SCORE_THRESHOLD,
  DETECTION_SCALES,
  DETECTION_TILE_SIZE,
  DETECTION_TILE_THRESHOLD,
  DETECTION_MAX_CANVAS_DIM,
  createFaceDetectorOptions,
  detectFaces,
  loadFaceDetectors,
  mergeBoxes,
} from '../../www/face_detection.js';

test('createFaceDetectorOptions uses the MediaPipe task model', () => {
  const options = createFaceDetectorOptions('IMAGE');

  assert.equal(options.runningMode, 'IMAGE');
  assert.equal(options.baseOptions.delegate, 'CPU');
  assert.equal(options.baseOptions.modelAssetPath, DETECTION_MODEL_ASSET_PATH);
  assert.match(options.baseOptions.modelAssetPath, /face_detector.*\.(task|tflite)$/);
});

test('createFaceDetectorOptions allows overriding model URL', () => {
  const options = createFaceDetectorOptions('VIDEO', 'https://example.com/custom.task');
  assert.equal(options.runningMode, 'VIDEO');
  assert.equal(options.baseOptions.modelAssetPath, 'https://example.com/custom.task');
});

test('loadFaceDetectors falls back to next model URL when first URL fails', async () => {
  const calls = [];
  const detectorInstance = { detect: async () => ({ detections: [] }) };
  const FaceDetector = {
    async createFromOptions(_vision, options) {
      calls.push(options.baseOptions.modelAssetPath);
      if (calls.length === 1) {
        throw new Error('404');
      }
      return detectorInstance;
    },
  };

  const detectors = await loadFaceDetectors(FaceDetector, {}, 'IMAGE');
  assert.equal(detectors.length, 1);
  assert.equal(detectors[0], detectorInstance);
  assert.deepEqual(calls, DETECTION_MODEL_ASSET_PATHS.slice(0, 2));
});

test('mergeBoxes merges overlaps and clamps expanded bounds', () => {
  const merged = mergeBoxes(
    [
      { x: 10, y: 10, width: 20, height: 20 },
      { x: 18, y: 14, width: 18, height: 18 },
      { x: 92, y: 92, width: 12, height: 12 },
    ],
    100,
    100
  );

  assert.equal(merged.length, 2);
  assert.deepEqual(merged[0], { x: 1, y: 1, width: 40, height: 38 });
  assert.deepEqual(merged[1], { x: 90, y: 90, width: 10, height: 10 });
});

test('detectFaces normalizes scaled detections and filters low-confidence hits', async () => {
  const calls = [];
  const fakeDetector = {
    async detect(canvas) {
      calls.push({ width: canvas.width, height: canvas.height });
      if (canvas.width === 100) {
        return {
          detections: [
            {
              categories: [{ score: 0.9 }],
              boundingBox: { originX: 10, originY: 20, width: 30, height: 20 },
            },
            {
              categories: [{ score: 0.19 }],
              boundingBox: { originX: 1, originY: 1, width: 5, height: 5 },
            },
          ],
        };
      }

      return {
        detections: [
          {
            categories: [{ score: 0.85 }],
            boundingBox: { originX: 15, originY: 30, width: 45, height: 30 },
          },
        ],
      };
    },
  };

  const boxes = await detectFaces({
    bitmap: {},
    imgW: 100,
    imgH: 80,
    faceDetectors: [fakeDetector],
    scales: [1, 1.5],
    createCanvas(_bitmap, width, height, scale) {
      return {
        width: Math.round(width * scale),
        height: Math.round(height * scale),
      };
    },
  });

  assert.deepEqual(calls, [
    { width: 100, height: 80 },
    { width: 150, height: 120 },
  ]);
  assert.deepEqual(boxes, [{ x: 0, y: 11, width: 52, height: 38 }]);
});

test('detection constants have expected sensitivity and range', () => {
  assert.ok(DETECTION_SCORE_THRESHOLD <= 0.25, 'threshold should be low enough to catch distant tiny faces');
  assert.ok(DETECTION_SCALES.length >= 6, 'should have several scale steps for tiny-face recovery');
  assert.ok(Math.max(...DETECTION_SCALES) >= 4, 'max scale should be at least 4x for very small faces');
  assert.ok(DETECTION_TILE_SIZE <= 640, 'tile size should stay compact enough to enlarge tiny faces');
  assert.ok(DETECTION_TILE_THRESHOLD <= 800, 'tiling should activate for medium-to-large images');
  assert.ok(DETECTION_MAX_CANVAS_DIM >= 2048, 'canvas cap should allow strong upscaling');
});

test('detectFaces runs tile-based detection for large images', async () => {
  const tileCalls = [];
  const fakeDetector = {
    async detect(input) {
      return {
        detections: [
          {
            categories: [{ score: 0.9 }],
            boundingBox: { originX: 10, originY: 10, width: 20, height: 20 },
          },
        ],
      };
    },
  };

  const boxes = await detectFaces({
    bitmap: {},
    imgW: 1000,
    imgH: 900,
    faceDetectors: [fakeDetector],
    scales: [1],
    createCanvas(_bitmap, width, height, _scale) {
      return { width, height };
    },
    createTileCanvas(_bitmap, tx, ty, tw, th) {
      tileCalls.push({ tx, ty, tw, th });
      return { width: tw, height: th };
    },
    tileSize: 640,
    tileThreshold: 800,
  });

  // Tiling should have been triggered because 1000 > 800.
  assert.ok(tileCalls.length > 0, 'tile detection should run for large images');
  // All returned boxes must be within image bounds.
  for (const box of boxes) {
    assert.ok(box.x >= 0 && box.y >= 0, 'box origin must be non-negative');
    assert.ok(box.x + box.width <= 1000, 'box must not exceed image width');
    assert.ok(box.y + box.height <= 900, 'box must not exceed image height');
  }
});

test('detectFaces skips tile-based detection for small images', async () => {
  const tileCalls = [];
  const fakeDetector = {
    async detect() {
      return { detections: [] };
    },
  };

  await detectFaces({
    bitmap: {},
    imgW: 400,
    imgH: 300,
    faceDetectors: [fakeDetector],
    scales: [1],
    createCanvas(_bitmap, width, height) {
      return { width, height };
    },
    createTileCanvas(_bitmap, tx, ty, tw, th) {
      tileCalls.push({ tx, ty, tw, th });
      return { width: tw, height: th };
    },
    tileSize: 640,
    tileThreshold: 800,
  });

  assert.equal(tileCalls.length, 0, 'tile detection should not run for small images');
});
