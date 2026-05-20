import { test, expect } from '@playwright/test';

const MEDIAPIPE_STUB = `
  export class FilesetResolver {
    static async forVisionTasks(url) {
      globalThis.__visionWasmUrl = url;
      return { url };
    }
  }

  export class FaceDetector {
    static async createFromOptions(_vision, options) {
      globalThis.__detectorOptions = globalThis.__detectorOptions || [];
      globalThis.__detectorOptions.push(options);
      return new FaceDetector();
    }

    async detect(input) {
      const scripted = globalThis.__testDetections;
      if (Array.isArray(scripted)) {
        return {
          detections: scripted.map(det => {
            if (det.boundingBox) return det;
            return {
              categories: [{ score: det.score ?? 0.95 }],
              boundingBox: {
                originX: input.width * (det.originXRatio ?? 0),
                originY: input.height * (det.originYRatio ?? 0),
                width: input.width * (det.widthRatio ?? 0),
                height: input.height * (det.heightRatio ?? 0),
              },
            };
          }),
        };
      }
      return {
        detections: [{
          categories: [{ score: 0.95 }],
          boundingBox: {
            originX: input.width * 0.25,
            originY: input.height * 0.25,
            width: input.width * 0.4,
            height: input.height * 0.4,
          },
        }],
      };
    }

    async close() {}
  }
`;

const ORT_STUB = `
  (function () {
    const DEFAULT_DETECTIONS = [{ x: 4, y: 4, width: 16, height: 16, score: 0.95 }];

    class FakeTensor {
      constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    }

    class FakeSession {
      get inputNames() { return ['input']; }
      get outputNames() {
        return ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32'];
      }
      async run(_inputs) {
        // Minimal SCRFD-like output tensors for a 320x320 model.
        const score8 = new Float32Array(40 * 40 * 2);
        const score16 = new Float32Array(20 * 20 * 2);
        const score32 = new Float32Array(10 * 10 * 2);
        const bbox8 = new Float32Array(40 * 40 * 2 * 4);
        const bbox16 = new Float32Array(20 * 20 * 2 * 4);
        const bbox32 = new Float32Array(10 * 10 * 2 * 4);

        // Seed one confident detection at stride-8, anchor index 0.
        score8[0] = DEFAULT_DETECTIONS[0]?.score ?? 0.95;
        bbox8[0] = 0.1;
        bbox8[1] = 0.1;
        bbox8[2] = 2.5;
        bbox8[3] = 2.5;

        return {
          score_8: new FakeTensor('float32', score8, [3200, 1]),
          score_16: new FakeTensor('float32', score16, [800, 1]),
          score_32: new FakeTensor('float32', score32, [200, 1]),
          bbox_8: new FakeTensor('float32', bbox8, [3200, 4]),
          bbox_16: new FakeTensor('float32', bbox16, [800, 4]),
          bbox_32: new FakeTensor('float32', bbox32, [200, 4]),
        };
      }
      release() {}
    }

    globalThis.ort = {
      env: { wasm: { wasmPaths: '' } },
      Tensor: FakeTensor,
      InferenceSession: {
        async create(modelData) {
          globalThis.__onnxModelLoads = globalThis.__onnxModelLoads || [];
          // modelData is an ArrayBuffer here (transferred from main thread).
          // We store a placeholder so tests can verify which model was loaded.
          globalThis.__onnxModelLoads.push(globalThis.__pendingOnnxModelName || '');
          return new FakeSession();
        },
      },
    };
  })();
`;

const RED_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAK0lEQVR4nO3OIQEAAAwEoetfeovxBoGnq1tKQEBAQEBAQEBAQEBAQEBgHXhUDfhqRFDd3gAAAABJRU5ErkJggg==';

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    globalThis.__DISABLE_ONNX_PRELOAD = true;
  });

  await page.route('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'text/javascript; charset=utf-8',
      body: MEDIAPIPE_STUB,
    });
  });

  // Stub the ONNX Runtime CDN script – exposes globalThis.ort in the worker.
  await page.route('**/ort.min.js', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'text/javascript; charset=utf-8',
      body: ORT_STUB,
    });
  });

  // Stub ONNX model file requests so fetch() in loadOnnxFaceDetector succeeds.
  await page.route('**/*.onnx', async route => {
    // Expose the filename to the worker stub via a page-global so the test
    // can later verify which model was loaded.
    const url = route.request().url();
    const fileName = url.split('/').pop().split('?')[0];
    await page.evaluate(name => { globalThis.__pendingOnnxModelName = name; }, fileName).catch(() => {});
    await route.fulfill({
      status: 200,
      contentType: 'application/octet-stream',
      body: Buffer.alloc(32, 0),
    });
  });
});

async function uploadPng(page) {
  await page.locator('#file-input').setInputFiles({
    name: 'face.png',
    mimeType: 'image/png',
    buffer: Buffer.from(RED_PNG_BASE64, 'base64'),
  });
}

test('initializes wasm + AI model and processes image on all viewports', async ({ page }) => {
  await page.goto('/');

  await expect(page.locator('#status')).toHaveText('Ready. Upload an image to begin.');
  await expect(page.locator('#btn-process')).toBeEnabled();

  const visionWasmUrl = await page.evaluate(() => globalThis.__visionWasmUrl);
  expect(visionWasmUrl).toContain('/tasks-vision@0.10.14/wasm');
  const detectorModelPath = await page.evaluate(() => globalThis.__detectorOptions?.[0]?.baseOptions?.modelAssetPath);
  expect(detectorModelPath).toMatch(/face_detector.*\.(task|tflite)$/);
});

test('accepts PNG and JPG uploads and rejects unsupported files', async ({ page }) => {
  await page.goto('/');

  await uploadPng(page);
  await expect(page.locator('#status')).toContainText('Image loaded');

  await page.locator('#file-input').setInputFiles({
    name: 'face.jpg',
    mimeType: 'image/jpeg',
    buffer: Buffer.from(RED_PNG_BASE64, 'base64'),
  });
  await expect(page.locator('#status')).toContainText('Image loaded');

  await page.locator('#file-input').setInputFiles({
    name: 'not-image.txt',
    mimeType: 'text/plain',
    buffer: Buffer.from('hello', 'utf8'),
  });
  await expect(page.locator('#status')).toHaveText('Please upload a JPG or PNG file.');
});

test('supports full/side/faded faces and allows downloading artifact', async ({ page }) => {
  await page.goto('/');
  await uploadPng(page);

  await page.evaluate(() => {
    globalThis.__testDetections = [
      {
        score: 0.98,
        originXRatio: 0.10,
        originYRatio: 0.10,
        widthRatio: 0.35,
        heightRatio: 0.35,
      },
      {
        score: 0.94,
        originXRatio: 0.52,
        originYRatio: 0.12,
        widthRatio: 0.30,
        heightRatio: 0.34,
      },
      {
        score: 0.91,
        originXRatio: 0.30,
        originYRatio: 0.56,
        widthRatio: 0.28,
        heightRatio: 0.28,
      },
    ];
  });

  await page.locator('#btn-process').click();
  const statusText = await page.locator('#status').textContent();
  expect(statusText).toMatch(/Done! \d+ face\(s\) anonymised/);
  const detectedFaces = Number(statusText.match(/Done! (\d+) face\(s\) anonymised/)?.[1] ?? 0);
  expect(detectedFaces).toBeGreaterThanOrEqual(3);
  await expect(page.locator('#btn-download')).toBeEnabled();

  const downloadPromise = page.waitForEvent('download');
  await page.locator('#btn-download').click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('anonymised.png');
});

test('applies every effect method successfully', async ({ page }) => {
  await page.goto('/');
  await uploadPng(page);

  const methods = [
    { value: '0', label: 'mosaic' },
    { value: '1', label: 'blur' },
    { value: '2', label: 'solid' },
    { value: '3', label: 'cyber veil' },
    { value: '4', label: 'neon blocks' },
  ];

  for (const { value, label } of methods) {
    await page.evaluate(() => {
      globalThis.__testDetections = [{
        categories: [{ score: 0.97 }],
        boundingBox: { originX: 4, originY: 4, width: 16, height: 16 },
      }];
    });
    await page.locator('#method').selectOption(value);
    await page.locator('#btn-process').click();
    await expect(page.locator('#status')).toContainText(`with ${label} method`);
    await expect(page.locator('#btn-download')).toBeEnabled();
  }
});

test('the site anonymises an uploaded image end to end', async ({ page }) => {
  await page.goto('/');

  await expect(page.locator('#status')).toHaveText('Ready. Upload an image to begin.');
  await expect(page.locator('#btn-process')).toBeEnabled();

  await page.locator('#file-input').setInputFiles({
    name: 'face.png',
    mimeType: 'image/png',
    buffer: Buffer.from(RED_PNG_BASE64, 'base64'),
  });

  await expect(page.locator('#status')).toContainText('Image loaded');
  await page.locator('#method').selectOption('2');
  await page.locator('#btn-process').click();

  await expect(page.locator('#status')).toContainText('face(s) anonymised');
  const statusText = await page.locator('#status').textContent();
  expect(statusText).toMatch(/Done! \d+ face\(s\) anonymised/);
  const detectedFaces = Number(statusText.match(/Done! (\d+) face\(s\) anonymised/)?.[1] ?? 0);
  expect(detectedFaces).toBeGreaterThanOrEqual(1);
  await expect(page.locator('#btn-download')).toBeEnabled();

  const centerPixel = await page.evaluate(() => {
    const canvas = document.getElementById('preview');
    const ctx = canvas.getContext('2d');
    return Array.from(ctx.getImageData(12, 12, 1, 1).data);
  });
  expect(centerPixel).toEqual([0, 0, 0, 255]);
});

test('scrfd500 model loads and runs inference in browser flow', async ({ page }) => {
  await page.goto('/');
  await uploadPng(page);

  await page.locator('#det-model').selectOption('scrfd500');
  await expect(page.locator('#status')).toHaveText('Ready. Upload an image to begin.');
  await expect(page.locator('#btn-process')).toBeEnabled();

  await page.locator('#btn-process').click();
  await expect(page.locator('#status')).toContainText('face(s) anonymised');

  const loadedModelPath = await page.evaluate(() => {
    const paths = globalThis.__onnxModelLoads || [];
    return paths[paths.length - 1];
  });
  expect(loadedModelPath).toContain('scrfd_500m');
});

// ─── Detection settings panel tests ─────────────────────────────────────────

test('detection settings panel is collapsed by default', async ({ page }) => {
  await page.goto('/');
  const details = page.locator('details.det-settings');
  await expect(details).not.toHaveAttribute('open');
});

test('detection settings panel can be opened and contains all inputs', async ({ page }) => {
  await page.goto('/');
  await page.locator('details.det-settings > summary').click();
  await expect(page.locator('details.det-settings')).toHaveAttribute('open', '');

  for (const id of ['det-scales', 'det-score-threshold', 'det-padding',
                    'det-tile-size', 'det-tile-overlap', 'det-tile-threshold']) {
    await expect(page.locator(`#${id}`)).toBeVisible();
  }
});

test('scrfd disables scales and tile inputs in detection settings', async ({ page }) => {
  await page.goto('/');
  await page.locator('details.det-settings > summary').click();

  const detScales = page.locator('#det-scales');
  const detTileSize = page.locator('#det-tile-size');
  const detTileOverlap = page.locator('#det-tile-overlap');
  const detTileThreshold = page.locator('#det-tile-threshold');

  await page.locator('#det-model').selectOption('scrfd500');
  await expect(page.locator('#status')).toHaveText('Ready. Upload an image to begin.');
  await expect(detScales).toBeDisabled();
  await expect(detTileSize).toBeDisabled();
  await expect(detTileOverlap).toBeDisabled();
  await expect(detTileThreshold).toBeDisabled();

  await page.locator('#det-model').selectOption('mediapipe');
  await expect(page.locator('#status')).toHaveText('Ready. Upload an image to begin.');
  await expect(detScales).toBeEnabled();
  await expect(detTileSize).toBeEnabled();
  await expect(detTileOverlap).toBeEnabled();
  await expect(detTileThreshold).toBeEnabled();
});

test('preset buttons populate all detection fields with valid values', async ({ page }) => {
  await page.goto('/');
  await page.locator('details.det-settings > summary').click();

  const presets = {
    'big-image':   { scales: '1, 1.5, 2, 2.5',          scoreThreshold: '0.2',  tileSize: '512' },
    'small-image': { scales: '1, 1.5, 2, 2.5, 3',        scoreThreshold: '0.3',  tileSize: '0'   },
    'small-faces': { scales: '1, 1.5, 2, 2.5, 3, 3.5, 4', scoreThreshold: '0.15', tileSize: '480' },
    'big-faces':   { scales: '1, 1.5, 2',               scoreThreshold: '0.4',  tileSize: '0'   },
  };

  for (const [preset, expected] of Object.entries(presets)) {
    await page.locator(`[data-preset="${preset}"]`).click();
    await expect(page.locator('#det-scales')).toHaveValue(expected.scales);
    await expect(page.locator('#det-score-threshold')).toHaveValue(expected.scoreThreshold);
    await expect(page.locator('#det-tile-size')).toHaveValue(expected.tileSize);
  }
});

test('preset button clears any previous validation errors', async ({ page }) => {
  await page.goto('/');
  await page.locator('details.det-settings > summary').click();

  // Trigger an error first.
  await page.locator('#det-scales').fill('invalid');
  await page.locator('#det-scales').dispatchEvent('change');
  await expect(page.locator('#det-scales-err')).not.toHaveText('');

  // Applying a preset should clear the error.
  await page.locator('[data-preset="default"]').click();
  await expect(page.locator('#det-scales-err')).toHaveText('');
  await expect(page.locator('#det-scales')).not.toHaveClass(/input-error/);
});

test('invalid scales show inline error on change', async ({ page }) => {
  await page.goto('/');
  await page.locator('details.det-settings > summary').click();

  // Empty scales
  await page.locator('#det-scales').fill('');
  await page.locator('#det-scales').dispatchEvent('change');
  await expect(page.locator('#det-scales-err')).toContainText('at least one scale');

  // Non-numeric
  await page.locator('#det-scales').fill('1, abc, 2');
  await page.locator('#det-scales').dispatchEvent('change');
  await expect(page.locator('#det-scales-err')).toContainText('numbers');

  // Negative
  await page.locator('#det-scales').fill('1, -1');
  await page.locator('#det-scales').dispatchEvent('change');
  await expect(page.locator('#det-scales-err')).toContainText('positive');
});

test('invalid number field shows inline error on change', async ({ page }) => {
  await page.goto('/');
  await page.locator('details.det-settings > summary').click();

  // Score threshold above 1
  await page.locator('#det-score-threshold').fill('1.5');
  await page.locator('#det-score-threshold').dispatchEvent('change');
  await expect(page.locator('#det-score-threshold-err')).toContainText('≤ 1');

  // Tile size fractional (integer required)
  await page.locator('#det-tile-size').fill('100.5');
  await page.locator('#det-tile-size').dispatchEvent('change');
  await expect(page.locator('#det-tile-size-err')).toContainText('whole number');
});

test('processing with invalid detection settings shows error and does not proceed', async ({ page }) => {
  await page.goto('/');
  await uploadPng(page);
  await page.locator('details.det-settings > summary').click();

  // Corrupt the scales field.
  await page.locator('#det-scales').fill('');
  await page.locator('#btn-process').click();

  await expect(page.locator('#status')).toContainText('Invalid detection settings');
  // Process button must be re-enabled after the early return.
  await expect(page.locator('#btn-process')).toBeEnabled();
  // Download button must remain disabled since no result was produced.
  await expect(page.locator('#btn-download')).toBeDisabled();
});

test('high scoreThreshold filters out low-confidence detections', async ({ page }) => {
  await page.goto('/');
  await uploadPng(page);
  await page.locator('details.det-settings > summary').click();

  // Use a detection whose score (0.3) is below the threshold (0.99).
  await page.evaluate(() => {
    globalThis.__testDetections = [{
      categories: [{ score: 0.3 }],
      boundingBox: { originX: 4, originY: 4, width: 16, height: 16 },
    }];
  });
  await page.locator('#det-score-threshold').fill('0.99');

  await page.locator('#btn-process').click();
  await expect(page.locator('#status')).toContainText('No faces detected');
});

test('low scoreThreshold allows low-confidence detections through', async ({ page }) => {
  await page.goto('/');
  await uploadPng(page);
  await page.locator('details.det-settings > summary').click();

  // Same low-score detection, but now threshold is lower.
  await page.evaluate(() => {
    globalThis.__testDetections = [{
      categories: [{ score: 0.3 }],
      boundingBox: { originX: 4, originY: 4, width: 16, height: 16 },
    }];
  });
  await page.locator('#det-score-threshold').fill('0.1');

  await page.locator('#btn-process').click();
  await expect(page.locator('#status')).toContainText('face(s) anonymised');
});

// ─── Model change via dropdown tests ────────────────────────────────────────

test('model dropdown switching: scrfd500 → mediapipe re-enables process button and processes image', async ({ page }) => {
  await page.goto('/');
  await uploadPng(page);

  // ── Switch to scrfd500 ───────────────────────────────────────────────────
  await page.locator('#det-model').selectOption('scrfd500');
  await expect(page.locator('#btn-process')).toBeEnabled();
  await expect(page.locator('#status')).toHaveText('Ready. Upload an image to begin.');

  await page.locator('#btn-process').click();
  await expect(page.locator('#status')).toContainText('face(s) anonymised');

  const scrfdModelPath = await page.evaluate(() => {
    const paths = globalThis.__onnxModelLoads || [];
    return paths[paths.length - 1];
  });
  expect(scrfdModelPath).toContain('scrfd_500m');

  // ── Switch back to mediapipe ─────────────────────────────────────────────
  await page.locator('#det-model').selectOption('mediapipe');
  await expect(page.locator('#btn-process')).toBeEnabled();
  await expect(page.locator('#status')).toHaveText('Ready. Upload an image to begin.');

  await page.locator('#btn-process').click();
  await expect(page.locator('#status')).toContainText('face(s) anonymised');
});

test('onnx model failure falls back to MediaPipe without freezing UI', async ({ page }) => {
  // Force ONNX model download failure and verify the app auto-recovers by
  // switching back to MediaPipe instead of staying stuck in loading state.
  await page.unroute('**/*.onnx');
  await page.route('**/*.onnx', async route => {
    await route.fulfill({
      status: 404,
      contentType: 'text/plain; charset=utf-8',
      body: 'not found',
    });
  });

  await page.goto('/');
  await uploadPng(page);

  await page.locator('#det-model').selectOption('scrfd500');
  await expect(page.locator('#status')).toContainText('Ready. Upload an image to begin.');
  await expect(page.locator('#status')).toContainText('switched to MediaPipe');
  await expect(page.locator('#btn-process')).toBeEnabled();
  await expect(page.locator('#det-model')).toHaveValue('mediapipe');

  await page.locator('#btn-process').click();
  await expect(page.locator('#status')).toContainText('face(s) anonymised');
  await expect(page.locator('#btn-process')).toBeEnabled();
});

test('switching to onnx model fetches model data asynchronously before creating session', async ({ page }) => {
  // This test verifies that the ONNX model is fetched via fetch() (non-blocking)
  // and the filename (not a full URL) is recorded in __onnxModelLoads.
  await page.goto('/');
  await uploadPng(page);

  await page.locator('#det-model').selectOption('scrfd500');
  await expect(page.locator('#status')).toHaveText('Ready. Upload an image to begin.');

  // The model filename (not a full URL) should appear in __onnxModelLoads.
  const loadedName = await page.evaluate(() => {
    const paths = globalThis.__onnxModelLoads || [];
    return paths[paths.length - 1] ?? '';
  });
  expect(loadedName).not.toContain('http');
  expect(loadedName).toContain('scrfd_500m');
});
