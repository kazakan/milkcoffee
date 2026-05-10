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

const RED_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAK0lEQVR4nO3OIQEAAAwEoetfeovxBoGnq1tKQEBAQEBAQEBAQEBAQEBgHXhUDfhqRFDd3gAAAABJRU5ErkJggg==';

test.beforeEach(async ({ page }) => {
  await page.route('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'text/javascript; charset=utf-8',
      body: MEDIAPIPE_STUB,
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

  await expect(page.locator('#status')).toContainText('Done! 1 face(s) anonymised');
  await expect(page.locator('#btn-download')).toBeEnabled();

  const centerPixel = await page.evaluate(() => {
    const canvas = document.getElementById('preview');
    const ctx = canvas.getContext('2d');
    return Array.from(ctx.getImageData(12, 12, 1, 1).data);
  });
  expect(centerPixel).toEqual([0, 0, 0, 255]);
});
