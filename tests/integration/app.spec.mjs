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

test('the site initializes and anonymises an uploaded image', async ({ page }) => {
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

  const detectorModelPath = await page.evaluate(() => globalThis.__detectorOptions?.[0]?.baseOptions?.modelAssetPath);
  expect(detectorModelPath).toMatch(/face_detector\.task$/);

  const centerPixel = await page.evaluate(() => {
    const canvas = document.getElementById('preview');
    const ctx = canvas.getContext('2d');
    return Array.from(ctx.getImageData(12, 12, 1, 1).data);
  });
  expect(centerPixel).toEqual([0, 0, 0, 255]);
});
