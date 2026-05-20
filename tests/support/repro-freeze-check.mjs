import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const imagePath = path.resolve(__dirname, 'repro-face.png');

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

page.on('console', msg => {
  console.log(`[console:${msg.type()}] ${msg.text()}`);
});
page.on('pageerror', err => {
  console.log(`[pageerror] ${err?.message || err}`);
});
page.on('crash', () => {
  console.log('[pagecrash] The page crashed.');
});

try {
  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle', timeout: 30000 });
  await page.locator('#status').waitFor({ state: 'visible', timeout: 15000 });
  console.log('Initial status:', await page.locator('#status').textContent());

  await page.locator('#file-input').setInputFiles(imagePath);
  await page.locator('#status').filter({ hasText: 'Image loaded' }).waitFor({ timeout: 10000 });
  console.log('After upload:', await page.locator('#status').textContent());

  const models = [
    { id: 'scrfd500', label: 'SCRFD-500M' },
    { id: 'mediapipe', label: 'MediaPipe' },
  ];

  for (const model of models) {
    console.log(`Switching to ${model.label}...`);
    const isDisabled = await page.locator(`#det-model option[value="${model.id}"]`).isDisabled();
    if (isDisabled) {
      console.log(`Skipping ${model.label}: option is disabled by app safety guard.`);
      continue;
    }
    await page.locator('#det-model').selectOption(model.id);
    await page.locator('#status').filter({ hasText: 'Ready. Upload an image to begin.' }).waitFor({ timeout: 30000 });
    const processDisabled = await page.locator('#btn-process').isDisabled();
    const modelDisabled = await page.locator('#det-model').isDisabled();
    console.log(`Model ${model.label} loaded. btn-process disabled=${processDisabled}, det-model disabled=${modelDisabled}`);

    await page.locator('#btn-process').click({ timeout: 10000 });
    await page.locator('#status').filter({ hasText: 'face(s) anonymised' }).waitFor({ timeout: 30000 });
    console.log(`Processed with ${model.label}:`, await page.locator('#status').textContent());
  }

  console.log('Repro check complete: no freeze observed in scripted run.');
} catch (err) {
  console.error('Repro check failed:', err);
  console.log('Page closed:', page.isClosed());
  console.log('Current URL:', page.url());
  console.log('Current status text:', await page.locator('#status').textContent({ timeout: 1000 }).catch(() => 'N/A'));
  console.log('btn-process disabled:', await page.locator('#btn-process').isDisabled({ timeout: 1000 }).catch(() => 'N/A'));
  console.log('det-model disabled:', await page.locator('#det-model').isDisabled({ timeout: 1000 }).catch(() => 'N/A'));
  await page.screenshot({ path: 'test-results/repro-freeze-check-failure.png', fullPage: true }).catch(() => {});
  const bodyText = await page.locator('body').innerText({ timeout: 1000 }).catch(() => 'N/A');
  console.log('Body text snapshot:', String(bodyText).slice(0, 400));
  process.exitCode = 1;
} finally {
  await browser.close();
}
