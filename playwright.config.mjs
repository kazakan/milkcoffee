import { defineConfig, devices } from '@playwright/test';

const port = Number(process.env.PORT || 4173);

export default defineConfig({
  testDir: './tests/integration',
  fullyParallel: false,
  retries: 0,
  timeout: 30_000,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    headless: true,
  },
  projects: [
    {
      name: 'desktop-chromium',
      use: {
        browserName: 'chromium',
        viewport: { width: 1280, height: 720 },
      },
    },
    {
      name: 'mobile-chromium',
      use: {
        ...devices['Pixel 5'],
        browserName: 'chromium',
      },
    },
  ],
  webServer: {
    command: `node tests/support/static-server.mjs`,
    port,
    reuseExistingServer: !process.env.CI,
  },
});
