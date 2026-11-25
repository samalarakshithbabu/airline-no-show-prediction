import { expect, test } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { registerAndLogin } from './helpers';

test('authenticated user can navigate every page and open mobile navigation', async ({ page }) => {
  await registerAndLogin(page);
  await expect(page.getByRole('heading', { name: 'Executive Dashboard' })).toBeVisible();

  const screenshotPath = resolve(__dirname, 'screenshots', 'dashboard.png');
  mkdirSync(dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });

  const pages = [
    { label: 'Prediction', heading: 'Passenger Prediction' },
    { label: 'Analytics', heading: 'Analytics' },
    { label: 'Overbooking', heading: 'Overbooking Simulator' },
    { label: 'Models', heading: 'Model Evaluation' },
  ];
  for (const item of pages) {
    await page.getByRole('link', { name: item.label, exact: true }).click();
    await expect(page.getByRole('heading', { name: item.heading })).toBeVisible();
  }

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('header button').click();
  await expect(page.getByRole('link', { name: 'Dashboard', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Log out' }).click();
  await expect(page).toHaveURL(/\/login$/);
});
