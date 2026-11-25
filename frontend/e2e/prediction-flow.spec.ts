import { expect, test } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { registerAndLogin } from './helpers';

test('prediction page keeps serving disabled when no model is approved', async ({ page }) => {
  await registerAndLogin(page);
  await page.getByRole('link', { name: 'Prediction', exact: true }).click();

  await expect(page.getByRole('status')).toContainText('No production model is approved');
  await expect(page.getByRole('button', { name: 'Predict No-Show Probability' })).toBeDisabled();
  await expect(page.getByLabel('Previous No-Shows')).toHaveCount(0);
  await expect(page.getByLabel('Booking Date')).toBeVisible();
  await expect(page.getByLabel('Flight Date')).toBeVisible();
  await expect(page.getByLabel('Departure Airport')).toBeVisible();
  await expect(page.getByLabel('Arrival Airport')).toBeVisible();

  const screenshotPath = resolve(__dirname, 'screenshots', 'prediction.png');
  mkdirSync(dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
});
