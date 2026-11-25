import { expect, test } from '@playwright/test';
import { registerAndLogin } from './helpers';

test('analytics tabs and overbooking simulation load API data', async ({ page }) => {
  await registerAndLogin(page);
  await page.getByRole('link', { name: 'Analytics', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Bookings Over Time' })).toBeVisible();

  await page.getByRole('button', { name: 'routes', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Route Performance' })).toBeVisible();
  await page.getByRole('button', { name: 'channels', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Booking Distribution by Channel' })).toBeVisible();
  await page.getByRole('button', { name: 'high risk', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'High-Risk Bookings' })).toBeVisible();
  await expect(page.getByRole('status')).toContainText('Administrator access is required');

  await page.getByRole('link', { name: 'Overbooking', exact: true }).click();
  await page.getByRole('button', { name: 'Run Simulation' }).click();
  await expect(page.getByRole('heading', { name: 'Highest-Scoring Simulated Scenario' })).toBeVisible();
  await expect(page.getByText(
    'This is only the highest-scoring simulated scenario under the supplied inputs and cost assumptions; it is not an operational recommendation.',
    { exact: true },
  )).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Scenario Comparison' })).toBeVisible();
  await expect(page.locator('tbody tr')).not.toHaveCount(0);
});
