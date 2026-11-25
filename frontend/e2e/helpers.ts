import { expect, Page } from '@playwright/test';
import { randomUUID } from 'node:crypto';

export async function registerAndLogin(page: Page) {
  const suffix = randomUUID().replace(/-/g, '');
  const username = `e2e-${suffix.slice(0, 16)}`;
  const password = 'BrowserTest!Password2026';
  const response = await page.request.post('http://127.0.0.1:5000/api/auth/register', {
    data: {
      username,
      email: `${suffix}@example.test`,
      password,
    },
  });
  expect(response.status()).toBe(201);

  await page.goto('/login');
  await page.getByPlaceholder('Enter username').fill(username);
  await page.getByPlaceholder('Enter password').fill(password);
  await page.getByRole('button', { name: 'Sign In' }).click();
  await expect(page).toHaveURL('http://127.0.0.1:3000/');
}
