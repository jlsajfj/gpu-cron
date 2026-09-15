// The demo is a consumer of the published package: it imports the same entry point an npm
// user would and renders whatever parse() returns. Nothing here reaches into the internals.

import { CronError, backend, isAvailable, parse, type CronMatch } from '../src/index.js';

const EXAMPLES = [
  'every weekday at 9am',
  'every 15 minutes',
  'first of the month at midnight',
  'weekends at noon',
  'every 6 hours on the hour',
];

function need<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`missing #${id}`);
  return found as T;
}

const input = need<HTMLInputElement>('input');
const form = need<HTMLFormElement>('ask');
const examples = need<HTMLDivElement>('examples');
const resultPanel = need<HTMLElement>('result');
const cronOut = need<HTMLElement>('cron');
const fires = need<HTMLUListElement>('fires');
const badges = need<HTMLDivElement>('badges');
const errorPanel = need<HTMLElement>('error');
const errorTitle = need<HTMLElement>('error-title');
const errorBody = need<HTMLElement>('error-body');
const themeButton = need<HTMLButtonElement>('theme');
const firedNote = need<HTMLElement>('fires-note');

let running = false;
let queued: string | null = null;
let debounce: number | undefined;

function badge(label: string, value: string, tone = ''): void {
  const element = document.createElement('span');
  element.className = `badge ${tone}`.trim();
  const key = document.createElement('span');
  key.className = 'badge-label';
  key.textContent = label;
  const val = document.createElement('span');
  val.className = 'badge-value';
  val.textContent = value;
  element.append(key, val);
  badges.append(element);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function renderFires(match: CronMatch): void {
  fires.replaceChildren();
  for (const iso of match.next) {
    const time = new Date(iso);
    const item = document.createElement('li');
    const day = document.createElement('span');
    day.className = 'fire-day';
    day.textContent = time.toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
    const clock = document.createElement('span');
    clock.className = 'fire-clock';
    clock.textContent = time.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    item.append(clock, day);
    fires.append(item);
  }
  firedNote.textContent =
    match.next.length === 0 ? 'This expression never fires.' : 'Local time, computed in this tab.';
}

function render(match: CronMatch, millis: number, params: number, bytes: number): void {
  cronOut.textContent = match.expression;
  renderFires(match);
  badges.replaceChildren();
  const info = backend();
  badge('params', params < 1000 ? `${params}` : `${(params / 1000).toFixed(0)}k`, 'good');
  badge('weights', formatBytes(bytes));
  badge('backend', `webgpu · ${info?.adapter || 'device'}`);
  badge('total', `${millis.toFixed(0)} ms`);
  resultPanel.hidden = false;
}

function showError(title: string, message: string): void {
  errorTitle.textContent = title;
  errorBody.textContent = message;
  errorPanel.hidden = false;
  resultPanel.hidden = true;
}

async function run(text: string): Promise<void> {
  if (running) {
    queued = text;
    return;
  }
  running = true;
  try {
    const started = performance.now();
    const match = await parse(text, { count: 5 });
    const millis = performance.now() - started;
    const info = backend();
    errorPanel.hidden = true;
    render(match, millis, info?.params ?? 0, info?.bytes ?? 0);
  } catch (error) {
    const needsWebGpu = error instanceof CronError && /WebGPU/.test(error.message);
    showError(
      needsWebGpu ? 'This browser has no WebGPU' : 'No cron for that',
      error instanceof Error ? error.message : String(error),
    );
  } finally {
    running = false;
    const next = queued;
    queued = null;
    if (next !== null && next !== text) void run(next);
  }
}

function schedule(text: string): void {
  window.clearTimeout(debounce);
  debounce = window.setTimeout(() => void run(text), 250);
}

function buildExamples(): void {
  for (const example of EXAMPLES) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'example';
    button.textContent = example;
    button.addEventListener('click', () => {
      input.value = example;
      input.focus();
      void run(example);
    });
    examples.append(button);
  }
}

function applyTheme(theme: string | null): void {
  if (theme === null) delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  try {
    if (theme === null) localStorage.removeItem('human-cron-theme');
    else localStorage.setItem('human-cron-theme', theme);
  } catch {
    // storage can be unavailable in a private window; the toggle still works for this page
  }
}

function initTheme(): void {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem('human-cron-theme');
  } catch {
    stored = null;
  }
  if (stored !== null) applyTheme(stored);
  themeButton.addEventListener('click', () => {
    // Cycles dark, light, then back to whatever the system prefers.
    const current = document.documentElement.dataset.theme;
    applyTheme(current === 'dark' ? 'light' : current === 'light' ? null : 'dark');
  });
}

async function gateOnBackend(): Promise<boolean> {
  if (await isAvailable()) return true;
  const tip = 'GPU not available';
  input.disabled = true;
  input.title = tip;
  input.placeholder = tip;
  for (const el of document.querySelectorAll<HTMLButtonElement>('form button, .examples button')) {
    el.disabled = true;
    el.title = tip;
  }
  return false;
}

function boot(): void {
  buildExamples();
  initTheme();
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    void run(input.value);
  });
  input.addEventListener('input', () => schedule(input.value));
  input.focus();

  // ?q= makes a prompt linkable, and gives the smoke test a way to drive a specific input.
  const deepLink = new URLSearchParams(location.search).get('q');
  if (deepLink !== null && deepLink.trim() !== '') input.value = deepLink;
  if (input.value.trim() === '') input.value = EXAMPLES[0] as string;
  void gateOnBackend().then((ok) => {
    if (ok) void run(input.value);
  });
}

boot();
