import { decode, newStats } from './decode.js';
import { nextFireTimes } from './cron.js';
import { isWellFormed } from './automaton.js';
import { loadRuntime, latchInferenceFailure, modelSizeLabel, type Runtime } from './runtime.js';

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

const stats = newStats();
let runtime: Runtime | null = null;
let running = false;
let queued: string | null = null;
let debounce: number | undefined;

const FAILURE_TITLES: Record<string, string> = {
  'no-webgpu': 'WebGPU is not available here',
  'no-onnx': 'The model is not in this build',
  'inference-failed': 'Inference stopped',
};

function showError(reason: string, message: string): void {
  errorTitle.textContent = FAILURE_TITLES[reason] ?? 'Something went wrong';
  errorBody.textContent = message;
  errorPanel.hidden = false;
  resultPanel.hidden = true;
}

function clearError(): void {
  errorPanel.hidden = true;
}

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

function renderFires(cron: string): void {
  fires.replaceChildren();
  const next = nextFireTimes(cron, 5);
  for (const time of next) {
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
    next.length === 0
      ? 'This expression never fires.'
      : 'Local time, computed in this tab.';
}

function renderResult(cron: string, millis: string, perToken: string): void {
  cronOut.textContent = cron;
  renderFires(cron);
  badges.replaceChildren();
  badge('runs on', runtime?.provider ?? 'model', runtime?.provider === 'webgpu' ? 'good' : 'plain');
  badge('model', runtime ? modelSizeLabel(runtime.modelBytes) : 'unknown');
  badge('per token', perToken);
  badge('total', millis);
  if (runtime !== null && !runtime.webgpuAvailable) {
    badge('note', 'no WebGPU in this browser, running the wasm build', 'warn');
  }
  resultPanel.hidden = false;
}

function formatMillis(ms: number): string {
  return ms < 1 ? `${ms.toFixed(2)} ms` : `${ms.toFixed(0)} ms`;
}

async function run(text: string): Promise<void> {
  if (runtime === null) return;
  if (running) {
    queued = text;
    return;
  }
  running = true;
  try {
    const decoded = await decode(text, runtime.logits, { stats });
    const cron = decoded.text;
    const perToken =
      decoded.tokenMillis.length === 0
        ? 'n/a'
        : formatMillis(decoded.tokenMillis.reduce((a, b) => a + b, 0) / decoded.tokenMillis.length);
    if (decoded.truncated || !isWellFormed(cron)) {
      showError(
        'inference-failed',
        `The decoder stopped early on ${JSON.stringify(cron)}. Try rephrasing.`,
      );
    } else {
      clearError();
      renderResult(cron, formatMillis(decoded.totalMillis), perToken);
    }
  } catch (error) {
    const failure = latchInferenceFailure(error);
    showError(failure.reason, failure.message);
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
    if (theme === null) localStorage.removeItem('nl-cron-theme');
    else localStorage.setItem('nl-cron-theme', theme);
  } catch {
    // storage can be unavailable in a private window; the toggle still works for this page
  }
}

function initTheme(): void {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem('nl-cron-theme');
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

async function boot(): Promise<void> {
  buildExamples();
  initTheme();
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    void run(input.value);
  });
  input.addEventListener('input', () => schedule(input.value));
  input.focus();

  const loaded = await loadRuntime();
  if (!loaded.ok) {
    showError(loaded.reason, loaded.message);
    return;
  }
  runtime = loaded.runtime;
  // ?q= makes a prompt linkable, and gives the smoke test a way to drive a specific input.
  const deepLink = new URLSearchParams(location.search).get('q');
  if (deepLink !== null && deepLink.trim() !== '') input.value = deepLink;
  if (input.value.trim() === '') input.value = EXAMPLES[0] as string;
  void run(input.value);
}

void boot();
