import { defineConfig } from 'vitest/config';

export default defineConfig({
  // The grammar and its conformance fixture live one directory up, outside the vite root.
  server: { fs: { allow: ['..'] } },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
    globals: false,
    // The two exhaustive exploration tests run 5.5-7s and flake against vitest's 5s
    // default whenever the machine is busy.
    testTimeout: 60_000,
  },
});
