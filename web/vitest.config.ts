import { defineConfig } from 'vitest/config';

export default defineConfig({
  // The grammar and its conformance fixture live one directory up, outside the vite root.
  server: { fs: { allow: ['..'] } },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
    globals: false,
  },
});
