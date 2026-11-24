import { defineConfig } from 'vitest/config';

// `base: './'` makes the build work from any sub-path (GitHub Pages project sites).
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: 'index.html',
        minimal: 'examples/minimal-host.html',
      },
    },
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
    testTimeout: 60000,
  },
});
