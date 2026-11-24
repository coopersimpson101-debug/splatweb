# Contributing to splatweb

Thanks for your interest! Bug reports, ideas and pull requests are all welcome.

## Setup

```bash
npm install
npm run dev     # local demo at http://localhost:5173
npm test        # vitest
npm run build   # typecheck + production build
```

## Guidelines

- Keep the "from scratch" spirit: no rendering, physics or maths libraries in `src/`.
- Every new algorithm should come with a test against a CPU reference or an analytic result.
- Report numbers only if you measured them, and say on which GPU, browser and settings.
- Keep PRs focused; describe what you changed and how you verified it.
- When reporting a GPU bug, include your browser, OS, GPU and the `navigator.gpu` adapter info
  (the bug-report form explains how to get it).

By contributing you agree that your contributions are licensed under the MIT License.
