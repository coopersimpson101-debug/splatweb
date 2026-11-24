/**
 * Progressive loading: pulls bytes from a ReadableStream (fetch body or File.stream()) and
 * feeds them to an incremental parser. `onProgress` fires as soon as new splats are decoded, so
 * the renderer can upload and draw a partially loaded scene.
 */
import { PlyStreamParser } from './ply';
import type { PlyParseOptions } from './ply';
import { SplatStreamParser } from './splat';
import type { SplatData } from './splatData';

export interface StreamParser {
  /** Feed a chunk; returns how many new splats were decoded. */
  push(chunk: Uint8Array): number;
  finish(): SplatData;
  /** Output buffer (available once the header / size is known). */
  readonly data: SplatData | null;
  /** Total number of splats expected (0 if not yet known). */
  readonly total: number;
}

export type SceneFormat = 'ply' | 'splat';

export interface LoadProgress {
  data: SplatData | null;
  loadedSplats: number;
  totalSplats: number;
  loadedBytes: number;
  totalBytes: number | undefined;
}

export interface LoadOptions extends PlyParseOptions {
  format?: SceneFormat;
  onProgress?: (p: LoadProgress) => void;
  signal?: AbortSignal;
}

export function guessFormat(name: string, firstBytes?: Uint8Array): SceneFormat {
  if (firstBytes && firstBytes.length >= 4) {
    if (firstBytes[0] === 0x70 && firstBytes[1] === 0x6c && firstBytes[2] === 0x79 && (firstBytes[3] === 0x0a || firstBytes[3] === 0x0d))
      return 'ply';
  }
  const lower = name.toLowerCase().split('?')[0];
  if (lower.endsWith('.ply')) return 'ply';
  if (lower.endsWith('.splat')) return 'splat';
  return firstBytes ? 'splat' : 'ply';
}

/** Reads a stream to completion through the appropriate parser. */
export async function loadFromStream(
  stream: ReadableStream<Uint8Array>,
  name: string,
  totalBytes: number | undefined,
  opts: LoadOptions = {},
): Promise<SplatData> {
  const reader = stream.getReader();
  let parser: StreamParser | null = null;
  let loadedBytes = 0;
  let lastReport = 0;
  const report = (force: boolean) => {
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (!force && now - lastReport < 50) return;
    lastReport = now;
    opts.onProgress?.({
      data: parser?.data ?? null,
      loadedSplats: parser?.data?.loaded ?? 0,
      totalSplats: parser?.total ?? 0,
      loadedBytes,
      totalBytes,
    });
  };
  try {
    for (;;) {
      if (opts.signal?.aborted) throw new DOMException('Load aborted', 'AbortError');
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || value.length === 0) continue;
      loadedBytes += value.length;
      if (!parser) {
        const fmt = opts.format ?? guessFormat(name, value);
        parser = fmt === 'ply' ? new PlyStreamParser(opts) : new SplatStreamParser(totalBytes);
      }
      const added = parser.push(value);
      if (added > 0) report(false);
    }
  } finally {
    reader.releaseLock();
  }
  if (!parser) throw new Error('Empty file');
  const data = parser.finish();
  report(true);
  return data;
}

export async function loadFromUrl(url: string, opts: LoadOptions = {}): Promise<SplatData> {
  const res = await fetch(url, { signal: opts.signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} while fetching ${url}`);
  const len = res.headers.get('content-length');
  // Content-Length is the *encoded* size; only trust it when the body is not compressed.
  const encoded = res.headers.get('content-encoding');
  const total = len && !encoded ? parseInt(len, 10) : undefined;
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    return loadFromStream(new Blob([buf]).stream(), url, buf.length, opts);
  }
  return loadFromStream(res.body, url, total, opts);
}

export function loadFromFile(file: File, opts: LoadOptions = {}): Promise<SplatData> {
  return loadFromStream(file.stream(), file.name, file.size, opts);
}
