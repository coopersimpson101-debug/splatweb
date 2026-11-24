/** Tiny FPS / timing overlay. Frame time uses an exponential moving average. */
import type { RendererStats } from '../gpu/renderer';

export class StatsOverlay {
  private ema = 16;
  private last = 0;
  private lastPaint = 0;

  constructor(private readonly el: HTMLElement) {}

  /** Call once per animation frame with the rAF timestamp. */
  tick(now: number, r: RendererStats, extra: Record<string, string> = {}): void {
    if (this.last > 0) {
      const dt = Math.min(250, now - this.last);
      this.ema += (dt - this.ema) * 0.05;
    }
    this.last = now;
    if (now - this.lastPaint < 250) return;
    this.lastPaint = now;
    const fmt = (v?: number) => (v === undefined ? 'n/a' : `${v.toFixed(2)} ms`);
    const lines = [
      `FPS ${(1000 / this.ema).toFixed(0)}  (${this.ema.toFixed(1)} ms)`,
      `splats ${r.splats.toLocaleString()}  visible ${r.visible.toLocaleString()}`,
      `gpu preprocess ${fmt(r.gpuPreprocessMs)}`,
      `gpu sort ${fmt(r.gpuSortMs)}`,
      `gpu raster ${fmt(r.gpuRenderMs)}`,
    ];
    if (r.cpuSortMs !== undefined) lines.push(`cpu sort ${fmt(r.cpuSortMs)}`);
    for (const [k, v] of Object.entries(extra)) lines.push(`${k} ${v}`);
    this.el.textContent = lines.join('\n');
  }
}
