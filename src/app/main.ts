/**
 * The demo viewer app (what GitHub Pages serves). Everything WebGPU-specific lives in
 * SplatRenderer; this file is DOM glue: device setup, canvas, UI, loading and the frame loop.
 *
 * URL parameters:
 *   ?url=<scene.ply|scene.splat>   load a remote scene (must be CORS-enabled)
 *   ?count=<n>                     size of the procedural demo scene (default 150000)
 *   ?sort=gpu|cpu  ?blend=back-to-front|front-to-back  ?flip=0|1  ?sh=0..3
 */
import { CameraControls } from '../camera/controls';
import { projectionMatrix } from '../camera/camera';
import { fromScaling } from '../core/math/mat4';
import { SplatRenderer } from '../gpu/renderer';
import type { BlendOrder, SortBackend } from '../gpu/renderer';
import { generateDemoScene } from '../io/procedural';
import type { SplatData } from '../io/splatData';
import { computeBounds } from '../io/splatData';
import { loadFromFile, loadFromUrl } from '../io/stream';
import type { LoadProgress } from '../io/stream';
import { StatsOverlay } from '../ui/stats';
import { isImmersiveVrAvailable, startImmersiveVr } from '../xr/xr';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);

function fatal(msg: string): never {
  const el = $('fatal');
  el.textContent = msg;
  el.hidden = false;
  throw new Error(msg);
}

let toastTimer = 0;
function toast(msg: string, ms = 3500) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => (el.hidden = true), ms);
}

async function initGpu() {
  if (!('gpu' in navigator) || !navigator.gpu)
    fatal('WebGPU is not available in this browser.\nTry a recent Chrome, Edge, or Safari 26+, or Firefox with WebGPU enabled.');
  const xrCompatible = 'xr' in navigator;
  const adapter =
    (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance', xrCompatible } as GPURequestAdapterOptions)) ??
    (await navigator.gpu.requestAdapter());
  if (!adapter) fatal('No WebGPU adapter found.');
  const features: GPUFeatureName[] = [];
  if (adapter.features.has('timestamp-query')) features.push('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((info) => fatal(`WebGPU device lost: ${info.message}`));
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message));
  return { adapter, device };
}

async function main() {
  const { adapter, device } = await initGpu();
  const canvas = $<HTMLCanvasElement>('canvas');
  const context = canvas.getContext('webgpu');
  if (!context) fatal('Could not create a WebGPU canvas context.');
  const format = navigator.gpu.getPreferredCanvasFormat();
  // Premultiplied canvas: the splats composite over the CSS background gradient.
  context.configure({ device, format, alphaMode: 'premultiplied' });

  const renderer = new SplatRenderer({
    device,
    colorFormat: format,
    sort: (params.get('sort') as SortBackend) ?? 'gpu',
    blend: (params.get('blend') as BlendOrder) ?? 'back-to-front',
  });
  const controls = new CameraControls(canvas);
  const stats = new StatsOverlay($('stats'));
  const info = adapter.info;
  const adapterName = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ') || 'unknown adapter';

  // ---- UI wiring --------------------------------------------------------------------------
  const sortSel = $<HTMLSelectElement>('sort');
  const blendSel = $<HTMLSelectElement>('blend');
  const shSel = $<HTMLSelectElement>('sh');
  const scaleIn = $<HTMLInputElement>('scale');
  const flipIn = $<HTMLInputElement>('flip');
  const modeSel = $<HTMLSelectElement>('mode');
  sortSel.value = renderer.sort;
  blendSel.value = renderer.blend;
  if (params.has('sh')) shSel.value = params.get('sh')!;
  renderer.maxShDegree = parseInt(shSel.value, 10);
  sortSel.onchange = () => (renderer.sort = sortSel.value as SortBackend);
  blendSel.onchange = () => (renderer.blend = blendSel.value as BlendOrder);
  shSel.onchange = () => (renderer.maxShDegree = parseInt(shSel.value, 10));
  scaleIn.oninput = () => (renderer.scaleModifier = parseFloat(scaleIn.value));
  const applyFlip = () => {
    renderer.modelMatrix = flipIn.checked ? fromScaling([1, -1, -1]) : fromScaling([1, 1, 1]);
  };
  flipIn.onchange = () => {
    applyFlip();
    if (current) frameScene(current);
  };
  modeSel.onchange = () => controls.setMode(modeSel.value as 'orbit' | 'fly');
  controls.onModeChange = (m) => (modeSel.value = m);
  $('toggle-panel').onclick = () => {
    const body = $('panel-body');
    body.hidden = !body.hidden;
    $('toggle-panel').innerHTML = body.hidden ? '+' : '&minus;';
  };

  // ---- scene loading ----------------------------------------------------------------------
  let current: SplatData | null = null;
  let loadAbort: AbortController | null = null;
  let framedVersion = -1;
  const progressEl = $('progress');
  const progressBar = progressEl.firstElementChild as HTMLElement;

  function frameScene(d: SplatData) {
    const b = computeBounds(d);
    const c = flipIn.checked ? [b.center[0], -b.center[1], -b.center[2]] : b.center;
    controls.setMode('orbit');
    controls.orbit.frame(c as [number, number, number], b.radius);
    controls.orbit.yaw = 0;
    controls.orbit.pitch = 0.25;
    framedVersion = ++controls.version;
  }

  function onProgress(p: LoadProgress) {
    if (p.data && p.data !== current) {
      current = p.data;
      renderer.setScene(p.data);
      if (p.data.loaded > 0) frameScene(p.data);
    } else if (p.data) {
      renderer.updateScene(p.data);
      // Keep re-framing while the scene grows, until the user touches the camera.
      if (controls.version === framedVersion && p.loadedSplats > 0) frameScene(p.data);
    }
    const frac = p.totalBytes ? p.loadedBytes / p.totalBytes : p.totalSplats ? p.loadedSplats / p.totalSplats : 0;
    progressBar.style.width = `${(frac * 100).toFixed(1)}%`;
  }

  async function load(what: { url?: string; file?: File }) {
    loadAbort?.abort();
    const abort = new AbortController();
    loadAbort = abort;
    progressEl.hidden = false;
    progressBar.style.width = '0%';
    flipIn.checked = params.get('flip') !== '0';
    applyFlip();
    current = null;
    const t0 = performance.now();
    try {
      const opts = { onProgress, signal: abort.signal };
      const data = what.file ? await loadFromFile(what.file, opts) : await loadFromUrl(what.url!, opts);
      if (abort.signal.aborted) return;
      onProgress({ data, loadedSplats: data.loaded, totalSplats: data.count, loadedBytes: 1, totalBytes: 1 });
      toast(`Loaded ${data.loaded.toLocaleString()} splats (SH degree ${data.shDegree}) in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    } catch (e) {
      if ((e as Error).name !== 'AbortError') toast(`Load failed: ${(e as Error).message}`, 8000);
    } finally {
      if (loadAbort === abort) progressEl.hidden = true;
    }
  }

  function loadDemo() {
    loadAbort?.abort();
    const n = parseInt(params.get('count') ?? '150000', 10);
    const t0 = performance.now();
    const data = generateDemoScene({ count: n });
    flipIn.checked = params.get('flip') === '1';
    applyFlip();
    current = data;
    renderer.setScene(data);
    frameScene(data);
    controls.orbit.distance *= 0.75;
    controls.orbit.target = [0, 1.2, 0];
    toast(`Procedural scene: ${data.loaded.toLocaleString()} splats generated in ${(performance.now() - t0).toFixed(0)} ms`);
  }

  $<HTMLInputElement>('file').onchange = (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (f) void load({ file: f });
  };
  $<HTMLFormElement>('url-form').onsubmit = (e) => {
    e.preventDefault();
    const url = $<HTMLInputElement>('url').value.trim();
    if (url) void load({ url });
  };
  $('demo').onclick = loadDemo;

  // Drag & drop anywhere.
  const drop = $('drop');
  let dragDepth = 0;
  window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; drop.hidden = false; });
  window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; drop.hidden = true; } });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    drop.hidden = true;
    const f = e.dataTransfer?.files?.[0];
    if (f) void load({ file: f });
  });

  // ---- canvas sizing ----------------------------------------------------------------------
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  const currentView = () => ({
    view: controls.viewMatrix(),
    proj: projectionMatrix(controls.lens, canvas.width / canvas.height),
    viewport: [canvas.width, canvas.height] as [number, number],
    near: controls.lens.near,
  });

  $('selftest').onclick = async () => {
    const r = await renderer.validateGpuSort(currentView());
    toast(r.message, 6000);
    console.info('[splatweb] self-test', r);
  };

  // ---- WebXR (experimental) ---------------------------------------------------------------
  let inXr = false;
  if (await isImmersiveVrAvailable()) {
    const btn = $<HTMLButtonElement>('vr');
    btn.hidden = false;
    btn.onclick = async () => {
      try {
        inXr = true;
        await startImmersiveVr(
          device,
          (fmt) => {
            const r = new SplatRenderer({ device, colorFormat: fmt, sort: renderer.sort, blend: renderer.blend, timestamps: false });
            r.modelMatrix = renderer.modelMatrix;
            if (current) r.setScene(current);
            return r;
          },
          () => (inXr = false),
        );
      } catch (e) {
        inXr = false;
        toast(`Could not start VR: ${(e as Error).message}`, 6000);
      }
    };
  }

  // ---- frame loop -------------------------------------------------------------------------
  let last = performance.now();
  let frame = 0;
  const loop = (now: number) => {
    requestAnimationFrame(loop);
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (inXr) return;
    controls.update(dt);
    if (++frame % 15 === 0) renderer.requestStats();
    renderer.render(
      { colorView: context.getCurrentTexture().createView(), clear: { r: 0, g: 0, b: 0, a: 0 } },
      currentView(),
    );
    stats.tick(now, renderer.stats, {
      mode: `${renderer.sort} sort, ${renderer.blend}`,
      res: `${canvas.width}x${canvas.height}`,
      gpu: adapterName,
    });
  };

  const url = params.get('url');
  if (url) void load({ url });
  else loadDemo();
  requestAnimationFrame(loop);
}

main().catch((e) => {
  console.error(e);
  const el = document.getElementById('fatal');
  if (el && el.hidden) {
    el.textContent = `splatweb failed to start: ${(e as Error).message}`;
    el.hidden = false;
  }
});
