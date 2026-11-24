/**
 * Generates docs/assets/banner.svg. The motif on the right is real renderer data, not a drawing:
 * splats of the built-in procedural scene (torus knot, sphere, cube, galaxy; floor omitted) are
 * projected with the CPU reference `projectSplat` (the same EWA maths as preprocess.wgsl), sorted
 * back to front, and each one is written as an SVG ellipse with its projected 2D covariance
 * (axes and rotation) and opacity. Colour is the accent pink, shaded by the splat's SH luminance.
 *
 *   npx vite-node scripts/make_banner.ts
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { OrbitCamera, projectionMatrix } from '../src/camera/camera';
import { projectSplat } from '../src/core/covariance';
import { generateDemoScene } from '../src/io/procedural';
import { shFloatsPerSplat } from '../src/io/splatData';
import type { Vec3 } from '../src/core/math/vec3';
import type { Quat } from '../src/core/math/quat';

const OUT = 'docs/assets/banner.svg';
const W = 1280, H = 400;
const VX = 620, VW = 660; // motif viewport on the right

const scene = generateDemoScene({ count: 7000 });
const nFloor = Math.floor(scene.loaded * 0.34);
const cam = new OrbitCamera();
cam.target = [-0.1, 1.65, 0];
cam.distance = 5.9;
cam.yaw = 0.6;
cam.pitch = 0.3;
const camIn = {
  view: cam.viewMatrix(),
  proj: projectionMatrix(cam.lens, VW / H),
  viewport: [VW, H] as [number, number],
  camPos: cam.eye,
  near: cam.lens.near,
};

const stride = shFloatsPerSplat(scene);
const tones = ['#831843', '#be185d', '#ec4899', '#f472b6', '#fbcfe8'];
const items: { d: number; s: string }[] = [];
for (let i = nFloor; i < scene.loaded; i++) {
  const p = projectSplat({
    position: Array.from(scene.positions.subarray(i * 3, i * 3 + 3)) as Vec3,
    scale: Array.from(scene.scales.subarray(i * 3, i * 3 + 3)) as Vec3,
    rotation: Array.from(scene.rotations.subarray(i * 4, i * 4 + 4)) as Quat,
    opacity: scene.opacities[i],
    sh: scene.sh.subarray(i * stride, (i + 1) * stride),
    shDegree: scene.shDegree,
  }, camIn);
  if (!p) continue;
  const cx = VX + p.center[0], cy = H - p.center[1]; // NDC y is up, SVG y is down
  const rx = Math.hypot(...p.axis1) / 3 * 1.6, ry = Math.hypot(...p.axis2) / 3 * 1.6; // ~1.6 sigma
  if (rx < 0.3) continue;
  const ang = (-Math.atan2(p.axis1[1], p.axis1[0]) * 180) / Math.PI;
  const lum = 0.2126 * p.color[0] + 0.7152 * p.color[1] + 0.0722 * p.color[2];
  const tone = Math.max(0, Math.min(tones.length - 1, Math.round(lum * (tones.length - 1))));
  const f = (v: number) => v.toFixed(1);
  items.push({
    d: p.depth,
    s: `<ellipse cx="${f(cx)}" cy="${f(cy)}" rx="${f(rx)}" ry="${f(Math.max(ry, 0.3))}" transform="rotate(${f(ang)} ${f(cx)} ${f(cy)})" fill="url(#t${tone})" opacity="${p.opacity.toFixed(2)}"/>`,
  });
}
items.sort((a, b) => b.d - a.d); // back to front, like the "over" blend mode

const grads = tones.map((c, k) =>
  `<radialGradient id="t${k}"><stop offset="0" stop-color="${c}"/><stop offset="0.55" stop-color="${c}" stop-opacity="0.55"/><stop offset="1" stop-color="${c}" stop-opacity="0"/></radialGradient>`).join('');

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="splatweb: Gaussian splatting renderer for WebGPU">
<defs>
<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="#21262d" stroke-width="1"/></pattern>
<linearGradient id="fade" x1="0" x2="1"><stop offset="0.45" stop-color="#0d1117" stop-opacity="0"/><stop offset="1" stop-color="#0d1117" stop-opacity="0.9"/></linearGradient>
<clipPath id="motif"><rect x="${VX}" y="0" width="${W - VX}" height="${H}"/></clipPath>
${grads}
</defs>
<rect width="${W}" height="${H}" fill="#0d1117"/>
<rect width="${W}" height="${H}" fill="url(#grid)"/>
<rect width="${W}" height="${H}" fill="url(#fade)" transform="rotate(180 ${W / 2} ${H / 2})"/>
<g clip-path="url(#motif)">
${items.map((it) => it.s).join('\n')}
</g>
<g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">
<rect x="72" y="118" width="6" height="92" fill="#F472B6"/>
<text x="100" y="196" font-size="88" font-weight="700" fill="#e6edf3">splatweb</text>
<text x="102" y="252" font-size="24" fill="#8b949e">3D Gaussian Splatting for WebGPU, from scratch</text>
<text x="102" y="290" font-size="18" fill="#F472B6">EWA projection · GPU radix sort · streaming .ply/.splat</text>
</g>
</svg>
`;
mkdirSync('docs/assets', { recursive: true });
writeFileSync(OUT, svg);
console.log(`wrote ${OUT}: ${items.length} projected splats, ${(svg.length / 1024).toFixed(0)} KiB`);
