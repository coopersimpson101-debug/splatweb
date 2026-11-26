/**
 * splatweb public API — import this from a host application/engine.
 *
 *   import { SplatRenderer, loadFromUrl } from 'splatweb';
 *
 * See docs/vanta-integration.md for the host contract.
 */
export { SplatRenderer } from './gpu/renderer';
export type { SplatRendererOptions, SplatView, SplatTarget, RendererStats, SortBackend, BlendOrder } from './gpu/renderer';
export { GpuRadixSorter } from './gpu/radixSort';
export type { SplatData, Bounds } from './io/splatData';
export { createSplatData, computeBounds } from './io/splatData';
export { parsePly, writePly, PlyStreamParser, parsePlyHeader } from './io/ply';
export { parseSplat, writeSplat, SplatStreamParser } from './io/splat';
export { loadFromUrl, loadFromFile, loadFromStream, guessFormat } from './io/stream';
export type { LoadOptions, LoadProgress, SceneFormat } from './io/stream';
export { generateDemoScene } from './io/procedural';
export { LiveScene, connectLive, decodeLiveMessage, flushLiveScene, upAxisModelMatrix, viewFromFrame, recordFloats, LIVE_MAGIC, LIVE_VERSION, MSG as LIVE_MSG } from './io/live';
export type { LiveMessage, LiveFrame, LiveHello, LiveSplats, LiveSceneOptions, LiveConnection, LiveRendererTarget, DirtyRanges } from './io/live';
export * as mat4 from './core/math/mat4';
export * as quat from './core/math/quat';
export * as vec3 from './core/math/vec3';
export { evalSH, shBasis } from './core/sh';
export { projectSplat, computeCov3D, projectCov3D, conicFromCov2D, quadAxes } from './core/covariance';
export { radixSortPairs, cpuDepthSort, depthKey } from './core/sort';
export { OrbitCamera, FlyCamera, projectionMatrix } from './camera/camera';
export { CameraControls } from './camera/controls';
export { isImmersiveVrAvailable, startImmersiveVr } from './xr/xr';
