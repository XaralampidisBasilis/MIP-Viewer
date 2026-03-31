# Project Context: MIP-Viewer

This file is a fast-reference map of the repository for future coding sessions.

## Purpose
- Browser-based medical volume renderer for Maximum Intensity Projection (MIP).
- Focus on interactive raymarching plus acceleration via precomputed maps.
- Main data format: NIFTI (`.nii.gz`).

## Tech Stack
- Runtime/build: Vite
- Rendering: Three.js (WebGL2)
- Compute/preprocess: TensorFlow.js WebGL backend + custom GPGPU programs
- UI controls: lil-gui
- Perf stats: stats.js
- Shader loading: vite-plugin-glsl

## High-Level Architecture
- `src/script.js`: app entrypoint, initializes TensorFlow first, then `Experience`.
- `src/javascript/Experience.js`: composition root and lifecycle manager.
- `src/javascript/World/World.js`: scene-level owner of `MIPViewer`.
- `src/javascript/World/MIPViewer/*`: core material/uniform/define wiring and per-frame raymarch uniform updates.
- `src/javascript/Computes/*`: tensor preprocessing and distance-map texture generation.
- `src/shaders/mip_viewer/*`: raycast/march/shade/debug GLSL pipeline.

## Startup Flow
1. `setTensorflow()` in `src/javascript/tensorflow.js`.
2. Create `Experience(canvas, webgl2Context)`.
3. Load resources via `Resources` + `NIFTILoader`.
4. On resources ready: run `Computes.start()`.
5. Build and start world/viewer, then GUI.
6. Per-frame loop: camera -> viewer uniform updates -> renderer draw.

## Data Pipeline
1. `NIFTILoader` parses NIFTI, converts typed array, derives dimensions/spacing/size.
2. `VolumeMap.computeTensor()`:
- builds 3D tensor from volume data,
- computes min/max,
- normalizes intensities to `[0, 1]`,
- optional trilinear downscale.
3. `DistanceMap.computeTexture()`:
- computes packed directional distance/skipping map using TFJS WebGL programs,
- uploads as integer `Data3DTexture` (format depends on selected variation).
4. `VolumeMap.computeTexture()` uploads normalized volume as `R16F` `Data3DTexture`.

## Runtime Rendering Model
- Geometry: unit box scaled to physical volume size.
- Material: custom `ShaderMaterial` with GLSL3, chunk-composed fragment shader.
- Ray setup: computed every frame from camera direction in object/index space.
- Marching/skipping behavior driven by shader defines and uniforms from `Configs`.

## Important Configs
- `blockSize`
- `downscaleEnabled`, `downscaleFactor`
- `marchingMethod` (`cells`, `traces`)
- `skippingMethod` (`shadow`, `distance`)
- `distanceVariation` (`1bit`, `5bit`, `8bit`, `10bit`)
- `skippingEnabled`, `debugEnabled`
- `colormap`

## Key Files To Touch For Common Tasks
- TensorFlow/WebGL flags: `src/javascript/tensorflow.js`
- Global config state: `src/javascript/Configs.js`
- GUI controls and binding: `src/javascript/GUI.js`
- Compute orchestration: `src/javascript/Computes/Computes.js`
- Volume preprocessing: `src/javascript/Computes/Maps/VolumeMap.js`
- Distance map generation: `src/javascript/Computes/Maps/DistanceMap.js`
- Material setup: `src/javascript/World/MIPViewer/MIPMaterial.js`
- Per-frame ray uniforms: `src/javascript/World/MIPViewer/RaymarchUniforms.js`
- Main fragment shader entry: `src/shaders/mip_viewer/fragment.glsl`

## Repository Shape (Practical)
- Shader-heavy codebase (`src/shaders` is the largest component).
- App/runtime logic in `src/javascript`.
- Experimental/math derivations in `src/matlab`.
- Static datasets and papers in `static`.

## Known Notes / Caveats
- Project expects desktop WebGL2; mobile is not targeted.
- Some texture upload formats use raw WebGL enum strings for compatibility in this Three.js version.
- `Computes.change()` is currently commented out in `Experience.change()` (world updates still occur).
- In `Computes.destroy()`, `shadowMap` is referenced while shadow map creation is currently disabled; revisit before relying on destroy path.

## Local Commands
- Install: `npm install`
- Dev: `npm run dev`
- Build: `npm run build`

## Maintenance Tip
- Keep this file updated when changing architecture, pipeline stages, or config semantics.
