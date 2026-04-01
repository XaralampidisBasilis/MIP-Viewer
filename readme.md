# MIP-Viewer

Browser-based maximum intensity projection (MIP) viewer for medical NIFTI volumes, built with Three.js, TensorFlow.js, and custom GLSL shaders.

This project focuses on interactive volume rendering in WebGL2, together with acceleration experiments based on precomputed shadow and distance maps, block-based skipping, and detailed shader debug views.

> ⚠️ **Note:** The demo does **not** run on mobile devices. Please open it on a desktop or laptop with GPU acceleration enabled. It takes around 1 min to load 

<p align="center">
  <img src="./static/images/mip_skeleton.png" alt="Main maximum intensity projection render of the default CTA head and neck dataset" width="100%" />
</p>

<p align="center">
  Default scene: CTA head-and-neck volume rendered in the browser with an interactive MIP pipeline.
</p>

## Highlights

- Interactive maximum intensity projection rendering for 3D medical volumes directly in the browser.
- NIFTI loading pipeline with normalization and optional downscaling before rendering.
- TensorFlow.js preprocessing that generates acceleration textures for the viewer pipeline.
- Configurable marching modes, skipping strategies, skipping methods, gradients, and colormaps.
- Rich debug visualizations for ray state, MIP distance, MIP position, gradients, normals, steepness, and fetch counts.

## Gallery

<p align="center">
  <img src="./static/images/mip_skeleton.png" alt="Main MIP render" width="48%" />
  <img src="./static/images/mip_skeleton_steepness.png" alt="Steepness debug view" width="48%" />
</p>
<p align="center">
  <img src="./static/images/mip_skeleton_position.png" alt="Position debug view" width="48%" />
  <img src="./static/images/mip_skeleton_distance.png" alt="Distance debug view" width="48%" />
</p>
<p align="center">
  <img src="./static/images/mip_skeleton_fetches_2.png" alt="Fetch count debug view two" width="48%" />
  <img src="./static/images/mip_skeleton_fetches_1.png" alt="Fetch count debug view one" width="48%" />
</p>

## Controls

The viewer uses ArcballControls for exploring the volume in 3D.

## Running locally

Install Node.js first, then run:

```bash
npm install
npm run dev
npm run build
```

- `npm run dev` starts the Vite development server.
- `npm run build` outputs the production build to `docs/`, which fits a GitHub Pages workflow.
