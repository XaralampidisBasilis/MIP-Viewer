# Project Summary: MIP-Viewer

## Overview

MIP-Viewer is a browser-based medical volume visualization project focused on interactive maximum intensity projection (MIP) rendering for 3D NIFTI datasets. It combines a real-time Three.js/WebGL2 renderer with TensorFlow.js preprocessing and a large custom GLSL shader pipeline to explore high-quality volume rendering, block-based acceleration, and rich internal debugging views.

At a practical level, the project loads compressed `.nii.gz` volumes, prepares them for GPU rendering, computes auxiliary acceleration textures, and then renders the dataset inside a shader-driven raymarching viewer. The repository is not just a viewer application; it is also a research and experimentation workspace that includes shader utilities, GPGPU programs, mathematical derivations, example datasets, and reference papers related to volume rendering and interpolation.

## What The Project Does

The project is built around the idea of rendering volumetric medical data directly in the browser without relying on a native application. Its main responsibilities are:

- Loading medical volumes stored in the NIFTI format.
- Parsing voxel intensities and metadata such as dimensions and spacing.
- Converting spatial units and deriving the physical size of the dataset.
- Normalizing intensity values into a shader-friendly range.
- Optionally downscaling the volume before rendering.
- Computing preprocessed distance or skipping maps to accelerate ray traversal.
- Uploading the processed data into GPU `Data3DTexture` objects.
- Rendering the volume through a custom GLSL MIP raymarcher.
- Exposing runtime controls for rendering mode, skipping mode, colormap, and debug output.

## Main Technical Idea

The viewer uses a two-stage approach:

1. Preprocess the volume on the GPU with TensorFlow.js running on the WebGL backend.
2. Render the volume with a custom fragment shader that raycasts a box enclosing the dataset.

This separation is one of the defining characteristics of the project. TensorFlow.js is not only used for machine learning here; it acts as a convenient browser-side GPGPU layer for tensor operations, resizing, normalization, and generation of packed acceleration textures. Three.js then takes over as the rendering framework and hosts the final shader material and 3D textures.

## End-To-End Pipeline

### 1. Application startup

The app starts in `src/script.js`. Before anything else, it initializes TensorFlow.js with the WebGL backend through `src/javascript/tensorflow.js`. The code explicitly enables WebGL2-oriented flags such as packed execution and forced half-float textures.

Once TensorFlow is ready, the application creates an `Experience` instance. `Experience` acts as the composition root and lifecycle coordinator for the whole app.

### 2. Resource loading

Resources are loaded through `src/javascript/Utils/Resources.js`. The currently configured source list lives in `src/javascript/sources.js`, where the active default dataset is set to:

- `static/nifti/cardiac/CTACardio.nii.gz`

Additional example datasets are also present in the repository, including cardiac, head, and colon data.

### 3. NIFTI parsing

`src/javascript/Utils/NIFTILoader.js` handles NIFTI loading and parsing. It:

- Decompresses `.nii.gz` files when needed.
- Reads the NIFTI header and raw image payload.
- Converts the payload to the correct typed array based on datatype.
- Extracts dimensions and spacing.
- Converts spatial units to meters.
- Computes the physical size of the volume from dimensions and spacing.

The result is a `Volume` object that downstream code can use both for preprocessing and rendering.

### 4. Volume preprocessing

The preprocessing stage is orchestrated by `src/javascript/Computes/Computes.js`.

The volume path is:

- `VolumeMap.computeTensor()`
- `DistanceMap.computeTexture()`
- `VolumeMap.computeTexture()`

`src/javascript/Computes/Maps/VolumeMap.js` is responsible for:

- Converting the raw volume array into a TensorFlow 3D tensor.
- Computing the minimum and maximum intensity values.
- Normalizing intensities to the `[0, 1]` range.
- Optionally resizing the volume with trilinear interpolation.
- Converting the final tensor into half-float texture data.
- Uploading the normalized volume as a `Data3DTexture` with `R16F` storage.

### 5. Acceleration texture generation

`src/javascript/Computes/Maps/DistanceMap.js` creates a precomputed block-level distance map used for skipping empty or non-promising regions during ray traversal.

The current implementation supports multiple packed storage variants:

- `1bit`
- `5bit`
- `8bit`
- `10bit`

These variations map to different integer texture layouts such as `R16UI`, `RGBA16UI`, `RGB32UI`, and `RGBA32UI`. The project includes custom GPGPU programs for generating these textures, and some upload paths use raw WebGL enum strings for compatibility with the current Three.js version in use.

### 6. Scene creation and rendering

`src/javascript/World/World.js` creates the main scene object, and `src/javascript/World/MIPViewer/MIPViewer.js` builds the actual renderable volume viewer.

The viewer:

- Creates a unit cube mesh.
- Scales that cube to the physical size of the dataset.
- Attaches a custom `ShaderMaterial`.
- Injects volume and distance textures into the shader uniforms.
- Configures shader defines based on runtime settings.

The final rendering is performed by a fragment shader in `src/shaders/mip_viewer/fragment.glsl`. That shader is assembled from many reusable GLSL chunks covering:

- ray setup
- intersections
- block and cell marching
- cubic maximization
- shading
- debug output
- math and polynomial solvers

### 7. Per-frame updates

Every frame, the app updates:

- camera state
- raymarch-related uniforms
- renderer output
- performance stats

`src/javascript/World/MIPViewer/RaymarchUniforms.js` computes the per-frame ray state from the active camera and current mesh transform. This includes direction, inverse direction, dominant axis, quadrant grouping, step distances, and box-relative distances used by the shader.

## Rendering Model

The project renders a volume by raycasting through a box that encloses the dataset. Instead of treating the data as a stack of 2D slices, the fragment shader computes a ray for each visible pixel and marches through the volume in data space.

The main rendering mode is maximum intensity projection, meaning the ray keeps the strongest sample encountered along its path. This is a natural fit for many medical imaging tasks where bright structures such as contrast-enhanced vessels should stand out clearly.

The codebase also experiments with more advanced traversal logic than a naive full-resolution raymarch. It can organize the volume into blocks and cells, precompute directional distance information, and use shader-level control flow to skip regions that do not need full evaluation.

## User Controls And Debugging

The viewer includes a `lil-gui` interface in `src/javascript/GUI.js`. The GUI exposes a small but important runtime control surface for exploring rendering behavior.

Configurable items include:

- marching method
- skipping method
- skipping enabled toggle
- colormap selection
- debug mode selection
- debug-specific limits and temporary variables

The debug system is a major part of the repository. It can visualize internal states such as:

- ray direction and stepping
- block and cell traversal
- trace positions
- MIP update locations
- gradients
- normals
- steepness
- curvature-related outputs
- fetch statistics
- custom debug variables

This makes the project useful not only as a viewer but also as a tool for inspecting and validating rendering algorithms.

## Camera And Interaction Model

The scene uses an orthographic camera together with `ArcballControls`. This gives the viewer a stable inspection workflow for rotating, zooming, and reframing the volume without the distortion of a perspective projection. On startup, the world frames the camera around the dataset bounds so the volume begins fully visible.

## Repository Structure

The repository is organized into a few major areas:

### `src/javascript`

This is the main application layer. It contains:

- app startup and lifecycle management
- configuration state
- camera and renderer setup
- GUI controls
- resource loading
- preprocessing orchestration
- viewer construction and per-frame uniform updates

### `src/shaders`

This is the largest and most specialized part of the project. It contains:

- the main MIP shader entrypoints
- shader chunks for marching and shading
- math helpers
- polynomial and root solvers
- interpolation and filtering utilities
- color maps
- geometric helpers
- debug modules

This directory represents the core rendering logic of the project.

### `src/javascript/Computes`

This area contains the TensorFlow.js and custom WebGL preprocessing layer. It includes:

- volume tensor creation
- normalization
- resizing programs
- map generation
- packed texture conversion

### `src/matlab`

This folder contains mathematical and derivation-oriented scripts related to interpolation, extrema, Bernstein formulations, and ray-volume analysis. It supports the experimental and algorithmic side of the repository.

### `static`

This folder contains:

- NIFTI datasets
- rendered example images
- a Word document
- reference papers used to inform the rendering work

## Major Runtime Components

The most important files for understanding or extending the project are:

- `src/script.js`: browser entrypoint.
- `src/javascript/Experience.js`: central app lifecycle and composition root.
- `src/javascript/Configs.js`: global configuration state and validation.
- `src/javascript/Utils/NIFTILoader.js`: NIFTI parsing and metadata extraction.
- `src/javascript/Computes/Computes.js`: preprocessing orchestration.
- `src/javascript/Computes/Maps/VolumeMap.js`: normalized volume tensor and texture generation.
- `src/javascript/Computes/Maps/DistanceMap.js`: packed acceleration texture generation.
- `src/javascript/World/MIPViewer/MIPViewer.js`: viewer assembly and runtime shader wiring.
- `src/javascript/World/MIPViewer/MIPMaterial.js`: shader material creation, uniforms, and defines.
- `src/javascript/World/MIPViewer/RaymarchUniforms.js`: frame-dependent ray setup.
- `src/shaders/mip_viewer/fragment.glsl`: final fragment shader entrypoint.

## Configuration Surface

The main runtime config object is defined in `src/javascript/Configs.js`. Important settings include:

- `blockSize`
- `downscaleEnabled`
- `downscaleFactor`
- `errorTolerance`
- `distanceVariation`
- `marchingMethod`
- `skippingMethod`
- `skippingEnabled`
- `debugEnabled`
- `colormap`
- `adaptivePixelRatioEnabled`

These values directly affect preprocessing outputs, shader defines, traversal logic, and debug behavior.

## Included Assets And Data

The repository includes several useful assets for development and experimentation:

- medical NIFTI datasets under `static/nifti`
- debug screenshots under `static/images`
- reference rendering papers under `static/paper/references`
- supporting project documentation such as `PROJECT_CONTEXT.md`

This makes the repository relatively self-contained: it includes both the software and much of the supporting material needed to understand why the implementation looks the way it does.

## Build And Run Model

The project uses Vite as the development and build tool:

- `npm install`
- `npm run dev`
- `npm run build`

Notable build characteristics:

- the Vite root is `src/`
- static assets are served from `static/`
- production output goes to `docs/`
- GLSL files are imported through `vite-plugin-glsl`

The `docs/` output directory suggests the project is set up to fit a GitHub Pages-style deployment workflow.

## Practical Characteristics

From the current repository state, the project is best understood as a desktop-focused, GPU-dependent WebGL2 application. Important practical characteristics are:

- It is not intended for mobile devices.
- It expects WebGL2 support.
- It relies heavily on GPU preprocessing and shader execution.
- Load time can be noticeable because the initial tensor preparation and texture generation happen before the viewer becomes fully interactive.
- It is well suited for experimentation with rendering algorithms and debug instrumentation.

## Current Caveats And Implementation Notes

Several code-level details are worth keeping in mind:

- `Experience.change()` currently forwards config changes to the world but has compute-side recomputation commented out.
- `Computes.destroy()` still references `shadowMap` even though shadow-map creation is currently disabled.
- Some integer `Data3DTexture` upload paths use raw WebGL enum strings instead of Three.js constants due to compatibility issues in the current Three.js version.
- The repository contains both active production paths and commented experimental alternatives, so some systems are intentionally in flux.

These do not change the core purpose of the project, but they are important context for anyone maintaining or extending it.

## In Short

MIP-Viewer is a WebGL2 medical volume rendering project that combines:

- NIFTI loading
- TensorFlow.js preprocessing
- Three.js rendering
- custom GLSL raymarching
- block-based acceleration ideas
- extensive internal debugging tools

It serves as both an interactive viewer and a development playground for advanced browser-based MIP rendering techniques.
