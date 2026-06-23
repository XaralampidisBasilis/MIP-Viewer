# WebGPU vs WebGL rendering path differences

## Goal

This file compares the current WebGPU path with the established WebGL path so the WebGPU renderer can be improved to match WebGL output and functionality.

Scope covered:

- Runtime renderer/material selection.
- Fragment shader ray setup, sampling, marching, shading, and debug logic.
- Volume and distance-map resource flow into the renderer.
- Compute-side shadow/distance preprocessing where it affects render output.

## High-level flow

### WebGL path

1. `Renderer` creates a `THREE.WebGLRenderer`.
2. `MIPViewer` chooses `MIPMaterial('webgl')`.
3. `MIPMaterial` creates a `THREE.ShaderMaterial` with:
   - `src/shaders/mip_viewer/vertex.glsl`
   - `src/shaders/mip_viewer/fragment.glsl`
   - GLSL `defines` and Three uniforms.
4. `fragment.glsl` includes:
   - common utils,
   - uniforms,
   - full mutable structs,
   - sampling/functions,
   - raycast,
   - compile-time selected march code,
   - shading,
   - optional debug modules.
5. Volume data is rendered from a `Data3DTexture` (`RedFormat` + `HalfFloatType`, linear filtering; WebGL may expose this as `R16F`).
6. Distance/skipping data is rendered from a packed integer `Data3DTexture`.

### WebGPU path

1. `Renderer` creates a `THREE.WebGPURenderer` when configured and supported.
2. `MIPViewer` chooses `MIPMaterial('webgpu')`.
3. `MIPMaterial` delegates to `MIPWebGPUMaterial`.
4. `MIPWebGPUMaterial` creates a `THREE.NodeMaterial` and supplies a WGSL fragment function through `wgslFn`.
5. WGSL modules are concatenated by `src/shaders/mip_viewer/wgsl/fragment/source.js`.
6. Volume data is sampled as a WebGPU `texture_3d<f32>` with a sampler. Packed distance data is converted to storage-buffer words and read in WGSL.

## Main logic differences

| Area | WebGL path | WebGPU path | Effect / risk |
|---|---|---|---|
| Material side | `BackSide` in `MIPMaterial.js`. | `DoubleSide` in `MIPWebGPUMaterial.js`. | Usually same color because ray origin is screen-derived, but WebGPU can shade extra fragments and may differ in edge/face ordering. Match `BackSide` for parity and performance unless WebGPU requires otherwise. |
| Shader selection | GLSL uses preprocessor `#if` defines for marching/skipping/debug. | WGSL receives method/define values as function parameters and branches at runtime. | Results should be equivalent if branches are complete, but WebGPU compiles one larger shader and cannot rely on GLSL-only unrolling/preprocessor behavior. |
| Vertex shader | Explicit simple GLSL vertex shader. | A WGSL vertex function exists, but the active `NodeMaterial` only installs the custom fragment node. | The default NodeMaterial vertex path likely matches the simple transform, but the custom WGSL vertex code is currently unused. |
| Volume sampling | Hardware `texture(u_textures.volume_map, position * inv_dimensions).r` with single-channel half-float linear filtering. | Hardware `textureSample(volume_map, volume_sampler, position * inv_dimensions).r` from `texture_3d<f32>`. Three derives WebGPU `r16float` from `RedFormat + HalfFloatType`; do not set WebGL-style `internalFormat = 'R16F'`. | Intended to match WebGL. This is a high-priority numeric parity point because any axis order or border handling mismatch changes MIP values. |
| Distance sampling | Integer `texelFetch` from packed `Data3DTexture`; format changes by encoding (`R16UI`, `RGBA16UI`, `RGB32UI`, `RGBA32UI`). | Storage-buffer load using `distance_words_per_voxel`, then the same masks/shifts are reproduced in WGSL. | Logic is close. Needs per-encoding tests, especially `5bit` because WebGPU render sees expanded `Uint16Array` components as u32 words. |
| Ray origin | GLSL derives ray origin from `gl_FragCoord`, inverse projection/view/model, then converts to index coordinates. | WGSL uses `screenCoordinate`, flips Y to GL-style coordinates, then does the same inverse transforms. | This must be screenshot-tested at multiple resolutions/pixel ratios. A half-pixel or Y-flip mismatch would move the whole ray field. |
| Ray direction and reversal | Shared JS uniform code in `RaymarchUniforms.js` computes direction, inverse, dominant axis, quadrant/group, and reverses negative dominant directions. | Same uniform source is passed into WGSL. | Likely equivalent. Keep as shared JS logic. |
| March method dispatch | GLSL includes either `march_cells` or `march_traces` at compile time. Each of those includes skipping/non-skipping variants at compile time. | WGSL `mip_viewer_fragment` branches over `marching_method` and `skipping_enabled` at runtime. | Functionally OK if all branches match; performance may be lower. |
| Cubic MIP mode | WebGL cells mode uses cubic reconstruction, optional Bernstein culling, and `BERNSTEIN_ENABLED` can be toggled. | WGSL cells mode always applies Bernstein culling in `update_cubic_mip`; `BERNSTEIN_ENABLED` is not passed or used. | Functionality mismatch. If the GUI toggles Bernstein off, WebGPU will not match WebGL. |
| Traces MIP mode | WebGL uses `trace` state and updates stats/debug during trace sampling. | WGSL computes trace samples directly and keeps only final MIP state. | Core color can match, but trace debug/stat outputs cannot. |
| MIP end state | WebGL `end_mip.glsl` computes final `mip.position`, gradient, Hessian, normal, and principal curvatures. | WGSL `MipState` only stores value, distance, and position. | Major functionality gap for debug modes `mip_normal`, `mip_gradient`, `mip_steepness`, `mip_curvatures`, and any future shading using derivatives. |
| Shading | Current WebGL `compute_shade.glsl` maps `mip.value` through the colormap. | WGSL maps `mip.value` through a ported colormap function. | Core color path is conceptually equivalent. Colormap coefficient parity should still be tested. |
| Debug | WebGL has modular debug for ray/cell/trace/block/mip/frag/box/stats/cubic/variables using live structs and counters. | WGSL has a single `debug_color` function with many approximations from final `ray`/`mip`, and many options return fallback color/value. | Major functionality gap. WebGPU can show some ray/block/cell/MIP info, but not full WebGL debug behavior. |
| Stats | WebGL tracks counts such as cells, traces, blocks, maxima, fetches. | WGSL has no stats struct/counters. | Debug stats options cannot match. |
| Define coverage | WebGL defines include `BERNSTEIN_ENABLED`, `PRODUCTION_ENABLED`, `VARIATION_ENABLED`, max debug controls, etc. | WebGPU only passes a subset; several GUI toggles do not affect WGSL. | GUI functionality differs by backend. |
| Texture/resource flow | WebGL renders directly from Three `Data3DTexture`s. | WebGPU render path now consumes the volume as a sampled `Data3DTexture`, but still converts packed distance CPU data into a runtime-sized `StorageBufferAttribute` via `syncWebGPUResources`. A tiny address-of node passes this buffer as `ptr<storage, array<u32>, read>` to native WGSL helpers. | Volume filtering now follows the GPU texture path. Distance still adds CPU repacking/readback; longer term, render should consume the WebGPU buffer or a WebGPU-legal integer texture layout directly. |
| Compute backend support | WebGL/TensorFlow distance code supports isotropic, unidirectional, and bidirectional distance transforms in helper functions. Active render path requests unidirectional packed textures. | WebGPU packed distance texture currently throws unless the requested variant is `unidirectional`. | OK for current config, but less complete than WebGL helpers. |
| Shadow map backend | WebGL packed texture path imports `GPGPUShadowMapPaths`. | WebGPU can choose `paths` or `faces` through `webgpuShadowBackend`. | WebGPU has extra alternative behavior. For WebGL parity, use `paths` first and compare `faces` separately. |
| Tensor shape convention | WebGL/TensorFlow tensors are shaped `[depth, height, width]`; GLSL helpers convert to xyz where needed. | Custom WebGPU tensors are shaped `[width, height, depth]`. | This is intentional but risky. Distance/shadow parity tests must cover non-cubic volumes to catch axis-order errors. |

## Result-critical parity points

These are the differences most likely to explain visible mismatches between WebGPU and WebGL:

1. **Ray/screen-coordinate parity**
   - Compare the ray debug modes first: direction, start/end position, dominant axis, quadrant, group, reverse.
   - Test multiple canvas sizes and pixel ratios.
   - WebGPU's Y flip and half-pixel handling must match WebGL's `gl_FragCoord` behavior.

2. **Volume sampler parity**
   - WebGL relies on hardware trilinear filtering of `R16F`.
   - WGSL manually reproduces `texture()` filtering from storage-buffer half-floats.
   - Validate with a tiny known 3D volume and sample positions at voxel centers, boundaries, and fractional positions.

3. **Distance texture parity**
   - For each encoding (`1bit`, `5bit`, `8bit`, `10bit`), compare WebGL `texelFetch` unpacking with WGSL `sample_distance`.
   - Include all dominant axes and all quadrant indices.

4. **Shadow/distance compute parity**
   - Compare packed distance texture data generated by `ComputePackedDistanceTexture.ts` and `WebGPUComputePackedDistanceTexture.js`.
   - Use non-cubic shapes so `[z,y,x]` vs `[x,y,z]` issues are obvious.

5. **Cubic/Bernstein behavior**
   - WebGL has a `BERNSTEIN_ENABLED == 0` branch.
   - WGSL does not.
   - Port this toggle or disable the GUI toggle for WebGPU until it is implemented.

6. **End-MIP derivative data**
   - WebGL computes gradient, Hessian, normal, and curvatures at the selected MIP position.
   - WGSL does not.
   - This is required for matching WebGL debug functionality and any derivative-based shading.

7. **Debug/stat behavior**
   - WebGL debug is tied to live march structs and counters.
   - WGSL debug is a partial final-state reconstruction.
   - Decide whether WebGPU must match every debug mode or whether unsupported modes should be hidden/disabled for WebGPU.

## Recommended WebGPU improvement plan

1. **Lock down numeric parity tests**
   - Add a small deterministic test volume.
   - Compare WebGL and WebGPU generated distance maps for all 12 ray families and all active encodings.
   - Compare final rendered MIP value/distance for selected camera directions.

2. **Make render state match WebGL**
   - Change WebGPU material side to `BackSide` unless a WebGPU-specific issue requires `DoubleSide`.
   - Confirm the default NodeMaterial vertex transform matches `vertex.glsl`, or explicitly wire the WGSL vertex function.

3. **Verify and, if needed, fix WGSL ray origin**
   - Use WebGL debug modes as reference for `ray_start_position`, `ray_end_position`, `ray_direction`, `ray_group_index`, and `ray_reversed`.
   - Fix Y orientation or half-pixel offset before changing march logic.

4. **Port missing define behavior**
   - Add `bernstein_enabled` to `mip_viewer_fragment`.
   - In WGSL `update_cubic_mip`, branch between the Bernstein-cull path and the unconditional cubic maximize path.
   - Audit `PRODUCTION_ENABLED`, `VARIATION_ENABLED`, and debug max controls; either implement or remove/hide them for WebGPU.

5. **Port WebGL end-MIP derivatives**
   - Add WGSL equivalents of:
     - `computeGradientTriquadraticBspline.glsl`
     - `computeSecondDerivatives.glsl`
     - `computePrincipalCurvatures.glsl`
   - Extend `MipState` to carry gradient, Hessian, normal, curvatures, terminated/update flags where needed.

6. **Decide the WebGPU debug target**
   - If full parity is required, port the WebGL structs/counters into WGSL march functions.
   - If only production rendering matters, explicitly document unsupported debug options and disable those GUI entries for WebGPU.

7. **Reduce WebGPU CPU round trips**
   - Today compute produces WebGPU buffers, then data is read back into CPU arrays, then the render path repacks those arrays into storage buffers.
   - Longer term, preserve the packed WebGPU buffer from `computePackedDistanceBufferWebGPU` and bind it directly to the WebGPU material.
   - Do the same for normalized volume data if possible.

8. **Keep one canonical shadow/distance mapping**
   - The WebGL and WebGPU code each encode axis/octant/stencil mapping.
   - Add parity tests or shared generated tables so `paths` WebGPU and WebGL cannot silently diverge.

## WebGPU texture format decision

The current WebGPU render material binds the volume as a sampled texture and the packed distance data as a storage buffer. This is the right immediate split:

- The volume is sampled many times during ray marching.
- The desired operation is hardware trilinear filtering.
- WebGPU can sample a `Data3DTexture` as a `texture_3d<f32>` plus sampler.
- Three maps `RedFormat + HalfFloatType` to WebGPU `r16float`. Do not set `texture.internalFormat = 'R16F'`, because WebGPU receives that invalid GL-style string as the `GPUTextureDescriptor.format`.

The packed distance map is different:

- It is sampled with integer texel semantics, not trilinear filtering.
- It needs exact bit unpacking.
- Keeping it as a storage buffer is functionally reasonable in WebGPU because there is no filtering benefit.
- If it is moved to a texture, the packed formats must be redesigned around WebGPU-supported GPU texture formats and Three's WebGPU node limitations.

Important format constraints:

| Existing packed texture | WebGL intent | WebGPU format status | Decision |
|---|---|---|---|
| `R16UI` | One 16-bit unsigned channel for `1bit`. | WebGPU has `r16uint`. Three maps `RedFormat + UnsignedShortType` to `r16uint`, but integer 3D texture node typing is limited in Three `0.184.0`. | Possible at raw WebGPU level; risky through current TSL/WGSL node path. |
| `RGBA16UI` | Four 16-bit unsigned channels for `5bit`. | WebGPU has `rgba16uint`. Three maps `RGBAFormat + UnsignedShortType` to `rgba16uint`, but sampled `Data3DTexture` bindings are declared as `texture_3d<f32>` by the current node builder. | Possible at raw WebGPU level; risky through current TSL/WGSL node path. |
| `RGB32UI` | Three 32-bit unsigned channels for `8bit`. | WebGPU does **not** have an `rgb32uint` texture format. | Do not use this format in WebGPU. The WebGPU compute path now emits `RGBA32UI` with one unused channel for `8bit`; the WebGL helper can still use `RGB32UI`. |
| `RGBA32UI` | Four 32-bit unsigned channels for `10bit`. | WebGPU has `rgba32uint`. Three maps `RGBAIntegerFormat + UnsignedIntType` to `rgba32uint`, but `Data3DTexture` integer typing is still a blocker in the node path. | Possible at raw WebGPU level; risky through current TSL/WGSL node path. |

Three `0.184.0` local findings:

- `Texture3DNode` exists and is suitable for sampled float 3D textures.
- `WGSLNodeBuilder` declares `Data3DTexture` uniforms as `texture_3d<f32>`.
- `NodeBuilder.getComponentTypeFromTexture()` only treats `DataTexture` with `IntType`/`UnsignedIntType` as integer, not `Data3DTexture`.
- `WebGPUBindingUtils` can set integer sample types for some data textures, but the generated WGSL type for `Data3DTexture` still becomes `texture_3d<f32>`.
- This makes integer 3D textures unsafe through the current `wgslFn`/TSL material path without patching Three or using lower-level WebGPU bindings.

Recommended resource plan for WebGPU functionality:

1. Bind the volume as a `Data3DTexture` with `RedFormat + HalfFloatType` and sample it with hardware trilinear filtering.
2. Keep the packed distance map as a storage buffer for now, since it only needs integer loads and bit shifts.
3. If distance texture binding becomes necessary later, keep the WebGPU path on WebGPU-legal layouts:
   - `1bit`: `r16uint` or `r32uint`
   - `5bit`: `rgba16uint`
   - `8bit`: `rgba32uint` with `.w` unused, or a new repacked layout
   - `10bit`: `rgba32uint`
4. Revisit integer 3D texture binding only after either:
   - using raw WebGPU bind groups outside the current node material path, or
   - confirming/upgrading to a Three version where `Data3DTexture` integer sample types generate `texture_3d<u32>` correctly.

## Current conclusion

The WebGPU path is a real partial port of the WebGL path, not a separate renderer from scratch. The core MIP color path is close in intent: ray setup, block/cell/trace marching, distance skipping, cubic maximization, and colormap mapping all exist.

The main missing functionality is around exact parity and observability: full debug/stat behavior, derivative outputs, the `BERNSTEIN_ENABLED` toggle, and robust proof that WGSL texture sampling and packed-distance reads match WebGL behavior.

The fastest route to matching WebGL is:

1. Prove/fix ray and sampler parity.
2. Prove/fix packed distance parity.
3. Port the missing `BERNSTEIN_ENABLED` and end-MIP derivative logic.
4. Either port or intentionally gate WebGL-only debug/stat modes.
5. Then optimize WebGPU resource flow to avoid CPU readback/repacking.
