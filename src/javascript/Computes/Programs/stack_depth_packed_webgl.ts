

import * as tf from '@tensorflow/tfjs';
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math';
import {getCoordsDataType} from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

/**
 * -----------------------------
 *  Program 1: stack raw slices
 * -----------------------------
 *
 * Batched:
 *   inputs:  D tensors of shape [B,H,W,2,2]
 *   output:  [B,D,H,W,2,2]
 *
 * Unbatched:
 *   inputs:  D tensors of shape [H,W,2,2]
 *   output:  [D,H,W,2,2]
 *
 * This is great for small D, but NOT for huge D (e.g. 200), because it produces
 * a big shader + needs D texture samplers.
 */
export class StackDepthPackedProgram implements GPGPUProgram 
{
  variableNames: string[];
  packedInputs = true;
  packedOutput = true;

  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(outputShape: number[], numSlices: number, hasBatch: boolean) {
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    // Name inputs S0, S1, ... S{D-1}
    this.variableNames = Array.from({length: numSlices}, (_, i) => `S${i}`);

    const outRank = outputShape.length;
    const dtype = getCoordsDataType(outRank);

    const selectSliceCode = () => {
      const lines: string[] = [];
      for (let i = 0; i < numSlices; i++) {
        const cond = i === 0 ? `if (d == ${i})` : `else if (d == ${i})`;
        if (hasBatch) {
          // slice input: [B,H,W,2,2] => getS{i}(b,h,w,r,c)
          lines.push(`${cond} { v = getS${i}(b, h, w, r, c); }`);
        } else {
          // slice input: [H,W,2,2] => getS{i}(h,w,r,c)
          lines.push(`${cond} { v = getS${i}(h, w, r, c); }`);
        }
      }
      lines.push(`else { v = vec4(0.); }`);
      return lines.join('\n        ');
    };

    if (hasBatch) {
      // output rank 6: [B,D,H,W,2,2]
      this.userCode = `
      void main() {
        ${dtype} outC = getOutputCoords();
        int b = outC.x;
        int d = outC.y;
        int h = outC.z;
        int w = outC.w;
        int r = outC.u;
        int c = outC.v;

        vec4 v;
        ${selectSliceCode()}
        setOutput(v);
      }`;
    } else {
      // output rank 5: [D,H,W,2,2]
      this.userCode = `
      void main() {
        ${dtype} outC = getOutputCoords();
        int d = outC.x;
        int h = outC.y;
        int w = outC.z;
        int r = outC.w;
        int c = outC.u;

        vec4 v;
        ${selectSliceCode()}
        setOutput(v);
      }`;
    }
  }
}

/**
 * --------------------------------------------
 *  Program 2: concat/stack "depth blocks"
 * --------------------------------------------
 *
 * Inputs are already "stacked blocks" with a depth dimension:
 *
 * Batched blocks:
 *   inputs:  N tensors of shape [B,Di,H,W,2,2]
 *   output:           shape [B,D ,H,W,2,2] where D = sum(Di)
 *
 * Unbatched blocks:
 *   inputs:  N tensors of shape [Di,H,W,2,2]
 *   output:           shape [D ,H,W,2,2]
 *
 * N is small (like 8–16), so this shader stays small and within sampler limits.
 */
export class StackDepthPackedBlocksProgram implements GPGPUProgram 
{
  variableNames: string[];
  packedInputs = true;
  packedOutput = true;

  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(
    outputShape: number[],
    blockDepths: number[], // [D0, D1, ...]
    hasBatch: boolean,
  ) {
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    const numBlocks = blockDepths.length;
    this.variableNames = Array.from({length: numBlocks}, (_, i) => `B${i}`);

    const outRank = outputShape.length;
    const dtype = getCoordsDataType(outRank);

    // Build prefix thresholds: t0 = D0, t1 = D0+D1, ...
    const thresholds: number[] = [];
    let acc = 0;
    for (const d of blockDepths) {
      acc += d;
      thresholds.push(acc);
    }

    const selectBlockCode = () => {
      const lines: string[] = [];
      let prev = 0;
      for (let i = 0; i < numBlocks; i++) {
        const t = thresholds[i];
        const cond = i === 0 ? `if (d < ${t})` : `else if (d < ${t})`;
        const offsetExpr = prev === 0 ? `d` : `(d - ${prev})`;

        if (hasBatch) {
          // block input: [B,Di,H,W,2,2] => getB{i}(b, localD, h, w, r, c)
          lines.push(`${cond} { v = getB${i}(b, ${offsetExpr}, h, w, r, c); }`);
        } else {
          // block input: [Di,H,W,2,2] => getB{i}(localD, h, w, r, c)
          lines.push(`${cond} { v = getB${i}(${offsetExpr}, h, w, r, c); }`);
        }
        prev = t;
      }
      lines.push(`else { v = vec4(0.); }`);
      return lines.join('\n        ');
    };

    if (hasBatch) {
      // output rank 6: [B,D,H,W,2,2]
      this.userCode = `
      void main() {
        ${dtype} outC = getOutputCoords();
        int b = outC.x;
        int d = outC.y;
        int h = outC.z;
        int w = outC.w;
        int r = outC.u;
        int c = outC.v;

        vec4 v;
        ${selectBlockCode()}
        setOutput(v);
      }`;
    } else {
      // output rank 5: [D,H,W,2,2]
      this.userCode = `
      void main() {
        ${dtype} outC = getOutputCoords();
        int d = outC.x;
        int h = outC.y;
        int w = outC.z;
        int r = outC.w;
        int c = outC.u;

        vec4 v;
        ${selectBlockCode()}
        setOutput(v);
      }`;
    }
  }
}

/**
 * Single-pass stack of raw slices (good only for small D).
 */
export function stackDepthPackedSlices(slices: tf.Tensor[]): tf.Tensor {
  if (slices.length === 0) throw new Error('stackDepthPacked: slices is empty.');

  const D = slices.length;
  const rank = slices[0].rank;

  // Check all ranks match
  for (let i = 1; i < D; i++) {
    if (slices[i].rank !== rank) {
      throw new Error(`All slices must have same rank. Got ${rank} and ${slices[i].rank}.`);
    }
  }

  // Determine batched vs unbatched
  const hasBatch = rank === 5; // [B,H,W,2,2]
  const unbatched = rank === 4; // [H,W,2,2]
  if (!hasBatch && !unbatched) {
    throw new Error(
      `Expected slice rank 5 ([B,H,W,2,2]) or rank 4 ([H,W,2,2]). Got rank=${rank}.`,
    );
  }

  // Validate trailing [2,2] and consistent shapes/dtypes
  const firstShape = slices[0].shape.slice();
  const twoA = firstShape[firstShape.length - 2];
  const twoB = firstShape[firstShape.length - 1];
  if (twoA !== 2 || twoB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${twoA},${twoB}] on slice 0.`);
  }

  const dtype = slices[0].dtype;
  for (let i = 0; i < D; i++) {
    const s = slices[i];
    if (s.dtype !== dtype) {
      throw new Error(
        `All slices must share dtype. slice[0]=${dtype}, slice[${i}]=${s.dtype}.`,
      );
    }
    if (
      s.shape.length !== firstShape.length ||
      !s.shape.every((v, idx) => v === firstShape[idx])
    ) {
      throw new Error(`All slices must share shape. slice[0]=${firstShape}, slice[${i}]=${s.shape}.`);
    }
  }

  // Compute output shape
  let outShape: number[];
  if (hasBatch) {
    const [B, H, W] = firstShape as unknown as [number, number, number, 2, 2];
    outShape = [B, D, H, W, 2, 2];
  } else {
    const [H, W] = firstShape as unknown as [number, number, 2, 2];
    outShape = [D, H, W, 2, 2];
  }

  const backend: any = tf.backend();
  const prog = new StackDepthPackedProgram(outShape, D, hasBatch);

  const info = backend.runWebGLProgram(prog, slices, dtype, null, true);
  return tf.engine().makeTensorFromTensorInfo(info);
}

/**
 * Single-pass concat of a small number of depth-blocks (good when N is small).
 */
export function stackDepthPackedBlocks(blocks: tf.Tensor[]): tf.Tensor {
  if (blocks.length === 0) throw new Error('stackDepthPackedBlocks: blocks is empty.');

  const N = blocks.length;
  const rank = blocks[0].rank;

  // Batched blocks rank 6: [B,Di,H,W,2,2]
  // Unbatched blocks rank 5: [Di,H,W,2,2]
  const hasBatch = rank === 6;
  const unbatched = rank === 5;
  if (!hasBatch && !unbatched) {
    throw new Error(
      `Expected block rank 6 ([B,Di,H,W,2,2]) or rank 5 ([Di,H,W,2,2]). Got rank=${rank}.`,
    );
  }

  // Check ranks/dtypes + trailing [2,2]
  const dtype = blocks[0].dtype;
  const base = blocks[0].shape.slice();

  const lastA = base[base.length - 2];
  const lastB = base[base.length - 1];
  if (lastA !== 2 || lastB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${lastA},${lastB}] on block 0.`);
  }

  // Validate consistent B,H,W and trailing [2,2]. Depth Di can vary.
  let B = 0, H = 0, W = 0;
  if (hasBatch) {
    // base: [B,D0,H,W,2,2]
    B = base[0];
    H = base[2];
    W = base[3];
  } else {
    // base: [D0,H,W,2,2]
    H = base[1];
    W = base[2];
  }

  const blockDepths: number[] = [];
  for (let i = 0; i < N; i++) {
    const t = blocks[i];
    if (t.rank !== rank) throw new Error(`All blocks must share rank. block[0]=${rank}, block[${i}]=${t.rank}.`);
    if (t.dtype !== dtype) throw new Error(`All blocks must share dtype. block[0]=${dtype}, block[${i}]=${t.dtype}.`);

    const s = t.shape;
    const a = s[s.length - 2];
    const b = s[s.length - 1];
    if (a !== 2 || b !== 2) throw new Error(`Expected trailing [2,2] on block[${i}], got [${a},${b}].`);

    if (hasBatch) {
      if (s.length !== 6) throw new Error(`block[${i}] expected rank 6, got shape=${s}`);
      if (s[0] !== B || s[2] !== H || s[3] !== W) {
        throw new Error(
          `Inconsistent block[${i}] shape. Expected [B,Di,H,W,2,2] with B=${B},H=${H},W=${W}, got ${s}.`,
        );
      }
      blockDepths.push(s[1]); // Di
    } else {
      if (s.length !== 5) throw new Error(`block[${i}] expected rank 5, got shape=${s}`);
      if (s[1] !== H || s[2] !== W) {
        throw new Error(
          `Inconsistent block[${i}] shape. Expected [Di,H,W,2,2] with H=${H},W=${W}, got ${s}.`,
        );
      }
      blockDepths.push(s[0]); // Di
    }
  }

  const D = blockDepths.reduce((a, b) => a + b, 0);
  const outShape = hasBatch ? [B, D, H, W, 2, 2] : [D, H, W, 2, 2];

  const backend: any = tf.backend();
  const prog = new StackDepthPackedBlocksProgram(outShape, blockDepths, hasBatch);

  const info = backend.runWebGLProgram(prog, blocks, dtype, null, true);
  return tf.engine().makeTensorFromTensorInfo(info);
}

/**
 * --------------------------------------------
 *  Multi-pass (chunk + tree-reduce) entrypoint
 * --------------------------------------------
 *
 * This is the scalable one:
 *   - Pass 1 stacks raw slices into depth-blocks of size K
 *   - Then repeatedly concatenates those blocks until one remains
 *
 * You can call this for 200+ slices safely.
 */

export function stackDepthPacked(slices: tf.Tensor[], chunkSize?: number,): tf.Tensor 
{
  if (slices.length === 0) throw new Error('stackDepthPackedMultiPass: slices is empty.');

  const K = chunkSize ?? 16;

  // If small enough, do it in one pass
  if (slices.length <= K) return stackDepthPackedSlices(slices);
    
  // We’ll build intermediate levels and dispose them as we go (except inputs).
  // To ensure we don’t accidentally dispose user-provided tensors, we only
  // dispose tensors we created (the intermediates).
  const created: tf.Tensor[] = [];

  // Pass 1: raw slices -> blocks
  let level: tf.Tensor[] = [];
  for (let i = 0; i < slices.length; i += K) 
  {
    const chunk = slices.slice(i, i + K);
    const block = stackDepthPackedSlices(chunk); // rank 6 (batched) or 5 (unbatched)
    level.push(block);
    created.push(block);
  }

  // Pass 2+: blocks -> fewer blocks, until one remains
  while (level.length > 1) 
  {
    const next: tf.Tensor[] = [];
    for (let i = 0; i < level.length; i += K) 
    {
      const chunk = level.slice(i, i + K);
      const merged = stackDepthPackedBlocks(chunk);
      next.push(merged);
      created.push(merged);
    }

    // Dispose the previous level blocks (they were created by us).
    // But keep anything that also appears in `next` (it won’t, but safe).
    for (const t of level) 
    {
      if (!next.includes(t)) t.dispose();
    }

    level = next;
  }

  // `level[0]` is final output. We must not dispose it.
  const out = level[0];

  // Remove `out` from created list so we don’t dispose it accidentally.
  for (let i = created.length - 1; i >= 0; i--) 
  {
    if (created[i] === out) created.splice(i, 1);
  }

  return out;
}