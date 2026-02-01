import * as tf from '@tensorflow/tfjs';
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math';
import { getCoordsDataType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

/**
 * -----------------------------
 *  Program 1: stack raw slices
 * -----------------------------
 *
 * Batched only:
 *   inputs:  D tensors of shape [B,H,W,2,2]
 *   output:  [B,D,H,W,2,2]
 *
 * This is great for small D, but NOT for huge D (e.g. 200), because it produces
 * a big shader + needs D texture samplers.
 */
export class StackDepthPackedSlicesBatchedProgram implements GPGPUProgram 
{
  variableNames: string[];
  packedInputs = true;
  packedOutput = true;

  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(outputShape: number[], numSlices: number) 
  {
    if (outputShape.length !== 6) {
      throw new Error(
        `StackDepthPackedSlicesProgram expects outputShape rank 6 [B,D,H,W,2,2], got ${outputShape}.`,
      );
    }

    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    // Name inputs S0, S1, ... S{D-1}
    this.variableNames = Array.from({ length: numSlices }, (_, i) => `S${i}`);

    const outRank = outputShape.length; // 6
    const dtype = getCoordsDataType(outRank);

    const selectSliceCode = () => 
    {
      const lines: string[] = [];
      for (let i = 0; i < numSlices; i++) 
      {
        const cond = i === 0 ? `if (d == ${i})` : `else if (d == ${i})`;

        // slice input: [B,H,W,2,2] => getS{i}(b,h,w,r,c)
        lines.push(`${cond} { v = getS${i}(b, h, w, r, c); }`);
      }
      lines.push(`else { v = vec4(0.); }`);
      return lines.join('\n        ');
    };

    // output rank 6: [B,D,H,W,2,2]
    this.userCode = `
    void main() 
    {
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
  }
}

/**
 * --------------------------------------------
 *  Program 2: concat/stack "depth blocks"
 * --------------------------------------------
 *
 * Batched blocks only:
 *   inputs:  N tensors of shape [B,Di,H,W,2,2]
 *   output:  shape [B,D ,H,W,2,2] where D = sum(Di)
 *
 * N is small (like 8–16), so this shader stays small and within sampler limits.
 */
export class StackDepthPackedBlocksBatchedProgram implements GPGPUProgram 
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
  ) {
    if (outputShape.length !== 6) {
      throw new Error(
        `StackDepthPackedBlocksProgram expects outputShape rank 6 [B,D,H,W,2,2], got ${outputShape}.`,
      );
    }

    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    const numBlocks = blockDepths.length;
    this.variableNames = Array.from({ length: numBlocks }, (_, i) => `B${i}`);

    const outRank = outputShape.length; // 6
    const dtype = getCoordsDataType(outRank);

    // Build prefix thresholds: t0 = D0, t1 = D0+D1, ...
    const thresholds: number[] = [];
    let acc = 0;
    for (const d of blockDepths) 
    {
      acc += d;
      thresholds.push(acc);
    }

    const selectBlockCode = () => 
    {
      const lines: string[] = [];
      let prev = 0;

      for (let i = 0; i < numBlocks; i++) 
      {
        const t = thresholds[i];
        const cond = i === 0 ? `if (d < ${t})` : `else if (d < ${t})`;
        const offsetExpr = prev === 0 ? `d` : `(d - ${prev})`;

        // block input: [B,Di,H,W,2,2] => getB{i}(b, localD, h, w, r, c)
        lines.push(`${cond} { v = getB${i}(b, ${offsetExpr}, h, w, r, c); }`);

        prev = t;
      }

      lines.push(`else { v = vec4(0.); }`);
      return lines.join('\n        ');
    };

    // output rank 6: [B,D,H,W,2,2]
    this.userCode = `
    void main() 
    {
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
  }
}

/**
 * Single-pass stack of raw slices (good only for small D).
 *
 * Batched only:
 *   inputs: D tensors [B,H,W,2,2]
 *   output: [B,D,H,W,2,2]
 */
export function stackDepthPackedSlices(slices: tf.Tensor[]): tf.Tensor 
{
  if (slices.length === 0) throw new Error('stackDepthPackedSlices: slices is empty.');

  const D = slices.length;
  const rank = slices[0].rank;

  // Batched slices must be rank 5: [B,H,W,2,2]
  if (rank !== 5) {
    throw new Error(
      `Expected slice rank 5 ([B,H,W,2,2]) only. Got rank=${rank} shape=${slices[0].shape}.`,
    );
  }

  // Check all ranks match
  for (let i = 1; i < D; i++) {
    if (slices[i].rank !== rank) {
      throw new Error(`All slices must have same rank. Got ${rank} and ${slices[i].rank}.`);
    }
  }

  // Validate trailing [2,2] and consistent shapes/dtypes
  const firstShape = slices[0].shape.slice(); // [B,H,W,2,2]
  const lastA = firstShape[firstShape.length - 2];
  const lastB = firstShape[firstShape.length - 1];
  if (lastA !== 2 || lastB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${lastA},${lastB}] on slice 0.`);
  }

  const dtype = slices[0].dtype;
  for (let i = 0; i < D; i++) 
  {
    const s = slices[i];
    if (s.dtype !== dtype) {
      throw new Error(`All slices must share dtype. slice[0]=${dtype}, slice[${i}]=${s.dtype}.`);
    }
    if (
      s.shape.length !== firstShape.length ||
      !s.shape.every((v, idx) => v === firstShape[idx])
    ) {
      throw new Error(
        `All slices must share shape. slice[0]=${firstShape}, slice[${i}]=${s.shape}.`,
      );
    }
  }

  // Compute output shape: [B,D,H,W,2,2]
  const B = firstShape[0];
  const H = firstShape[1];
  const W = firstShape[2];
  const outShape: number[] = [B, D, H, W, 2, 2];

  const backend: any = tf.backend();
  const prog = new StackDepthPackedSlicesBatchedProgram(outShape, D);

  const info = backend.runWebGLProgram(prog, slices, dtype, null, true);
  return tf.engine().makeTensorFromTensorInfo(info);
}

/**
 * Single-pass concat of a small number of depth-blocks (good when N is small).
 *
 * Batched only:
 *   inputs:  N tensors [B,Di,H,W,2,2]
 *   output:  [B,D ,H,W,2,2]
 */
export function stackDepthPackedBlocks(blocks: tf.Tensor[]): tf.Tensor 
{
  if (blocks.length === 0) throw new Error('stackDepthPackedBlocks: blocks is empty.');

  const N = blocks.length;
  const rank = blocks[0].rank;

  // Batched blocks rank 6: [B,Di,H,W,2,2]
  if (rank !== 6) {
    throw new Error(
      `Expected block rank 6 ([B,Di,H,W,2,2]) only. Got rank=${rank} shape=${blocks[0].shape}.`,
    );
  }

  const dtype = blocks[0].dtype;
  const base = blocks[0].shape.slice(); // [B,D0,H,W,2,2]

  const lastA = base[base.length - 2];
  const lastB = base[base.length - 1];
  if (lastA !== 2 || lastB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${lastA},${lastB}] on block 0.`);
  }

  // Validate consistent B,H,W; depth Di may vary
  const B = base[0];
  const H = base[2];
  const W = base[3];

  const blockDepths: number[] = [];
  for (let i = 0; i < N; i++) 
  {
    const t = blocks[i];
    if (t.rank !== rank) {
      throw new Error(
        `All blocks must share rank. block[0]=${rank}, block[${i}]=${t.rank}.`,
      );
    }
    if (t.dtype !== dtype) {
      throw new Error(
        `All blocks must share dtype. block[0]=${dtype}, block[${i}]=${t.dtype}.`,
      );
    }

    const s = t.shape;
    const a = s[s.length - 2];
    const b = s[s.length - 1];
    if (a !== 2 || b !== 2) {
      throw new Error(`Expected trailing [2,2] on block[${i}], got [${a},${b}].`);
    }

    if (s.length !== 6) throw new Error(`block[${i}] expected rank 6, got shape=${s}`);
    if (s[0] !== B || s[2] !== H || s[3] !== W) {
      throw new Error(
        `Inconsistent block[${i}] shape. Expected [B,Di,H,W,2,2] with B=${B},H=${H},W=${W}, got ${s}.`,
      );
    }

    blockDepths.push(s[1]); // Di
  }

  const D = blockDepths.reduce((a, b) => a + b, 0);
  const outShape: number[] = [B, D, H, W, 2, 2];

  const backend: any = tf.backend();
  const prog = new StackDepthPackedBlocksBatchedProgram(outShape, blockDepths);

  const info = backend.runWebGLProgram(prog, blocks, dtype, null, true);
  return tf.engine().makeTensorFromTensorInfo(info);
}

/**
 * --------------------------------------------
 *  Multi-pass (chunk + tree-reduce) entrypoint
 * --------------------------------------------
 *
 * Batched only:
 *   slices: D tensors [B,H,W,2,2]
 *   output: [B,D,H,W,2,2]
 *
 * Scales to large D:
 *   - Pass 1 stacks raw slices into depth-blocks of size K
 *   - Then repeatedly concatenates those blocks until one remains
 */
export function stackDepthPacked(slices: tf.Tensor[], chunkSize?: number): tf.Tensor 
{
  if (slices.length === 0) throw new Error('stackDepthPacked: slices is empty.');

  const K = chunkSize ?? 16;

  // If small enough, do it in one pass
  if (slices.length <= K) return stackDepthPackedSlices(slices);

  // Pass 1: raw slices -> blocks
  let level: tf.Tensor[] = [];
  for (let i = 0; i < slices.length; i += K) 
  {
    const chunk = slices.slice(i, i + K);
    const block = stackDepthPackedSlices(chunk); // always rank 6 now
    level.push(block);
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
    }

    // Dispose previous level blocks (they were created by us).
    for (const t of level) 
    {
      if (!next.includes(t)) t.dispose();
    }

    level = next;
  }

  // Final output
  return level[0];
}
