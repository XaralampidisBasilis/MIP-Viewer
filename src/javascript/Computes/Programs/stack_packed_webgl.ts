import * as tf from '@tensorflow/tfjs';
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math';
import { getCoordsDataType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

type Axis3 = 0 | 1 | 2;

function assertAxis3(axis: number): asserts axis is Axis3 {
  if (axis !== 0 && axis !== 1 && axis !== 2) {
    throw new Error(`axis must be 0, 1, or 2. Got axis=${axis}.`);
  }
}

function axisName(axis: Axis3) {
  return axis === 0 ? 'depth' : axis === 1 ? 'height' : 'width';
}

/**
 * -----------------------------
 *  Program 1: stack raw slices
 * -----------------------------
 *
 * UNBATCHED ONLY (packed):
 * axis=0 inputs:  K tensors [H,W,2,2]   -> output [K,H,W,2,2]
 * axis=1 inputs:  K tensors [D,W,2,2]   -> output [D,K,W,2,2]
 * axis=2 inputs:  K tensors [D,H,2,2]   -> output [D,H,K,2,2]
 *
 * Good for small K (e.g. <= 16), because it produces a shader with K samplers.
 */
export class StackPackedSlicesUnbatchedProgram implements GPGPUProgram {
  variableNames: string[];
  packedInputs = true;
  packedOutput = true;

  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(outputShape: number[], numSlices: number, axis: Axis3) {
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    // Name inputs S0, S1, ... S{numSlices-1}
    this.variableNames = Array.from({ length: numSlices }, (_, i) => `S${i}`);

    const outRank = outputShape.length; // must be 5: [D,H,W,2,2]
    const dtype = getCoordsDataType(outRank);

    // Which output coordinate chooses the slice?
    const kExpr = axis === 0 ? 'd' : axis === 1 ? 'h' : 'w';

    // How to index into each slice (rank 4), depending on axis:
    // axis=0 slice is [H,W,2,2] -> (h,w,r,c)
    // axis=1 slice is [D,W,2,2] -> (d,w,r,c)
    // axis=2 slice is [D,H,2,2] -> (d,h,r,c)
    const sliceArgs =
      axis === 0 ? 'h, w, r, c'
      : axis === 1 ? 'd, w, r, c'
      : 'd, h, r, c';

    const selectSliceCode = () => {
      const lines: string[] = [];
      for (let i = 0; i < numSlices; i++) {
        const cond = i === 0 ? `if (k == ${i})` : `else if (k == ${i})`;
        lines.push(`${cond} { v = getS${i}(${sliceArgs}); }`);
      }
      lines.push(`else { v = vec4(0.); }`);
      return lines.join('\n        ');
    };

    this.userCode = `
    void main()
    {
      ${dtype} outC = getOutputCoords();
      int d = outC.x;
      int h = outC.y;
      int w = outC.z;
      int r = outC.w;
      int c = outC.u;

      int k = ${kExpr};

      vec4 v;
      ${selectSliceCode()}
      setOutput(v);
    }`;
  }
}

/**
 * --------------------------------------------
 *  Program 2: concat/stack "axis blocks"
 * --------------------------------------------
 *
 * UNBATCHED ONLY (packed):
 * axis=0 inputs: N tensors [Di,H,W,2,2] -> output [sumDi,H,W,2,2]
 * axis=1 inputs: N tensors [D,Hi,W,2,2] -> output [D,sumHi,W,2,2]
 * axis=2 inputs: N tensors [D,H,Wi,2,2] -> output [D,H,sumWi,2,2]
 *
 * N is small (like 8–16), so shader stays small and within sampler limits.
 */
export class StackPackedBlocksUnbatchedProgram implements GPGPUProgram {
  variableNames: string[];
  packedInputs = true;
  packedOutput = true;

  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(
    outputShape: number[],
    blockSizes: number[], // sizes along chosen axis: [S0, S1, ...]
    axis: Axis3,
  ) {
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    const numBlocks = blockSizes.length;
    this.variableNames = Array.from({ length: numBlocks }, (_, i) => `B${i}`);

    const outRank = outputShape.length; // must be 5
    const dtype = getCoordsDataType(outRank);

    const kExpr = axis === 0 ? 'd' : axis === 1 ? 'h' : 'w';

    // Build thresholds: t0 = S0, t1 = S0+S1, ...
    const thresholds: number[] = [];
    let acc = 0;
    for (const s of blockSizes) {
      acc += s;
      thresholds.push(acc);
    }

    // Build args to getB{i}(...) depending on axis
    const blockCallArgs = (localK: string) =>
      axis === 0 ? `${localK}, h, w, r, c`
      : axis === 1 ? `d, ${localK}, w, r, c`
      : `d, h, ${localK}, r, c`;

    const selectBlockCode = () => {
      const lines: string[] = [];
      let prev = 0;

      for (let i = 0; i < numBlocks; i++) {
        const t = thresholds[i];
        const cond = i === 0 ? `if (k < ${t})` : `else if (k < ${t})`;
        const localK = prev === 0 ? `k` : `(k - ${prev})`;

        lines.push(`${cond} { v = getB${i}(${blockCallArgs(localK)}); }`);
        prev = t;
      }

      lines.push(`else { v = vec4(0.); }`);
      return lines.join('\n        ');
    };

    this.userCode = `
    void main()
    {
      ${dtype} outC = getOutputCoords();
      int d = outC.x;
      int h = outC.y;
      int w = outC.z;
      int r = outC.w;
      int c = outC.u;

      int k = ${kExpr};

      vec4 v;
      ${selectBlockCode()}
      setOutput(v);
    }`;
  }
}

/**
 * Single-pass stack of raw slices (UNBATCHED ONLY).
 */
export function stackPackedSlices(slices: tf.Tensor[], axis: Axis3 = 0): tf.Tensor {
  if (slices.length === 0) throw new Error('stackPackedSlices: slices is empty.');
  assertAxis3(axis);

  const K = slices.length;
  const rank = slices[0].rank;

  // slices must be rank-4
  if (rank !== 4) {
    throw new Error(`Expected slice rank 4. Got rank=${rank}.`);
  }

  // Validate shape/dtype consistency + trailing [2,2]
  const firstShape = slices[0].shape.slice(); // rank 4
  const twoA = firstShape[2];
  const twoB = firstShape[3];
  if (twoA !== 2 || twoB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${twoA},${twoB}] on slice 0.`);
  }

  const dtype = slices[0].dtype;
  for (let i = 0; i < K; i++) {
    const s = slices[i];
    if (s.rank !== 4) throw new Error(`slice[${i}] expected rank 4, got rank=${s.rank}.`);
    if (s.dtype !== dtype) {
      throw new Error(`All slices must share dtype. slice[0]=${dtype}, slice[${i}]=${s.dtype}.`);
    }
    if (s.shape.length !== 4 || !s.shape.every((v, idx) => v === firstShape[idx])) {
      throw new Error(
        `All slices must share shape. slice[0]=${firstShape}, slice[${i}]=${s.shape}.`,
      );
    }
  }

  // Infer output [D,H,W,2,2] from axis + slice shape + K
  let D: number, H: number, W: number;

  if (axis === 0) {
    // slices are [H,W,2,2]
    H = firstShape[0];
    W = firstShape[1];
    D = K;
  } else if (axis === 1) {
    // slices are [D,W,2,2]
    D = firstShape[0];
    W = firstShape[1];
    H = K;
  } else {
    // axis === 2, slices are [D,H,2,2]
    D = firstShape[0];
    H = firstShape[1];
    W = K;
  }

  const outShape: number[] = [D, H, W, 2, 2];

  const backend: any = tf.backend();
  const prog = new StackPackedSlicesUnbatchedProgram(outShape, K, axis);

  const info = backend.runWebGLProgram(prog, slices, dtype, null, true);
  return tf.engine().makeTensorFromTensorInfo(info);
}

/**
 * Single-pass concat of a small number of axis-blocks (UNBATCHED ONLY).
 */
export function stackPackedBlocks(blocks: tf.Tensor[], axis: Axis3 = 0): tf.Tensor {
  if (blocks.length === 0) throw new Error('stackPackedBlocks: blocks is empty.');
  assertAxis3(axis);

  const N = blocks.length;
  const rank = blocks[0].rank;

  // blocks must be rank-5: [D,H,W,2,2]
  if (rank !== 5) {
    throw new Error(`Expected block rank 5 ([D,H,W,2,2]). Got rank=${rank}.`);
  }

  const dtype = blocks[0].dtype;
  const base = blocks[0].shape.slice(); // [*,*,*,2,2]
  if (base.length !== 5) {
    throw new Error(`block[0] expected rank 5, got shape=${base}.`);
  }

  const lastA = base[3];
  const lastB = base[4];
  if (lastA !== 2 || lastB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${lastA},${lastB}] on block 0.`);
  }

  // dims excluding axis must match across blocks
  const baseD = base[0];
  const baseH = base[1];
  const baseW = base[2];

  const blockSizes: number[] = [];
  for (let i = 0; i < N; i++) {
    const t = blocks[i];
    if (t.rank !== 5) throw new Error(`block[${i}] expected rank 5, got rank=${t.rank}.`);
    if (t.dtype !== dtype) {
      throw new Error(`All blocks must share dtype. block[0]=${dtype}, block[${i}]=${t.dtype}.`);
    }

    const s = t.shape; // [D,H,W,2,2]
    if (s.length !== 5) throw new Error(`block[${i}] expected rank 5, got shape=${s}.`);
    if (s[3] !== 2 || s[4] !== 2) {
      throw new Error(`Expected trailing [2,2] on block[${i}], got [${s[3]},${s[4]}].`);
    }

    if (axis === 0) {
      // [Di,H,W,2,2] -> match H,W
      if (s[1] !== baseH || s[2] !== baseW) {
        throw new Error(
          `Inconsistent block[${i}] shape for axis=0. Expected [Di,${baseH},${baseW},2,2], got ${s}.`,
        );
      }
    } else if (axis === 1) {
      // [D,Hi,W,2,2] -> match D,W
      if (s[0] !== baseD || s[2] !== baseW) {
        throw new Error(
          `Inconsistent block[${i}] shape for axis=1. Expected [${baseD},Hi,${baseW},2,2], got ${s}.`,
        );
      }
    } else {
      // axis === 2, [D,H,Wi,2,2] -> match D,H
      if (s[0] !== baseD || s[1] !== baseH) {
        throw new Error(
          `Inconsistent block[${i}] shape for axis=2. Expected [${baseD},${baseH},Wi,2,2], got ${s}.`,
        );
      }
    }

    blockSizes.push(s[axis]);
  }

  const total = blockSizes.reduce((a, b) => a + b, 0);

  let outShape: number[];
  if (axis === 0) outShape = [total, baseH, baseW, 2, 2];
  else if (axis === 1) outShape = [baseD, total, baseW, 2, 2];
  else outShape = [baseD, baseH, total, 2, 2];

  const backend: any = tf.backend();
  const prog = new StackPackedBlocksUnbatchedProgram(outShape, blockSizes, axis);

  const info = backend.runWebGLProgram(prog, blocks, dtype, null, true);
  return tf.engine().makeTensorFromTensorInfo(info);
}

/**
 * --------------------------------------------
 *  Multi-pass (chunk + tree-reduce) entrypoint
 * --------------------------------------------
 *
 * UNBATCHED ONLY (packed):
 * axis=0: slices [H,W,2,2] -> out [D,H,W,2,2]
 * axis=1: slices [D,W,2,2] -> out [D,H,W,2,2]
 * axis=2: slices [D,H,2,2] -> out [D,H,W,2,2]
 */
export function stackPacked(slices: tf.Tensor[], axis: Axis3 = 0, chunkSize?: number): tf.Tensor {
  if (slices.length === 0) throw new Error('stackPacked: slices is empty.');
  assertAxis3(axis);

  const K = chunkSize ?? 16;

  // If small enough, do it in one pass
  if (slices.length <= K) return stackPackedSlices(slices, axis);

  // Pass 1: raw slices -> blocks
  let level: tf.Tensor[] = [];
  for (let i = 0; i < slices.length; i += K) {
    const chunk = slices.slice(i, i + K);
    const block = stackPackedSlices(chunk, axis); // rank-5 block along chosen axis
    level.push(block);
  }

  // Pass 2+: blocks -> fewer blocks, until one remains
  while (level.length > 1) {
    const next: tf.Tensor[] = [];
    for (let i = 0; i < level.length; i += K) {
      const chunk = level.slice(i, i + K);
      const merged = stackPackedBlocks(chunk, axis);
      next.push(merged);
    }

    // Safe to dispose: created by us in this function.
    for (const t of level) t.dispose();
    level = next;
  }

  return level[0];
}
