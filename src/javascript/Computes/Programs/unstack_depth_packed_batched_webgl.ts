import * as tf from '@tensorflow/tfjs';
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math';
import { getCoordsDataType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

/**
 * Batched-only unstack:
 *   input:  [B, D, H, W, 2, 2]
 *   output: D tensors, each [B, H, W, 2, 2]
 */
export class UnstackDepthPackedBatchedProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = true;

  // Custom uniform: which depth slice to read.
  customUniforms = [{ name: 'uDepth', type: 'int' as const }];

  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(outputShape: number[]) {
    // outputShape must be rank-5: [B, H, W, 2, 2]
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    const outRank = outputShape.length; // 5
    const dtype = getCoordsDataType(outRank);

    this.userCode = `
      void main() {
        ${dtype} outC = getOutputCoords();
        int b = outC.x;
        int h = outC.y;
        int w = outC.z;
        int r = outC.w;
        int c = outC.u;

        vec4 v = getA(b, uDepth, h, w, r, c);
        setOutput(v);
      }
    `;
  }
}

export function unstackDepthPacked(x: tf.Tensor, axis?: number): tf.Tensor[] {
  if (x.rank !== 6) {
    throw new Error(`Batched unstack expects rank-6 [B,D,H,W,2,2]. Got rank=${x.rank}.`);
  }

  // Depth axis is fixed at 1 for [B,D,H,W,2,2]
  const expectedAxis = 1;
  const ax = axis ?? expectedAxis;
  if (ax !== expectedAxis) {
    throw new Error(`Only axis=${expectedAxis} supported for [B,D,H,W,2,2]. Got axis=${ax}.`);
  }

  const twoA = x.shape[4];
  const twoB = x.shape[5];
  if (twoA !== 2 || twoB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${twoA},${twoB}].`);
  }

  const [B, D, H, W] = x.shape as unknown as [number, number, number, number, 2, 2];
  const outShape = [B, H, W, 2, 2];

  const backend: any = tf.backend();
  const prog = new UnstackDepthPackedBatchedProgram(outShape);

  const ys: tf.Tensor[] = [];
  for (let d = 0; d < D; d++) {
    const info = backend.runWebGLProgram(prog, [x], x.dtype, [[d]], true);
    ys.push(tf.engine().makeTensorFromTensorInfo(info));
  }

  return ys;
}
