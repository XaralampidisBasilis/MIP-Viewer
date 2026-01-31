import * as tf from '@tensorflow/tfjs';
import {GPGPUProgram, useShapeUniforms} from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math';
import {getCoordsDataType} from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

export class UnstackDepthPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = true;

  // Tell backend we have a custom uniform.
  customUniforms = [{name: 'uDepth', type: 'int' as const}];

  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(outputShape: number[], hasBatch: boolean) {
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    const outRank = outputShape.length;
    const dtype = getCoordsDataType(outRank);

    if (hasBatch) {
      // output: [B,H,W,2,2]
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
    } else {
      // output: [H,W,2,2]
      this.userCode = `
      void main() {
        ${dtype} outC = getOutputCoords();
        int h = outC.x;
        int w = outC.y;
        int r = outC.z;
        int c = outC.w;

        vec4 v = getA(uDepth, h, w, r, c);
        setOutput(v);
      }
      `;
    }
  }
}

export function unstackDepthPacked(x: tf.Tensor, axis?: number): tf.Tensor[] {
  if (x.rank !== 6 && x.rank !== 5) {
    throw new Error(`Expected rank-6 or rank-5, got rank=${x.rank}.`);
  }

  const hasBatch = x.rank === 6;
  const expectedAxis = hasBatch ? 1 : 0;
  const ax = axis ?? expectedAxis;
  if (ax !== expectedAxis) {
    throw new Error(`Only depth axis supported: axis=${expectedAxis} for rank=${x.rank}. Got axis=${ax}.`);
  }

  const twoA = x.shape[x.shape.length - 2];
  const twoB = x.shape[x.shape.length - 1];
  if (twoA !== 2 || twoB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${twoA},${twoB}].`);
  }

  let B = 1, D: number, H: number, W: number;
  if (hasBatch) {
    [B, D, H, W] = x.shape as [number, number, number, number, 2, 2];
  } else {
    [D, H, W] = x.shape as [number, number, number, 2, 2];
  }

  const outShape = hasBatch ? [B, H, W, 2, 2] : [H, W, 2, 2];

  const backend: any = tf.backend();

  // Compile once.
  const prog = new UnstackDepthPackedProgram(outShape, hasBatch);

  const ys: tf.Tensor[] = [];
  
  for (let d = 0; d < D; d++) 
  {
    const s = backend.runWebGLProgram(prog, [x], x.dtype, [[d]], true);
    const y = tf.engine().makeTensorFromTensorInfo(s)

    ys.push(y);
  }

  return ys;
}
