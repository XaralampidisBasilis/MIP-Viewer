import * as tf from '@tensorflow/tfjs';
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math';
import { getCoordsDataType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

type Axis3 = 0 | 1 | 2;

/**
 * Unbatched-only unstack for packed rank-5:
 *   input:  [D, H, W, 2, 2]
 *
 * axis=0 (depth):  returns D tensors, each [H, W, 2, 2]
 * axis=1 (height): returns H tensors, each [D, W, 2, 2]
 * axis=2 (width):  returns W tensors, each [D, H, 2, 2]
 */
export class UnstackPackedUnbatchedProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = true;

  // Which slice (along the chosen axis) to read:
  customUniforms = [{ name: 'uIndex', type: 'int' as const }];

  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;

  constructor(outputShape: number[], axis: Axis3) {
    // outputShape must be rank-4: [..., ..., 2, 2]
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    const outRank = outputShape.length; // should be 4
    const dtype = getCoordsDataType(outRank);

    // Map output coords -> input coords depending on axis
    // Input coords are always (d, h, w, r, c).
    let mapCoords: string;
    if (axis === 0) {
      // output: [H, W, 2, 2]
      mapCoords = `
        int d = uIndex;
        int h = outC.x;
        int w = outC.y;
      `;
    } else if (axis === 1) {
      // output: [D, W, 2, 2]
      mapCoords = `
        int d = outC.x;
        int h = uIndex;
        int w = outC.y;
      `;
    } else {
      // axis === 2
      // output: [D, H, 2, 2]
      mapCoords = `
        int d = outC.x;
        int h = outC.y;
        int w = uIndex;
      `;
    }

    this.userCode = `
      void main() {
        ${dtype} outC = getOutputCoords();

        int r = outC.z;
        int c = outC.w;

        ${mapCoords}

        vec4 v = getA(d, h, w, r, c);
        setOutput(v);
      }
    `;
  }
}

export function unstackPacked(x: tf.Tensor, axis: Axis3 = 0): tf.Tensor[] {
  if (x.rank !== 5) {
    throw new Error(`Unbatched unstack expects rank-5 [D,H,W,2,2]. Got rank=${x.rank}.`);
  }
  if (axis !== 0 && axis !== 1 && axis !== 2) {
    throw new Error(`axis must be 0, 1, or 2. Got axis=${axis}.`);
  }

  const [D, H, W, twoA, twoB] = x.shape as unknown as [number, number, number, number, number];
  if (twoA !== 2 || twoB !== 2) {
    throw new Error(`Expected trailing [2,2], got [${twoA},${twoB}].`);
  }

  const sliceCount = [D, H, W][axis];

  // Remove the chosen axis from [D,H,W], keep [2,2]
  const base3 = [D, H, W];
  const outShape = base3.filter((_, i) => i !== axis).concat([2, 2]); // rank-4

  const backend: any = tf.backend();
  const prog = new UnstackPackedUnbatchedProgram(outShape, axis);

  const ys: tf.Tensor[] = [];
  for (let i = 0; i < sliceCount; i++) {
    // One custom uniform (uIndex). tfjs WebGL backend expects it as a 1-element array.
    const info = backend.runWebGLProgram(prog, [x], x.dtype, [[i]], true);
    ys.push(tf.engine().makeTensorFromTensorInfo(info));
  }

  return ys;
}
