import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class MapPackedProgram implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    customUniforms = [
        { name: 'a', type: 'float' as const }, 
        { name: 'b', type: 'float' as const }
    ]

    constructor
    (
        inputShape: [number, number, number], 
    ) 
    {
        this.outputShape = inputShape
        this.userCode = `
        void main() 
        {
            float d = b - a;
            vec4 x = getAAtOutCoords();
            vec4 y = (x - vec4(a)) / vec4(d);
            vec4 t = clamp(y, 0.0, 1.0);

            setOutput(t);
        }
        `
    }
}

export function mapPacked(tensor: tf.Tensor3D, a: number, b: number): tf.Tensor3D  
{
    const program = new MapPackedProgram(tensor.shape)
    const customValues = [[a], [b]]

    return runWebGLProgram(program, [tensor], 'float32', customValues, false) as tf.Tensor3D   
}

function runWebGLProgram(prog: GPGPUProgram, inputs: tf.Tensor[], dtype?: tf.DataType, customValues?: number[][], preventEagerUnpackingOfOutput?: boolean): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs, dtype, customValues, preventEagerUnpackingOfOutput)
    return tf.engine().makeTensorFromTensorInfo(info) 
}