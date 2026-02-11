import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class NormalizePackedProgram implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    customUniforms = [
        { name: 'minValue', type: 'float' as const }, 
        { name: 'maxValue', type: 'float' as const }
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
            float range = maxValue - minValue;
            setOutput((getAAtOutCoords() - vec4(minValue)) / vec4(range));
        }
        `
    }
}

export function normalizePacked(tensor: tf.Tensor3D): { normalized: tf.Tensor3D, minValue: number, maxValue: number } 
{
    const minValue = tf.tidy(() => tf.min(tensor).dataSync()[0])
    const maxValue = tf.tidy(() => tf.max(tensor).dataSync()[0])
    const customValues = [[minValue], [maxValue]]

    const program = new NormalizePackedProgram(tensor.shape)
    const normalized = runWebGLProgram(program, [tensor], 'float32', customValues, false) as tf.Tensor3D   

    return { normalized, minValue, maxValue } 
}

function runWebGLProgram(prog: GPGPUProgram, inputs: tf.Tensor[], dtype?: tf.DataType, customValues?: number[][], preventEagerUnpackingOfOutput?: boolean): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs, dtype, customValues, preventEagerUnpackingOfOutput)
    return tf.engine().makeTensorFromTensorInfo(info) 
}