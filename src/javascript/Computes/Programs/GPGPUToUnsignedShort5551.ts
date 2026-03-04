import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUPackUnsignedShort5551 implements GPGPUProgram 
{
    variableNames = ['R', 'G', 'B', 'A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(inputShape: number[]) 
    {
        this.outputShape = inputShape
        this.userCode = `
        uint packUnsignedShort5551(uint x, uint y, uint z, uint o)
        {
            uint u = 
            (clamp(x, 0u, 31u) << 11) |
            (clamp(y, 0u, 31u) <<  6) |
            (clamp(z, 0u, 31u) <<  1) |
            (clamp(o, 0u, 31u) <<  0);

            return u;
        }

        float uint16ToFloat16(uint p)
        {
            // lower 16 bits are the half we care about
            return unpackHalf2x16(p & 0xFFFFu).x;
        }

        uint float16ToUint16(float f)
        {
            // half in low 16 bits, upper 16 bits are whatever (we'll mask when needed)
            return packHalf2x16(vec2(f, 0.0));
        }

        float uint15ToFloat16(uint u)
        {
            // Encode u into a NORMAL half: sign=1 bit, exp=4 bits (stored as 1..16), frac=10 bits.
            uint sign = (u >> 14) & 1u;
            uint exp  = ((u >> 10) & 0xFu) + 1u;   // 1..16 => normal (never 0, never 31)
            uint frac =  u & 0x3FFu;        // 10-bit fraction
            uint bits = (sign << 15) | (exp << 10) | frac;

            return uint16ToFloat16(bits);
        }

        uint float16ToUint15(float f)
        {
            // Reverse: recover sign/exp/frac from half bits, then pack back into 15-bit u.
            uint bits = float16ToUint16(f);
            uint sign = (bits >> 15) & 1u;
            uint exp  = (bits >> 10) & 0x1Fu;      // expected 1..16 if data was untouched
            uint frac =  bits & 0x3FFu;
            return (sign << 14) | ((exp - 1u) << 10) | frac;
        }

        void main() 
        {
            uint r = uint(getRAtOutCoords());
            uint g = uint(getGAtOutCoords());
            uint b = uint(getBAtOutCoords());
            uint a = uint(getAAtOutCoords());

            uint p = packUnsignedShort5551(r, g, b, a);
            setOutput(float(p));
        }
        `
    }
}

function runProgram(prog: GPGPUProgram, inputs: tf.Tensor[]) : tf.Tensor 
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs)
    return tf.engine().makeTensorFromTensorInfo(info) as tf.Tensor
}

export function packUnsignedShort5551(inputR: tf.Tensor, inputG: tf.Tensor, inputB: tf.Tensor, inputA: tf.Tensor): tf.Tensor
{
    const shape = inputR.shape
    const program = new GPGPUPackUnsignedShort5551(shape)
    return runProgram(program, [inputR, inputG, inputB, inputA])
}
