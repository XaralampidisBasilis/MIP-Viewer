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
        uint rgbaToUint5551(uint r, uint g, uint b, uint a)
        {
            uint u = 
            (clamp(r, 0u, 31u) << 11) |
            (clamp(g, 0u, 31u) <<  6) |
            (clamp(b, 0u, 31u) <<  1) |
            (clamp(a, 0u, 31u) <<  0);

            return u;
        }

        float uint16ToFloat16(uint p)
        {
            // lower 16 bits are the half we care about
            return unpackHalf2x16(p & 0xFFFFu).x;
        }

        uint float16ToUint16(float f)
        {
            // half in low 16 bits, upper 16 bits are whatever (we ignore them)
            return packHalf2x16(vec2(f, 0.0));
        }

        float uint15ToFloat16(uint u)
        {
            uint sign = (u >> 14) & 1u;     // 1-bit sign
            uint exp  = (u >> 10) & 0xFu;   // 4-bit exp
            uint frac = (u >>  0) & 0x3FFu; // 10-bit fraction

            // Shift exponent from 0, to avoid subnormals, and get into normalized range. 
            // Max input of 15 becomes 16, which is the max normal, not infinity.
            exp += 1u; // 1..16
            
            // Encode u into a normal float16: sign=1 bit, exp=4 bits (stored as 1..16), frac=10 bits.
            uint bits = (sign << 15) | (exp << 10) | frac;

            return uint16ToFloat16(bits);
        }

        uint float16ToUint15(float f)
        {
            // Reverse: recover sign/exp/frac from half bits, then pack back into 15-bit u.
            uint bits = float16ToUint16(f);

            uint sign = (bits >> 15) & 1u;
            uint exp  = (bits >> 10) & 0x1Fu;     // 5-bit exp
            uint frac = (bits >>  0) & 0x3FFu;    // 10-bit fraction

            // undo the +1 we did to avoid subnormals
            exp = (exp - 1u) & 0xFu;

            // pack back into 15-bit u: sign=1 bit, exp=4 bits, frac=10 bits
            return (sign << 14) | (exp << 10) | frac;
        }

        void main() 
        {
            uint r = uint(getRAtOutCoords());
            uint g = uint(getGAtOutCoords());
            uint b = uint(getBAtOutCoords());
            uint a = uint(getAAtOutCoords());

            uint u = rgbaToUint5551(r, g, b, a);
            float f = uint15ToFloat16(u);
            
            setOutput(float(f));
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
