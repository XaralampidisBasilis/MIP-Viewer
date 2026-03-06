import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class PackUnsignedShort5551ToUnsignedShort implements GPGPUProgram 
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
        uint uint551ToUint16(uint r, uint g, uint b, uint a)
        {
            r = clamp(r, 0u, 31u);  // 5 bits
            g = clamp(g, 0u, 31u);  // 5 bits
            b = clamp(b, 0u, 31u);  // 5 bits
            a = clamp(a, 0u, 1u);   // 1 bit

            return (r << 11) | (g << 6) | (b << 1) | a;
        }

        void uint16ToUint5551(uint u, out uint r, out uint g, out uint b, out uint a)
        {
            r = (u >> 11) & 31u;  // 5 bits
            g = (u >>  6) & 31u;  // 5 bits
            b = (u >>  1) & 31u;  // 5 bits
            a = (u >>  0) &  1u;  // 1 bit
        }

        void main() 
        {
            uint r = uint(getRAtOutCoords());
            uint g = uint(getGAtOutCoords());
            uint b = uint(getBAtOutCoords());
            uint a = uint(getAAtOutCoords());

            uint u = uint551ToUint16(r, g, b, a);
            
            setOutput(float(u));
        }
        `
    }
}

class PackUnsignedShort5x3ToNormalizedHalfFloat implements GPGPUProgram 
{
    variableNames = ['A0', 'A1', 'A2']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(inputShape: number[]) 
    {
        this.outputShape = inputShape
        this.userCode = `
        uint uint5x3ToUint15(uint a0, uint a1, uint a2)
        {
            a0 = clamp(a0, 0u, 31u); // 5 bits
            a1 = clamp(a1, 0u, 31u); // 5 bits
            a2 = clamp(a2, 0u, 31u); // 5 bits

            return (a0 << 10) | (a1 << 5) | (a2 << 0);
        }

        void uint15ToUint5x3(uint u, out uint a0, out uint a1, out uint a2)
        {
            a0 = (u >> 10) & 31u;
            a1 = (u >>  5) & 31u;
            a2 = (u >>  0) & 31u;
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

        float uint15ToNormalizedFloat16(uint u)
        {
            uint sign = (u >> 14) & 1u;     // 1-bit sign
            uint exp  = (u >> 10) & 0xFu;   // 4-bit exp
            uint frac = (u >>  0) & 0x3FFu; // 10-bit fraction
            
            // shift left, and if it was zero, add one to avoid denormals, 
            // and get into the normalized exponent range 1..30. 
            uint guard = (exp == 0u) ? 1u : 0u; 
            exp = (exp << 1) + guard; // 5-bit normalized exp

            // Encode u into a normal float16: sign=1 bit, exp=5 bits, frac=10 bits.
            uint bits = (sign << 15) | (exp << 10) | frac;

            return uint16ToFloat16(bits);
        }

        uint normalizedFloat16ToUint15(float f)
        {
            // Reverse: recover sign/exp/frac from half bits, then pack back into 15-bit u.
            uint bits = float16ToUint16(f);

            uint sign = (bits >> 15) & 1u;        // 1-bit sign
            uint exp  = (bits >> 10) & 0x1Fu;     // 5-bit exp
            uint frac = (bits >>  0) & 0x3FFu;    // 10-bit fraction

            // undo the shift we did to avoid denormals
            exp = (exp >> 1) & 0xFu;

            // pack back into 15-bit u: sign=1 bit, exp=4 bits, frac=10 bits
            return (sign << 14) | (exp << 10) | frac;
        }

        void main() 
        {
            uint a0 = uint(getA0AtOutCoords());
            uint a1 = uint(getA1AtOutCoords());
            uint a2 = uint(getA2AtOutCoords());

            uint u = uint5x3ToUint15(a0, a1, a2);
            float f = uint15ToNormalizedFloat16(u);
            
            setOutput(float(f));
        }
        `
    }
}

class PackUnsignedInt5x6ToNormalizedFloat implements GPGPUProgram 
{
    variableNames = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(inputShape: number[]) 
    {
        this.outputShape = inputShape
        this.userCode = `
        // Pack 6x uint5 into 30-bit payload (stored in a uint)
        uint packUint5x6ToUint30(uint a0, uint a1, uint a2, uint a3, uint a4, uint a5)
        {
            a0 = clamp(a0, 0u, 31u);
            a1 = clamp(a1, 0u, 31u);
            a2 = clamp(a2, 0u, 31u);
            a3 = clamp(a3, 0u, 31u);
            a4 = clamp(a4, 0u, 31u);
            a5 = clamp(a5, 0u, 31u);

            return (a0 << 25) | (a1 << 20) | (a2 << 15) | (a3 << 10) | (a4 << 5) | (a5 << 0);
        }

        void unpackUint30ToUint5x6(uint u, out uint a0, out uint a1, out uint a2, out uint a3, out uint a4, out uint a5)
        {
            a0 = (u >> 25) & 31u;
            a1 = (u >> 20) & 31u;
            a2 = (u >> 15) & 31u;
            a3 = (u >> 10) & 31u;
            a4 = (u >>  5) & 31u;
            a5 = (u >>  0) & 31u;
        }

        // Encode 30-bit payload into a *normalized, finite* float32
        float uint30ToNormalizedFloat32(uint u)
        {
            uint sign       = (u >> 29) & 1u;        // 1 bit
            uint expPayload = (u >> 22) & 0x7Fu;     // 7 bits
            uint frac22     =  u        & 0x3FFFFFu; // 22 bits

            // Avoid exp==0 (subnormals/zero). Stay in 1..254, avoid 255 (Inf/NaN).
            uint guard = (expPayload == 0u) ? 1u : 0u;
            uint exp8  = (expPayload << 1) + guard;  // 1,2..254 (always normal, finite)

            // Put payload frac into low 22 bits of the 23-bit fraction; top fraction bit stays 0.
            uint frac23 = frac22; // bit 22 is implicitly 0

            uint bits = (sign << 31) | (exp8 << 23) | frac23;
            return uintBitsToFloat(bits);
        }

        // Decode float32 back into the original 30-bit payload
        uint normalizedFloat32ToUint30(float f)
        {
            uint bits = floatBitsToUint(f);

            uint sign = (bits >> 31) & 1u;
            uint exp8 = (bits >> 23) & 0xFFu;
            uint frac =  bits        & 0x7FFFFFu;    // 23 bits

            uint expPayload = (exp8 >> 1) & 0x7Fu;
            uint frac22     =  frac & 0x3FFFFFu;     // low 22

            return (sign << 29) | (expPayload << 22) | frac22;
        }

        void main() 
        {
            uint a0 = uint(getA0AtOutCoords());
            uint a1 = uint(getA1AtOutCoords());
            uint a2 = uint(getA2AtOutCoords());
            uint a3 = uint(getA3AtOutCoords());
            uint a4 = uint(getA4AtOutCoords());
            uint a5 = uint(getA5AtOutCoords());

            uint u = packUint5x6ToUint30(a0, a1, a2, a3, a4, a5);
            float f = uint30ToNormalizedFloat32(u);
            
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

export function packUnsignedShort5551ToUnsignedShort(R: tf.Tensor, G: tf.Tensor, B: tf.Tensor, A: tf.Tensor): tf.Tensor
{
    const shape = R.shape
    const program = new PackUnsignedShort5551ToUnsignedShort(shape)
    return runProgram(program, [R, G, B, A])
}

export function packUnsignedShort5x3ToNormalizedHalfFloat(A0: tf.Tensor, A1: tf.Tensor, A2: tf.Tensor): tf.Tensor
{
    const shape = A0.shape
    const program = new PackUnsignedShort5x3ToNormalizedHalfFloat(shape)
    return runProgram(program, [A0, A1, A2])
}