import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'

type Mismatch = {
  inputBits: number
  outputBits: number
  inputClass: string
  outputClass: string
  inputValue: number
  outputValue: number
}

function u16Hex(x: number): string 
{
    return '0x' + x.toString(16).padStart(4, '0')
}

/**
 * Reinterpret one uint16 as one float16 value, using the browser's Float16Array.
 * Assumes Float16Array exists.
 */
function uint16ToFloat16Number(bits: number): number 
{
    const buffer = new ArrayBuffer(2)
    const u16 = new Uint16Array(buffer)
    const f16 = new Float16Array(buffer as ArrayBuffer)
    u16[0] = bits & 0xffff
    return Number(f16[0])
}

/**
 * Encode one JS number into float16 and return its raw uint16 bits,
 * using the browser's Float16Array.
 *
 * Important:
 * - This follows the browser's Float16Array rounding / canonicalization rules.
 * - Distinct NaN payloads will generally collapse here.
 */
function float16NumberToUint16(value: number): number 
{
    const buffer = new ArrayBuffer(2)
    const u16 = new Uint16Array(buffer)
    const f16 = new Float16Array(buffer as ArrayBuffer)
    f16[0] = value
    return u16[0]
}

/**
 * Classify a float16 bit pattern for debugging.
 */
function classifyFloat16Bits(bits: number): string 
{
    const sign = (bits >>> 15) & 0x1
    const exp = (bits >>> 10) & 0x1f
    const frac = bits & 0x03ff

    if (exp === 0) 
    {
        if (frac === 0) return sign ? '-zero' : '+zero'
        return sign ? '-subnormal' : '+subnormal'
    }

    if (exp === 0x1f) {
        if (frac === 0) return sign ? '-inf' : '+inf'
        return 'nan'
    }

    return sign ? '-normal' : '+normal'
}

/**
 * Builds a Float16Array that contains every possible half-float bit pattern
 * by writing the raw uint16s into the same buffer.
 */
function buildAllFloat16Values(): { inputBits: Uint16Array; inputValues: Float16Array } 
{
    const count = 65536
    const buffer = new ArrayBuffer(count * 2)
    const inputBits = new Uint16Array(buffer)
    const inputValues = new Float16Array(buffer as ArrayBuffer)

    for (let i = 0; i < count; i++) 
    {
        inputBits[i] = i
    }

    return { inputBits, inputValues }
}

/**
 * Small helper to get backend info that often matters for interpreting results.
 */
function logBackendInfo(): void 
{
    const backend = tf.backend() as any
    const gl = backend?.gpgpu?.gl as WebGLRenderingContext | WebGL2RenderingContext | undefined

    console.log('Backend:', tf.getBackend())

    if (!gl) 
    {
        console.log('No WebGL context was found on the backend object.')
        return
    }

    console.log('WebGL version:', gl instanceof WebGL2RenderingContext ? 'WebGL2' : 'WebGL1')
    console.log('Renderer:', gl.getParameter(gl.RENDERER))
    console.log('Vendor:', gl.getParameter(gl.VENDOR))

    const extHalfFloat =
        gl.getExtension('EXT_color_buffer_half_float') ||
        gl.getExtension('EXT_color_buffer_float')

    console.log('Half/float color buffer extension available:', !!extHalfFloat)
}

/**
 * Main round-trip test.
 *
 * What it tests:
 * 1. Construct all 65536 raw half-float bit patterns.
 * 2. Reinterpret them as Float16Array values.
 * 3. Upload to a tf tensor on WebGL.
 * 4. Read back with dataSync().
 * 5. Re-encode read values into float16 raw bits.
 * 6. Compare bit-for-bit.
 *
 * Caveat:
 * This does NOT preserve NaN payload identity through the JS number layer.
 * So many NaN inputs may already collapse before TFJS even sees them as distinct.
 */
async function testAllFloat16RoundTrip(): Promise<void> 
{
    await tf.setBackend('webgl')
    await tf.ready()

    const { inputBits, inputValues } = buildAllFloat16Values()

    console.log('Creating tensor with all 65536 float16 bit patterns...')
    const x = tf.tensor1d(new Float32Array(inputValues))

    console.log('Reading back with dataSync()...')
    const out = x.dataSync() as Float32Array | Float64Array | Int32Array | Uint8Array

    const mismatches: Mismatch[] = []
    let sameCount = 0

    const classCounts = new Map<string, { total: number; same: number; changed: number }>()

    for (let i = 0; i < 65536; i++) {
        const inBits = inputBits[i]
        const outValue = Number(out[i])
        const outBits = float16NumberToUint16(outValue)

        const inClass = classifyFloat16Bits(inBits)
        const outClass = classifyFloat16Bits(outBits)

        const key = inClass
        const stats = classCounts.get(key) ?? { total: 0, same: 0, changed: 0 }
        stats.total++

        if (inBits === outBits) {
        sameCount++
        stats.same++
        } else {
        stats.changed++
        mismatches.push({
            inputBits: inBits,
            outputBits: outBits,
            inputClass: inClass,
            outputClass: outClass,
            inputValue: Number(inputValues[i]),
            outputValue: outValue,
        })
        }

        classCounts.set(key, stats)
    }

    const changedCount = 65536 - sameCount

    console.log('')
    console.log('===== SUMMARY =====')
    console.log(`Total tested:   65536`)
    console.log(`Bit-exact same: ${sameCount}`)
    console.log(`Changed:        ${changedCount}`)
    console.log(`Survival rate:  ${(100 * sameCount / 65536).toFixed(4)}%`)
    console.log('')

    console.log('===== BY INPUT CLASS =====')
    for (const [cls, stats] of classCounts.entries()) 
    {
        console.log(
        `${cls.padEnd(12)} total=${String(stats.total).padStart(5)}  ` +
        `same=${String(stats.same).padStart(5)}  ` +
        `changed=${String(stats.changed).padStart(5)}`
        )
    }

    console.log('')
    console.log('===== FIRST 100 MISMATCHES =====')
    for (let i = 0; i < Math.min(100, mismatches.length); i++) 
    {
        const m = mismatches[i]
        console.log(
        `[${i}] ` +
        `in=${u16Hex(m.inputBits)} (${m.inputClass}, value=${String(m.inputValue)})  ->  ` +
        `out=${u16Hex(m.outputBits)} (${m.outputClass}, value=${String(m.outputValue)})`
        )
    }

    console.log('')
    console.log('===== SPECIAL COUNTS =====')

    let nanInputs = 0
    let nanChanged = 0
    let subnormalInputs = 0
    let subnormalChanged = 0
    let zeroInputs = 0
    let zeroChanged = 0
    let infInputs = 0
    let infChanged = 0

    for (let i = 0; i < 65536; i++) 
    {
        const inBits = inputBits[i]
        const outBits = float16NumberToUint16(Number(out[i]))

        const inClass = classifyFloat16Bits(inBits)
        const changed = inBits !== outBits

        if (inClass === 'nan') {
        nanInputs++
        if (changed) nanChanged++
        } else if (inClass.includes('subnormal')) {
        subnormalInputs++
        if (changed) subnormalChanged++
        } else if (inClass.includes('zero')) {
        zeroInputs++
        if (changed) zeroChanged++
        } else if (inClass.includes('inf')) {
        infInputs++
        if (changed) infChanged++
        }
    }

    console.log(`NaN inputs:        ${nanInputs}, changed: ${nanChanged}`)
    console.log(`Subnormal inputs:  ${subnormalInputs}, changed: ${subnormalChanged}`)
    console.log(`Zero inputs:       ${zeroInputs}, changed: ${zeroChanged}`)
    console.log(`Infinity inputs:   ${infInputs}, changed: ${infChanged}`)

    x.dispose()
}

testAllFloat16RoundTrip().catch((err) => 
{
    console.error(err)
})

/* Results:
float16BitSurvivalTest.ts:139 Creating tensor with all 65536 float16 bit patterns...
float16BitSurvivalTest.ts:142 Reading back with dataSync()...
float16BitSurvivalTest.ts:182 
float16BitSurvivalTest.ts:183 ===== SUMMARY =====
float16BitSurvivalTest.ts:184 Total tested:   65536
float16BitSurvivalTest.ts:185 Bit-exact same: 64003
float16BitSurvivalTest.ts:186 Changed:        1533
float16BitSurvivalTest.ts:187 Survival rate:  97.6608%
float16BitSurvivalTest.ts:188 
float16BitSurvivalTest.ts:190 ===== BY INPUT CLASS =====
float16BitSurvivalTest.ts:193 +zero        total=    1  same=    1  changed=    0
float16BitSurvivalTest.ts:193 +subnormal   total= 1023  same= 1023  changed=    0
float16BitSurvivalTest.ts:193 +normal      total=30720  same=30720  changed=    0
float16BitSurvivalTest.ts:193 +inf         total=    1  same=    1  changed=    0
float16BitSurvivalTest.ts:193 nan          total= 2046  same=  513  changed= 1533
float16BitSurvivalTest.ts:193 -zero        total=    1  same=    1  changed=    0
float16BitSurvivalTest.ts:193 -subnormal   total= 1023  same= 1023  changed=    0
float16BitSurvivalTest.ts:193 -normal      total=30720  same=30720  changed=    0
float16BitSurvivalTest.ts:193 -inf         total=    1  same=    1  changed=    0
float16BitSurvivalTest.ts:200 
float16BitSurvivalTest.ts:201 ===== FIRST 100 MISMATCHES =====
float16BitSurvivalTest.ts:205 [0] in=0x7c01 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [1] in=0x7c02 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [2] in=0x7c03 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [3] in=0x7c04 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [4] in=0x7c05 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [5] in=0x7c06 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [6] in=0x7c07 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [7] in=0x7c08 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [8] in=0x7c09 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [9] in=0x7c0a (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [10] in=0x7c0b (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [11] in=0x7c0c (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [12] in=0x7c0d (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [13] in=0x7c0e (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [14] in=0x7c0f (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [15] in=0x7c10 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [16] in=0x7c11 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [17] in=0x7c12 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [18] in=0x7c13 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [19] in=0x7c14 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [20] in=0x7c15 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [21] in=0x7c16 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [22] in=0x7c17 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [23] in=0x7c18 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [24] in=0x7c19 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [25] in=0x7c1a (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [26] in=0x7c1b (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [27] in=0x7c1c (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [28] in=0x7c1d (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [29] in=0x7c1e (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [30] in=0x7c1f (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [31] in=0x7c20 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [32] in=0x7c21 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [33] in=0x7c22 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [34] in=0x7c23 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [35] in=0x7c24 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [36] in=0x7c25 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [37] in=0x7c26 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [38] in=0x7c27 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [39] in=0x7c28 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [40] in=0x7c29 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [41] in=0x7c2a (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [42] in=0x7c2b (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [43] in=0x7c2c (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [44] in=0x7c2d (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [45] in=0x7c2e (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [46] in=0x7c2f (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [47] in=0x7c30 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [48] in=0x7c31 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [49] in=0x7c32 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [50] in=0x7c33 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [51] in=0x7c34 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [52] in=0x7c35 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [53] in=0x7c36 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [54] in=0x7c37 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [55] in=0x7c38 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [56] in=0x7c39 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [57] in=0x7c3a (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [58] in=0x7c3b (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [59] in=0x7c3c (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [60] in=0x7c3d (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [61] in=0x7c3e (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [62] in=0x7c3f (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [63] in=0x7c40 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [64] in=0x7c41 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [65] in=0x7c42 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [66] in=0x7c43 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [67] in=0x7c44 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [68] in=0x7c45 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [69] in=0x7c46 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [70] in=0x7c47 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [71] in=0x7c48 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [72] in=0x7c49 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [73] in=0x7c4a (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [74] in=0x7c4b (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [75] in=0x7c4c (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [76] in=0x7c4d (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [77] in=0x7c4e (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [78] in=0x7c4f (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [79] in=0x7c50 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [80] in=0x7c51 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [81] in=0x7c52 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [82] in=0x7c53 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [83] in=0x7c54 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [84] in=0x7c55 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [85] in=0x7c56 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [86] in=0x7c57 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [87] in=0x7c58 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [88] in=0x7c59 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [89] in=0x7c5a (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [90] in=0x7c5b (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [91] in=0x7c5c (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [92] in=0x7c5d (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [93] in=0x7c5e (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [94] in=0x7c5f (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [95] in=0x7c60 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [96] in=0x7c61 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [97] in=0x7c62 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [98] in=0x7c63 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:205 [99] in=0x7c64 (nan, value=NaN)  ->  out=0x7e00 (nan, value=NaN)
float16BitSurvivalTest.ts:212 
float16BitSurvivalTest.ts:213 ===== SPECIAL COUNTS =====
float16BitSurvivalTest.ts:247 NaN inputs:        2046, changed: 1022
float16BitSurvivalTest.ts:248 Subnormal inputs:  2046, changed: 0
float16BitSurvivalTest.ts:249 Zero inputs:       2, changed: 0
float16BitSurvivalTest.ts:250 Infinity inputs:   2, changed: 0
*/
