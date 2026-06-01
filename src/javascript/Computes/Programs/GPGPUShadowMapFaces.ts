/**
 * Extended anisotropic shadow maps for maximum intensity projection.
 *
 * This is the face-min/max version of the cell-based shadow-map idea from
 * Mroz, Hauser, and Groeller, "Interactive High-Quality Maximum Intensity
 * Projection". The implementation keeps the old GPGPUShadowMap behavior, but
 * uses the orientation notation from GPGPUShadowMapPaths2: shaders are written
 * in one canonical local +z sweep space and offsets are converted to the
 * physical TensorFlow [z, y, x] layout by a dominant-axis/octant pair.
 */
import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstackPacked } from './unstack_packed_keepDims_webgl'
import { stackPacked } from './stack_packed_keepDims_webgl'
import {
    type Axis,
    type Octant,
    applyPermutation,
    dominantAxisOctantToPermuteReverse,
    inversePermutation,
    reverseOctant,
} from '../../Utils/ShadowMapUtils'
import { minPool3d } from './pool3d'

type Shape3 = [number, number, number]
type Shape3Packed = [number, number, number, 2, 2]
type CoordExpr = number | string


function xyz(zyx: [CoordExpr, CoordExpr, CoordExpr]): string
{
    return [zyx[2], zyx[1], zyx[0]].join(', ')
}

/**
 * Offset for cell-corner addressing in the initial face min/max pass.
 * Reversal mirrors a unit cell corner: coordinate 0 becomes 1 and coordinate
 * 1 becomes 0.
 */
function cellOffset(
    x: number,
    y: number,
    z: number,
    dominantAxis: Axis,
    octant: Octant
): string
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(dominantAxis, octant)

    const zyx = applyPermutation([z, y, x], permute)

    for (const dim of reverse) zyx[dim] = 1 - zyx[dim]

    return xyz(zyx)
}

/**
 * Offset for propagated face-map addressing. Reversal flips the signed offset.
 */
function voxelOffset(
    x: number,
    y: number,
    z: number,
    dominantAxis: Axis,
    octant: Octant
): string
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(dominantAxis, octant)
    const zyx = applyPermutation([z, y, x], permute)

    for (const dim of reverse) zyx[dim] = -zyx[dim]

    return xyz(zyx)
}

/**
 * Offset for slice propagation. The previous sweep-plane is supplied as a
 * separate tensor, so movement along the sweep axis is erased.
 */
function sliceOffset(
    x: number,
    y: number,
    z: number,
    dominantAxis: Axis,
    octant: Octant
): string
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(dominantAxis, octant)
    const zyx = applyPermutation([z, y, x], permute)

    for (const dim of reverse) zyx[dim] = -zyx[dim]

    zyx[permute[0]] = 0

    return xyz(zyx)
}

function addOneShape([depth, height, width]: Shape3): Shape3
{
    return [depth + 1, height + 1, width + 1]
}

class ComputeFaceMinimaProgram implements GPGPUProgram
{
    variableNames: string[]
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        volumeShape: Shape3,
        dominantAxis: Axis = 'z',
        octant: Octant = '+++',
        hollow = false
    ) {
        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = addOneShape(volumeShape)

        this.variableNames = hollow ? ['A', 'B'] : ['A']
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth - 1}, ${inHeight - 1}, ${inDepth - 1});

        struct CellValues
        {
            float v000;
            float v100;
            float v010;
            float v001;
            float v011;
            float v101;
            float v110;
            float v111;
        };

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        bool insideVolume(ivec3 p)
        {
            if (p.x < minCoords.x || p.x > maxCoords.x) return false;
            if (p.y < minCoords.y || p.y > maxCoords.y) return false;
            if (p.z < minCoords.z || p.z > maxCoords.z) return false;

            return true;
        }

        ivec3 outputCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float volumeAt(ivec3 p)
        {
            return insideVolume(p) ? getA(p.z, p.y, p.x) : 0.0;
        }

        ${hollow ? `
        float holesAt(ivec3 p)
        {
            return insideVolume(p) ? getB(p.z, p.y, p.x) : 0.0;
        }

        bool isHollow(ivec3 p)
        {
            return holesAt(p) > 0.5;
        }
        ` : ''}

        CellValues cellValues(ivec3 coords)
        {
            ivec3 base = coords - 1;

            CellValues c;
            c.v000 = volumeAt(base + ivec3(${cellOffset(0, 0, 0, dominantAxis, octant)}));
            c.v100 = volumeAt(base + ivec3(${cellOffset(1, 0, 0, dominantAxis, octant)}));
            c.v010 = volumeAt(base + ivec3(${cellOffset(0, 1, 0, dominantAxis, octant)}));
            c.v001 = volumeAt(base + ivec3(${cellOffset(0, 0, 1, dominantAxis, octant)}));
            c.v011 = volumeAt(base + ivec3(${cellOffset(0, 1, 1, dominantAxis, octant)}));
            c.v101 = volumeAt(base + ivec3(${cellOffset(1, 0, 1, dominantAxis, octant)}));
            c.v110 = volumeAt(base + ivec3(${cellOffset(1, 1, 0, dominantAxis, octant)}));
            c.v111 = volumeAt(base + ivec3(${cellOffset(1, 1, 1, dominantAxis, octant)}));

            return c;
        }

        float faceX(CellValues c)
        {
            return min4(c.v100, c.v110, c.v101, c.v111);
        }

        float faceY(CellValues c)
        {
            return min4(c.v010, c.v110, c.v011, c.v111);
        }

        float faceZ(CellValues c)
        {
            return min4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            ivec3 coords = outputCoords();
            CellValues c = cellValues(coords);

            ${hollow ? `
            if (isHollow(coords))
            {
                setOutput(vec4(0.0, 0.0, 0.0, 1.0));
                return;
            }
            ` : ''}

            setOutput(vec4(faceX(c), faceY(c), faceZ(c), 0.0));
        }
        `
    }
}

class ComputeFaceMaximaProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        volumeShape: Shape3,
        dominantAxis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = addOneShape(volumeShape)

        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth - 1}, ${inHeight - 1}, ${inDepth - 1});

        struct CellValues
        {
            float v000;
            float v100;
            float v010;
            float v001;
            float v011;
            float v101;
            float v110;
            float v111;
        };

        float max4(float a, float b, float c, float d)
        {
            return max(max(max(a, b), c), d);
        }

        bool insideVolume(ivec3 p)
        {
            if (p.x < minCoords.x || p.x > maxCoords.x) return false;
            if (p.y < minCoords.y || p.y > maxCoords.y) return false;
            if (p.z < minCoords.z || p.z > maxCoords.z) return false;

            return true;
        }

        ivec3 outputCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float volumeAt(ivec3 p)
        {
            return insideVolume(p) ? getA(p.z, p.y, p.x) : 0.0;
        }

        CellValues cellValues(ivec3 coords)
        {
            ivec3 base = coords - 1;

            CellValues c;
            c.v000 = volumeAt(base + ivec3(${cellOffset(0, 0, 0, dominantAxis, octant)}));
            c.v100 = volumeAt(base + ivec3(${cellOffset(1, 0, 0, dominantAxis, octant)}));
            c.v010 = volumeAt(base + ivec3(${cellOffset(0, 1, 0, dominantAxis, octant)}));
            c.v001 = volumeAt(base + ivec3(${cellOffset(0, 0, 1, dominantAxis, octant)}));
            c.v011 = volumeAt(base + ivec3(${cellOffset(0, 1, 1, dominantAxis, octant)}));
            c.v101 = volumeAt(base + ivec3(${cellOffset(1, 0, 1, dominantAxis, octant)}));
            c.v110 = volumeAt(base + ivec3(${cellOffset(1, 1, 0, dominantAxis, octant)}));
            c.v111 = volumeAt(base + ivec3(${cellOffset(1, 1, 1, dominantAxis, octant)}));

            return c;
        }

        float faceX(CellValues c)
        {
            return max4(c.v100, c.v110, c.v101, c.v111);
        }

        float faceY(CellValues c)
        {
            return max4(c.v010, c.v110, c.v011, c.v111);
        }

        float faceZ(CellValues c)
        {
            return max4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            CellValues c = cellValues(outputCoords());

            setOutput(vec4(faceX(c), faceY(c), faceZ(c), 0.5));
        }
        `
    }
}

class PropagateFaceMinmaxProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        outputShape: Shape3Packed,
        dominantAxis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [depth, height, width] = outputShape

        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool insideVolume(ivec3 p)
        {
            if (p.x < minCoords.x || p.x > maxCoords.x) return false;
            if (p.y < minCoords.y || p.y > maxCoords.y) return false;
            if (p.z < minCoords.z || p.z > maxCoords.z) return false;

            return true;
        }

        ivec3 outputCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        vec4 currentSliceAt(ivec3 p)
        {
            return insideVolume(p) ? getA(p.z, p.y, p.x, 0, 0) : vec4(0.0);
        }

        vec4 previousSliceAt(ivec3 p)
        {
            return insideVolume(p) ? getB(p.z, p.y, p.x, 0, 0) : vec4(0.0);
        }

        float minmaxX(vec4 c111, vec4 c110, vec4 c101, vec4 c100)
        {
            return max(c111.x, min(c110.z, max(c101.y, c100.z)));
        }

        float minmaxY(vec4 c111, vec4 c110, vec4 c011, vec4 c010)
        {
            return max(c111.y, min(c110.z, max(c011.x, c010.z)));
        }

        float minmaxZ(vec4 c111, vec4 c110, vec4 c101, vec4 c011)
        {
            return max(c111.z, min(c110.z, min(c101.y, c011.x)));
        }

        vec4 propagatedMinima(ivec3 p)
        {
            vec4 c111 =  currentSliceAt(p + ivec3(${sliceOffset( 0,  0,  0, dominantAxis, octant)}));
            vec4 c011 =  currentSliceAt(p + ivec3(${sliceOffset(-1,  0,  0, dominantAxis, octant)}));
            vec4 c101 =  currentSliceAt(p + ivec3(${sliceOffset( 0, -1,  0, dominantAxis, octant)}));
            vec4 c001 =  currentSliceAt(p + ivec3(${sliceOffset(-1, -1,  0, dominantAxis, octant)}));
            vec4 c110 = previousSliceAt(p + ivec3(${sliceOffset( 0,  0, -1, dominantAxis, octant)}));
            vec4 c010 = previousSliceAt(p + ivec3(${sliceOffset(-1,  0, -1, dominantAxis, octant)}));
            vec4 c100 = previousSliceAt(p + ivec3(${sliceOffset( 0, -1, -1, dominantAxis, octant)}));
            vec4 c000 = previousSliceAt(p + ivec3(${sliceOffset(-1, -1, -1, dominantAxis, octant)}));

            c011.x = minmaxX(c011, c010, c001, c000);
            c101.y = minmaxY(c101, c100, c001, c000);

            c111.x = minmaxX(c111, c110, c101, c100);
            c111.y = minmaxY(c111, c110, c011, c010);
            c111.z = minmaxZ(c111, c110, c101, c011);

            return c111;
        }

        void main()
        {
            setOutput(propagatedMinima(outputCoords()));
        }
        `
    }
}

class ComputeCellShadowsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = false
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        outputShape: Shape3,
        dominantAxis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [depth, height, width] = outputShape

        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        float min3(float a, float b, float c)
        {
            return min(min(a, b), c);
        }

        ivec3 outputCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        vec4 minimaAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x, 0, 0);
        }

        vec4 maximaAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getB(p.z, p.y, p.x, 0, 0);
        }

        vec3 maxValues(ivec3 p)
        {
            vec4 c111 = maximaAt(p + ivec3(${voxelOffset( 0,  0,  0, dominantAxis, octant)}));

            return vec3(c111.x, c111.y, c111.z);
        }

        vec3 minValues(ivec3 p)
        {
            vec4 c011 = minimaAt(p + ivec3(${voxelOffset(-1,  0,  0, dominantAxis, octant)}));
            vec4 c101 = minimaAt(p + ivec3(${voxelOffset( 0, -1,  0, dominantAxis, octant)}));
            vec4 c110 = minimaAt(p + ivec3(${voxelOffset( 0,  0, -1, dominantAxis, octant)}));

            return vec3(c011.x, c101.y, c110.z);
        }

        bool cellShadow(vec3 minima, vec3 maxima)
        {
            return all(lessThan(maxima - minima, vec3(tolerance)));
        }

        void main()
        {
            ivec3 p = outputCoords();

            setOutput(float(cellShadow(minValues(p), maxValues(p))));
        }
        `
    }
}

function sweepInfo(dominantAxis: Axis, octant: Octant): { axis: 0 | 1 | 2, backwards: boolean }
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(dominantAxis, octant)
    const axis = permute[0]

    return { axis, backwards: reverse.includes(axis) }
}

function propagateFaceMinima(
    minima: tf.Tensor5D,
    dominantAxis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const { axis, backwards } = sweepInfo(dominantAxis, octant)
    const slices = unstackPacked(minima, axis)
    tf.dispose(minima)

    const shape = slices[0].shape as Shape3Packed
    const propagate = new PropagateFaceMinmaxProgram(shape, dominantAxis, octant)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const previous = i - step
        const next = runWebGLProgram(propagate, [slices[i], slices[previous]], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = next
    }

    const propagated = stackPacked(slices, axis) as tf.Tensor5D
    if (verbose) logMean('minimaPropagated', propagated)

    return propagated
}

function unidirectionalMinimaMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new ComputeFaceMinimaProgram(volume.shape as Shape3, dominantAxis, octant)
    const minima = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('minimaStart', minima)

    return propagateFaceMinima(minima, dominantAxis, octant, verbose)
}

function unidirectionalMinimaMapHollow(
    volume: tf.Tensor3D,
    holes: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new ComputeFaceMinimaProgram(volume.shape as Shape3, dominantAxis, octant, true)
    const minima = runWebGLProgram(program, [volume, holes], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('minimaStart', minima)

    return propagateFaceMinima(minima, dominantAxis, octant, verbose)
}

function unidirectionalMaximaMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new ComputeFaceMaximaProgram(volume.shape as Shape3, dominantAxis, octant)
    const maxima = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('maxima', maxima)

    return maxima
}

function unidirectionalShadowMap(
    minima: tf.Tensor5D,
    maxima: tf.Tensor5D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = minima.shape.slice(0, 3) as Shape3
    const program = new ComputeCellShadowsProgram(shape, dominantAxis, octant)
    const shadows = runWebGLProgram(program, [minima, maxima], 'float32', [[tolerance]], true) as tf.Tensor3D
    if (verbose) logMean('shadows', shadows)

    return shadows
}

export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const minimaMap = unidirectionalMinimaMap(volume, dominantAxis, octant)
    if (verbose) logMean('minimaMap', minimaMap)

    const maximaMap = unidirectionalMaximaMap(volume, dominantAxis, octant)
    if (verbose) logMean('maximaMap', maximaMap)

    const shadowsMap = unidirectionalShadowMap(minimaMap, maximaMap, dominantAxis, octant, tolerance)
    if (verbose) logMean('shadowsMap', shadowsMap)

    tf.dispose([minimaMap, maximaMap])

    return shadowsMap
}

export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const backwardOctant = reverseOctant(octant)

    const forwardShadowMap = computeUnidirectionalShadowMap(volume, dominantAxis, octant, tolerance)
    if (verbose) logMean('forwardShadowMap', forwardShadowMap)

    const backwardMinimaMap = unidirectionalMinimaMapHollow(volume, forwardShadowMap, dominantAxis, backwardOctant)
    const backwardMaximaMap = unidirectionalMaximaMap(volume, dominantAxis, backwardOctant)
    const backwardShadowMap = unidirectionalShadowMap(backwardMinimaMap, backwardMaximaMap, dominantAxis, backwardOctant, tolerance)
    if (verbose) logMean('backwardShadowMap', backwardShadowMap)

    tf.dispose([backwardMinimaMap, backwardMaximaMap])

    const shadowMap = tf.maximum(forwardShadowMap, backwardShadowMap) 
    if (verbose) logMean('shadowMap', shadowMap)

    tf.dispose([forwardShadowMap, backwardShadowMap])

    return shadowMap as tf.Tensor3D
}


export function computeBidirectionalBlockShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadowMap = computeBidirectionalShadowMap(volume, dominantAxis, octant, tolerance, verbose)
    if (blockSize === 1) return shadowMap

    const blockShadowMap = minPool3d(shadowMap, blockSize, blockSize, 'same') as tf.Tensor3D
    if (verbose) logMean('blockShadowMap', blockShadowMap)

    tf.dispose(shadowMap)

    return blockShadowMap
}

function logMean(name: string, tensor: tf.Tensor): void
{
    console.log(name, tf.tidy(() => tensor.mean([0, 1, 2]).dataSync()))
}

function runWebGLProgram(
    program: GPGPUProgram,
    inputs: tf.Tensor[],
    dtype?: tf.DataType,
    uniforms?: number[][],
    preventEagerUnpackingOfOutput?: boolean
): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(program, inputs, dtype, uniforms, preventEagerUnpackingOfOutput)

    return tf.engine().makeTensorFromTensorInfo(info)
}
