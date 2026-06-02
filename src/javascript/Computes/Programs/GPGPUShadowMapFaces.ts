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

    for (const dim of reverse) zyx[dim] = 1 - zyx[dim]

    return xyz(zyx)
}

/**
 * Offset for propagated face-map addressing. Reversal flips the signed offset.
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

class ComputeFaceMinimaProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth + 1, height + 1, width + 1, 2, 2]
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

        float volumeAt(ivec3 p)
        {
            return insideVolume(p) ? getA(p.z, p.y, p.x) : 0.0;
        }

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

        CellValues cellValues(ivec3 coords)
        {
            ivec3 base = coords - 1;

            CellValues c;
            c.v000 = volumeAt(base + ivec3(${voxelOffset(0, 0, 0, axis, octant)}));
            c.v100 = volumeAt(base + ivec3(${voxelOffset(1, 0, 0, axis, octant)}));
            c.v010 = volumeAt(base + ivec3(${voxelOffset(0, 1, 0, axis, octant)}));
            c.v001 = volumeAt(base + ivec3(${voxelOffset(0, 0, 1, axis, octant)}));
            c.v011 = volumeAt(base + ivec3(${voxelOffset(0, 1, 1, axis, octant)}));
            c.v101 = volumeAt(base + ivec3(${voxelOffset(1, 0, 1, axis, octant)}));
            c.v110 = volumeAt(base + ivec3(${voxelOffset(1, 1, 0, axis, octant)}));
            c.v111 = volumeAt(base + ivec3(${voxelOffset(1, 1, 1, axis, octant)}));

            return c;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float faceXMin(CellValues c)
        {
            return min4(c.v100, c.v110, c.v101, c.v111);
        }

        float faceYMin(CellValues c)
        {
            return min4(c.v010, c.v110, c.v011, c.v111);
        }

        float faceZMin(CellValues c)
        {
            return min4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            CellValues c = cellValues(outputCoords());

            setOutput(vec4(faceXMin(c), faceYMin(c), faceZMin(c), 0.0));
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
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth + 1, height + 1, width + 1, 2, 2]
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

        float volumeAt(ivec3 p)
        {
            return insideVolume(p) ? getA(p.z, p.y, p.x) : 0.0;
        }

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

        CellValues cellValues(ivec3 coords)
        {
            ivec3 base = coords - 1;

            CellValues c;
            c.v000 = volumeAt(base + ivec3(${voxelOffset(0, 0, 0, axis, octant)}));
            c.v100 = volumeAt(base + ivec3(${voxelOffset(1, 0, 0, axis, octant)}));
            c.v010 = volumeAt(base + ivec3(${voxelOffset(0, 1, 0, axis, octant)}));
            c.v001 = volumeAt(base + ivec3(${voxelOffset(0, 0, 1, axis, octant)}));
            c.v011 = volumeAt(base + ivec3(${voxelOffset(0, 1, 1, axis, octant)}));
            c.v101 = volumeAt(base + ivec3(${voxelOffset(1, 0, 1, axis, octant)}));
            c.v110 = volumeAt(base + ivec3(${voxelOffset(1, 1, 0, axis, octant)}));
            c.v111 = volumeAt(base + ivec3(${voxelOffset(1, 1, 1, axis, octant)}));

            return c;
        }

        float max4(float a, float b, float c, float d)
        {
            return max(max(max(a, b), c), d);
        }

        float faceXMax(CellValues c)
        {
            return max4(c.v100, c.v110, c.v101, c.v111);
        }

        float faceYMax(CellValues c)
        {
            return max4(c.v010, c.v110, c.v011, c.v111);
        }

        float faceZMax(CellValues c)
        {
            return max4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            CellValues c = cellValues(outputCoords());

            setOutput(vec4(faceXMax(c), faceYMax(c), faceZMax(c), 0.0));
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
        shape: Shape3Packed,
        axis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = shape
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

        vec3 currentSliceAt(ivec3 p)
        {
            return insideVolume(p) ? getA(p.z, p.y, p.x, 0, 0).xyz : vec3(0.0);
        }

        vec3 previousSliceAt(ivec3 p)
        {
            return insideVolume(p) ? getB(p.z, p.y, p.x, 0, 0).xyz : vec3(0.0);
        }

        float faceXMinmax(vec3 c111, vec3 c110, vec3 c101, vec3 c100)
        {
            return max(c111.x, min(c110.z, max(c101.y, c100.z)));
        }

        float faceYMinmax(vec3 c111, vec3 c110, vec3 c011, vec3 c010)
        {
            return max(c111.y, min(c110.z, max(c011.x, c010.z)));
        }

        float faceZMinmax(vec3 c111, vec3 c110, vec3 c101, vec3 c011)
        {
            return max(c111.z, min(c110.z, min(c101.y, c011.x)));
        }

        vec3 cellMinmaxAt(ivec3 p)
        {
            vec3 c111 =  currentSliceAt(p + ivec3(${sliceOffset( 0,  0,  0, axis, octant)}));
            vec3 c011 =  currentSliceAt(p + ivec3(${sliceOffset(-1,  0,  0, axis, octant)}));
            vec3 c101 =  currentSliceAt(p + ivec3(${sliceOffset( 0, -1,  0, axis, octant)}));
            vec3 c001 =  currentSliceAt(p + ivec3(${sliceOffset(-1, -1,  0, axis, octant)}));
            vec3 c110 = previousSliceAt(p + ivec3(${sliceOffset( 0,  0, -1, axis, octant)}));
            vec3 c010 = previousSliceAt(p + ivec3(${sliceOffset(-1,  0, -1, axis, octant)}));
            vec3 c100 = previousSliceAt(p + ivec3(${sliceOffset( 0, -1, -1, axis, octant)}));
            vec3 c000 = previousSliceAt(p + ivec3(${sliceOffset(-1, -1, -1, axis, octant)}));

            c011.x = faceXMinmax(c011, c010, c001, c000);
            c101.y = faceYMinmax(c101, c100, c001, c000);

            c111.x = faceXMinmax(c111, c110, c101, c100);
            c111.y = faceYMinmax(c111, c110, c011, c010);
            c111.z = faceZMinmax(c111, c110, c101, c011);

            return c111;
        }

        void main()
        {
            setOutput(vec4(cellMinmaxAt(outputCoords()), 0.0));
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
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = shape
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
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        vec3 faceMinmaxAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x, 0, 0).xyz;
        }

        vec3 faceMaximaAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getB(p.z, p.y, p.x, 0, 0).xyz;
        }

        bool cellShadow(ivec3 p)
        {
            vec3 faceMinmax = faceMinmaxAt(p);
            vec3 faceMaxima = faceMaximaAt(p);
            
            return all(lessThan(faceMaxima - faceMinmax, vec3(tolerance)));
        }

        void main()
        {
            setOutput(float(cellShadow(outputCoords())));
        }
        `
    }
}

class ComputeFaceHolesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth, height, width, 2, 2]
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

        bool cellShadow(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x) > 0.5;
        }

        bvec3 faceHoles(ivec3 p)
        {
            bool c000 = cellShadow(p + ivec3(${cellOffset(0, 0, 0, axis, octant)}));
            bool c100 = cellShadow(p + ivec3(${cellOffset(1, 0, 0, axis, octant)}));
            bool c010 = cellShadow(p + ivec3(${cellOffset(0, 1, 0, axis, octant)}));
            bool c001 = cellShadow(p + ivec3(${cellOffset(0, 0, 1, axis, octant)}));

            return bvec3(c000 && c100, c000 && c010, c000 && c001);
        }

        void main()
        {
            setOutput(vec4(faceHoles(outputCoords()), 0.0));
        }
        `
    }
}

class HollowFaceMinimaProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: Shape3Packed,
        axis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [depth, height, width, ] = shape

        this.outputShape = shape
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

        vec3 faceMinimaAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x, 0, 0).xyz;
        }

        vec3 faceHolesAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getB(p.z, p.y, p.x, 0, 0).xyz;
        }

        void main()
        {
            ivec3 p = outputCoords();
            vec3 hollowMinima = mix(faceMinimaAt(p), vec3(0.0), faceHolesAt(p));

            setOutput(vec4(hollowMinima, 0.0));
        }
        `
    }
}

function computeFaceMinima(
    volume: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new ComputeFaceMinimaProgram(volume.shape, axis, octant)
    const faceMinima = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('faceMinima', faceMinima)

    return faceMinima
}

function computeFaceMaxima(
    volume: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new ComputeFaceMaximaProgram(volume.shape, axis, octant)
    const faceMaxima = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('faceMaxima', faceMaxima)

    return faceMaxima
}

function computeFaceMinmax(
    faceMinima: tf.Tensor5D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)
    const dimension = permute[0]
    const backwards = reverse.includes(dimension)

    const slices = unstackPacked(faceMinima, dimension)
    const shape = slices[0].shape as Shape3Packed
    const propagate = new PropagateFaceMinmaxProgram(shape, axis, octant)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    const faceMinmax = stackPacked(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean('faceMinmax', faceMinmax)

    return faceMinmax as tf.Tensor5D
}

function computeCellShadows(
    faceMinmax: tf.Tensor5D,
    faceMaxima: tf.Tensor5D,
    axis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = faceMinmax.shape.slice(0, 3) as Shape3
    const program = new ComputeCellShadowsProgram(shape, axis, octant)
    const cellShadows = runWebGLProgram(program, [faceMinmax, faceMaxima], 'bool', [[tolerance]], true) 
    if (verbose) logMean('cellShadows', cellShadows)

    return cellShadows as tf.Tensor3D
}

function computeFaceHoles(
    cellShadows: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new ComputeFaceHolesProgram(cellShadows.shape, axis, octant)
    const faceHoles = runWebGLProgram(program, [cellShadows], 'bool', [], true) as tf.Tensor5D
    if (verbose) logMean('faceHoles', faceHoles)

    return faceHoles
}

function hollowFaceMinima(
    faceMinima: tf.Tensor5D,
    faceHoles: tf.Tensor5D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D  
{
    const shape = faceMinima.shape as Shape3Packed
    const program = new HollowFaceMinimaProgram(shape, axis, octant)
    const hollowMinima = runWebGLProgram(program, [faceMinima, faceHoles], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('hollowMinima', hollowMinima)

    return hollowMinima
}

export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const faceMinima = computeFaceMinima(volume, axis, octant, verbose)
    const faceMinmax = computeFaceMinmax(faceMinima, axis, octant, verbose)
    tf.dispose(faceMinima)
    
    const faceMaxima = computeFaceMaxima(volume, axis, octant, verbose)
    const cellShadows = computeCellShadows(faceMinmax, faceMaxima, axis, octant, tolerance, verbose)
    tf.dispose([faceMinmax, faceMaxima])
    
    return cellShadows
}

export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const forwardShadows = computeUnidirectionalShadowMap(volume, axis, octant, tolerance, verbose)

    const backwardOctant = reverseOctant(octant)
    const faceHoles = computeFaceHoles(forwardShadows, axis, backwardOctant, verbose)

    const faceMinima = computeFaceMinima(volume, axis, backwardOctant, verbose)
    const hollowMinima = hollowFaceMinima(faceMinima, faceHoles, axis, backwardOctant, verbose)
    tf.dispose(faceHoles)

    const faceMinmax = computeFaceMinmax(hollowMinima, axis, backwardOctant, verbose)
    tf.dispose(faceMinima)
    
    const faceMaxima = computeFaceMaxima(volume, axis, backwardOctant, verbose)
    const backwardShadows = computeCellShadows(faceMinmax, faceMaxima, axis, backwardOctant, tolerance, verbose)
    tf.dispose([faceMinmax, faceMaxima])

    const shadows = tf.logicalOr(forwardShadows, backwardShadows)
    tf.dispose([forwardShadows, backwardShadows])
    if (verbose) logMean('shadows', shadows)
    
    return shadows as tf.Tensor3D
}

export function computeBidirectionalBlockShadowMap(
    volume: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadowMap = computeBidirectionalShadowMap(volume, axis, octant, tolerance, verbose)
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
