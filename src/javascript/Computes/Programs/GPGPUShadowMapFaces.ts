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
import { unstack3dPacked } from './unstack_packed_keepDims_webgl'
import { stack3dPacked } from './stack_packed_keepDims_webgl'
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

    for (const dimension of reverse) zyx[dimension] = 1 - zyx[dimension]

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

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

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

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

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

        bool insideVolume(ivec3 voxelCoords)
        {
            return 
                voxelCoords.x >= minCoords.x && voxelCoords.x <= maxCoords.x &&
                voxelCoords.y >= minCoords.y && voxelCoords.y <= maxCoords.y &&
                voxelCoords.z >= minCoords.z && voxelCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        float volumeAt(ivec3 voxelCoords)
        {
            return insideVolume(voxelCoords) ? getA(voxelCoords.z, voxelCoords.y, voxelCoords.x) : 0.0;
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

        CellValues cellValues(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            CellValues c;
            c.v000 = volumeAt(voxelCoords + ivec3(${voxelOffset(0, 0, 0, axis, octant)}));
            c.v100 = volumeAt(voxelCoords + ivec3(${voxelOffset(1, 0, 0, axis, octant)}));
            c.v010 = volumeAt(voxelCoords + ivec3(${voxelOffset(0, 1, 0, axis, octant)}));
            c.v001 = volumeAt(voxelCoords + ivec3(${voxelOffset(0, 0, 1, axis, octant)}));
            c.v011 = volumeAt(voxelCoords + ivec3(${voxelOffset(0, 1, 1, axis, octant)}));
            c.v101 = volumeAt(voxelCoords + ivec3(${voxelOffset(1, 0, 1, axis, octant)}));
            c.v110 = volumeAt(voxelCoords + ivec3(${voxelOffset(1, 1, 0, axis, octant)}));
            c.v111 = volumeAt(voxelCoords + ivec3(${voxelOffset(1, 1, 1, axis, octant)}));

            return c;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float computeFaceXMin(CellValues c)
        {
            return min4(c.v100, c.v110, c.v101, c.v111);
        }

        float computeFaceYMin(CellValues c)
        {
            return min4(c.v010, c.v110, c.v011, c.v111);
        }

        float computeFaceZMin(CellValues c)
        {
            return min4(c.v001, c.v011, c.v101, c.v111);
        }

        vec3 computeFaceMinima(ivec3 cellCoords)
        {
            CellValues c = cellValues(cellCoords);

            return vec3(
                computeFaceXMin(c), 
                computeFaceYMin(c), 
                computeFaceZMin(c)
            );
        }

        void main()
        {
            setOutput(vec4(computeFaceMinima(outputCoords()), 0.0));
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

        bool insideVolume(ivec3 voxelCoords)
        {
            return 
                voxelCoords.x >= minCoords.x && voxelCoords.x <= maxCoords.x &&
                voxelCoords.y >= minCoords.y && voxelCoords.y <= maxCoords.y &&
                voxelCoords.z >= minCoords.z && voxelCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        float volumeAt(ivec3 voxelCoords)
        {
            return insideVolume(voxelCoords) ? getA(voxelCoords.z, voxelCoords.y, voxelCoords.x) : 0.0;
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

        CellValues cellValues(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            CellValues c;
            c.v000 = volumeAt(voxelCoords + ivec3(${voxelOffset(0, 0, 0, axis, octant)}));
            c.v100 = volumeAt(voxelCoords + ivec3(${voxelOffset(1, 0, 0, axis, octant)}));
            c.v010 = volumeAt(voxelCoords + ivec3(${voxelOffset(0, 1, 0, axis, octant)}));
            c.v001 = volumeAt(voxelCoords + ivec3(${voxelOffset(0, 0, 1, axis, octant)}));
            c.v011 = volumeAt(voxelCoords + ivec3(${voxelOffset(0, 1, 1, axis, octant)}));
            c.v101 = volumeAt(voxelCoords + ivec3(${voxelOffset(1, 0, 1, axis, octant)}));
            c.v110 = volumeAt(voxelCoords + ivec3(${voxelOffset(1, 1, 0, axis, octant)}));
            c.v111 = volumeAt(voxelCoords + ivec3(${voxelOffset(1, 1, 1, axis, octant)}));

            return c;
        }

        float max4(float a, float b, float c, float d)
        {
            return max(max(max(a, b), c), d);
        }

        float computeFaceXMax(CellValues c)
        {
            return max4(c.v100, c.v110, c.v101, c.v111);
        }

        float computeFaceYMax(CellValues c)
        {
            return max4(c.v010, c.v110, c.v011, c.v111);
        }

        float computeFaceZMax(CellValues c)
        {
            return max4(c.v001, c.v011, c.v101, c.v111);
        }

        vec3 computeFaceMaxima(ivec3 cellCoords)
        {
            CellValues c = cellValues(cellCoords);

            return vec3(
                computeFaceXMax(c), 
                computeFaceYMax(c), 
                computeFaceZMax(c)
            );
        }

        void main()
        {
            setOutput(vec4(computeFaceMaxima(outputCoords()), 0.0));
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

        bool insideSlice(ivec3 sliceCoords)
        {
            return 
                ${(width  > 1) ? 'sliceCoords.x >= minCoords.x && sliceCoords.x <= maxCoords.x' : 'true'} &&
                ${(height > 1) ? 'sliceCoords.y >= minCoords.y && sliceCoords.y <= maxCoords.y' : 'true'} &&
                ${(depth  > 1) ? 'sliceCoords.z >= minCoords.z && sliceCoords.z <= maxCoords.z' : 'true'};
        }

        ivec3 outputCoords()
        {
            ivec5 sliceCoords = getOutputCoords();
            return ivec3(sliceCoords.z, sliceCoords.y, sliceCoords.x);
        }

        vec3 currentSliceAt(ivec3 sliceCoords)
        {
            return insideSlice(sliceCoords) ? getA(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0).xyz : vec3(0.0);
        }

        vec3 previousSliceAt(ivec3 sliceCoords)
        {
            return insideSlice(sliceCoords) ? getB(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0).xyz : vec3(0.0);
        }

        float computeFaceXMinmax(vec3 c111, vec3 c110, vec3 c101, vec3 c100)
        {
            float bottleneck = min(c110.z, max(c101.y, c100.z));
            return max(c111.x, bottleneck);
        }

        float computeFaceYMinmax(vec3 c111, vec3 c110, vec3 c011, vec3 c010)
        {
            float bottleneck = min(c110.z, max(c011.x, c010.z));
            return max(c111.y, bottleneck);
        }

        float computeFaceZMinmax(vec3 c111, vec3 c110, vec3 c101, vec3 c011)
        {
            float bottleneck = min(c110.z, min(c101.y, c011.x));
            return max(c111.z, bottleneck);
        }

        vec3 computeFaceMinmax(ivec3 sliceCoords)
        {
            vec3 c111 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0,  0, axis, octant)}));
            vec3 c011 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0,  0, axis, octant)}));
            vec3 c101 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1,  0, axis, octant)}));
            vec3 c001 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1,  0, axis, octant)}));
            vec3 c110 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0, -1, axis, octant)}));
            vec3 c010 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0, -1, axis, octant)}));
            vec3 c100 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1, -1, axis, octant)}));
            vec3 c000 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1, -1, axis, octant)}));

            c111.x = computeFaceXMinmax(c111, c110, c101, c100);
            c111.y = computeFaceYMinmax(c111, c110, c011, c010);

            c011.x = computeFaceXMinmax(c011, c010, c001, c000);
            c101.y = computeFaceYMinmax(c101, c100, c001, c000);
            c111.z = computeFaceZMinmax(c111, c110, c101, c011);

            return c111;
        }

        void main()
        {
            setOutput(vec4(computeFaceMinmax(outputCoords()), 0.0));
        }
        `
    }
}

class ComputeFaceShadowsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

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

        bool insideVolume(ivec3 cellCoords)
        {
            return 
                cellCoords.x >= minCoords.x && cellCoords.x <= maxCoords.x &&
                cellCoords.y >= minCoords.y && cellCoords.y <= maxCoords.y &&
                cellCoords.z >= minCoords.z && cellCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        vec3 faceMinmaxAt(ivec3 cellCoords)
        {
            cellCoords = clamp(cellCoords, minCoords, maxCoords);
            return getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).xyz;
        }

        vec3 faceMaximaAt(ivec3 cellCoords)
        {
            cellCoords = clamp(cellCoords, minCoords, maxCoords);
            return getB(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).xyz;
        }

        bvec3 computeFaceShadows(ivec3 cellCoords)
        {
            vec3 c111 = faceMaximaAt(cellCoords + ivec3(${cellOffset( 0,  0,  0, axis, octant)}));
            vec3 c011 = faceMinmaxAt(cellCoords + ivec3(${cellOffset(-1,  0,  0, axis, octant)}));
            vec3 c101 = faceMinmaxAt(cellCoords + ivec3(${cellOffset( 0, -1,  0, axis, octant)}));
            vec3 c110 = faceMinmaxAt(cellCoords + ivec3(${cellOffset( 0,  0, -1, axis, octant)}));

            bool faceXShadow = (c111.x - c011.x < tolerance);
            bool faceYShadow = (c111.y - c101.y < tolerance);
            bool faceZShadow = (c111.z - c110.z < tolerance);

            return bvec3(
                faceXShadow, 
                faceYShadow, 
                faceZShadow
            );
        }

        void main()
        {   
            setOutput(vec4(computeFaceShadows(outputCoords()), 1.0));
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

        bool insideVolume(ivec3 cellCoords)
        {
            return 
                cellCoords.x >= minCoords.x && cellCoords.x <= maxCoords.x &&
                cellCoords.y >= minCoords.y && cellCoords.y <= maxCoords.y &&
                cellCoords.z >= minCoords.z && cellCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        bool cellShadow(ivec3 cellCoords)
        {
            cellCoords = clamp(cellCoords, minCoords, maxCoords);
            return getA(cellCoords.z, cellCoords.y, cellCoords.x) > 0.5;
        }

        bvec3 computeFaceHoles(ivec3 cellCoords)
        {
            bool c000 = cellShadow(cellCoords + ivec3(${cellOffset(0, 0, 0, axis, octant)}));
            bool c100 = cellShadow(cellCoords + ivec3(${cellOffset(1, 0, 0, axis, octant)}));
            bool c010 = cellShadow(cellCoords + ivec3(${cellOffset(0, 1, 0, axis, octant)}));
            bool c001 = cellShadow(cellCoords + ivec3(${cellOffset(0, 0, 1, axis, octant)}));

            return bvec3(c000);
        }

        void main()
        {
            setOutput(vec4(computeFaceHoles(outputCoords()), 1.0));
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

        bool insideVolume(ivec3 cellCoords)
        {
            return 
                cellCoords.x >= minCoords.x && cellCoords.x <= maxCoords.x &&
                cellCoords.y >= minCoords.y && cellCoords.y <= maxCoords.y &&
                cellCoords.z >= minCoords.z && cellCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        vec3 faceMinimaAt(ivec3 cellCoords)
        {
            return getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).xyz;
        }

        bvec3 faceHolesAt(ivec3 cellCoords)
        {
            return greaterThan(getB(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).xyz, vec3(0.5));
        }

        void main()
        {
            ivec3 cellCoords = outputCoords();

            vec3 faceMinima = faceMinimaAt(cellCoords);
            bvec3 faceHoles = faceHolesAt(cellCoords);

            vec3 hollowMinima = vec3(
                faceHoles.x ? 0.0 : faceMinima.x,
                faceHoles.y ? 0.0 : faceMinima.y,
                faceHoles.z ? 0.0 : faceMinima.z
            );

            setOutput(vec4(hollowMinima, 0.0));
        }
        `
    }
}

export function computeFaceMinima(
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

export function computeFaceMinmax(
    faceMinima: tf.Tensor5D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)
    const dimension = permute[0]
    const backwards = reverse.includes(dimension)

    const slices = unstack3dPacked(faceMinima, dimension)
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

    const faceMinmax = stack3dPacked(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean('faceMinmax', faceMinmax)

    return faceMinmax as tf.Tensor5D
}

export function computeFaceMaxima(
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

export function computeFaceShadows(
    faceMinmax: tf.Tensor5D,
    faceMaxima: tf.Tensor5D,
    axis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = faceMaxima.shape as Shape3Packed
    const program = new ComputeFaceShadowsProgram(shape, axis, octant)
    const faceShadows = runWebGLProgram(program, [faceMinmax, faceMaxima], 'bool', [[tolerance]], true) 
    if (verbose) logMean('faceShadows', faceShadows)

    return faceShadows as tf.Tensor5D
}

export function computeFaceHoles(
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

export function hollowFaceMinima(
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
    const faceShadows = computeFaceShadows(faceMinmax, faceMaxima, axis, octant, tolerance, verbose)
    tf.dispose([faceMinmax, faceMaxima])

    const cellShadows = tf.all(faceShadows, [3, 4])
    tf.dispose(faceShadows)
    
    return cellShadows as tf.Tensor3D
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
    const faceHoles = computeFaceHoles(forwardShadows, axis, octant, verbose)

    const faceMinima = computeFaceMinima(volume, axis, backwardOctant, verbose)
    const hollowMinima = hollowFaceMinima(faceMinima, faceHoles, axis, backwardOctant, verbose)
    tf.dispose([faceMinima, faceHoles])

    const faceMinmax = computeFaceMinmax(hollowMinima, axis, backwardOctant, verbose)
    tf.dispose(hollowMinima)
    
    const faceMaxima = computeFaceMaxima(volume, axis, backwardOctant, verbose)
    const faceShadows = computeFaceShadows(faceMinmax, faceMaxima, axis, backwardOctant, tolerance, verbose)
    tf.dispose([faceMinmax, faceMaxima])

    const backwardShadows = tf.all(faceShadows, [3, 4])
    tf.dispose(faceShadows)

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
    const shadows = computeBidirectionalShadowMap(volume, axis, octant, tolerance, verbose)
    if (blockSize === 1) return shadows

    const blockShadows = minPool3d(shadows, blockSize, blockSize, 'same') as tf.Tensor3D
    tf.dispose(shadows)
    if (verbose) logMean('blockShadows', blockShadows)

    return blockShadows
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
