
import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { stack3dPacked } from './stack3dPacked'
import { unstack3dPacked } from './unstack3dPacked'
import { minPool3d } from './pool3d'
import { avgPool3dPacked, minPool3dPacked, maxPool3dPacked } from './pool3dPacked';
import * as su from '../../Utils/ShadowMapUtils'

function xyz(zyx: [number, number, number]): string
{
    return [zyx[2], zyx[1], zyx[0]].join(', ')
}

function voxelOffset(
    x: number,
    y: number,
    z: number,
    axis: su.Axis,
    octant: su.Octant
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)
    
    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

    return xyz(zyx)
}

function sliceOffset(
    x: number,
    y: number,
    z: number,
    axis: su.Axis,
    octant: su.Octant
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)
    
    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

    zyx[permute[0]] = 0

    return xyz(zyx)
}

function cellOffset(
    x: number,
    y: number,
    z: number,
    axis: su.Axis,
    octant: su.Octant
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)
    
    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = 1 - zyx[dimension]

    return xyz(zyx)
}

class ComputeEdgeMarginsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth, height, width, 2, 2]
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
            ivec5 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        float volumeAt(ivec3 voxelCoords)
        {
            return insideVolume(voxelCoords) ? getA(voxelCoords.z, voxelCoords.y, voxelCoords.x) : 0.0;
        }

        vec4 computeEdgeMargins(ivec3 voxelCoords)
        {
            float v111 = volumeAt(voxelCoords + ivec3(${voxelOffset( 0,  0,  0, axis, octant)}));
            float v000 = volumeAt(voxelCoords + ivec3(${voxelOffset(-1, -1, -1, axis, octant)}));
            float v010 = volumeAt(voxelCoords + ivec3(${voxelOffset(-1,  0, -1, axis, octant)}));
            float v100 = volumeAt(voxelCoords + ivec3(${voxelOffset( 0, -1, -1, axis, octant)}));
            float v110 = volumeAt(voxelCoords + ivec3(${voxelOffset( 0,  0, -1, axis, octant)}));

            return vec4(v000, v010, v100, v110) - v111;
        }

        void main()
        {
            setOutput(computeEdgeMargins(outputCoords()));
        }
        `
    }
}

class computeEdgeHolesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth - 1, height - 1, width - 1, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec5 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        float cellShadowAt(ivec3 cellCoords)
        {
            cellCoords = clamp(cellCoords, minCoords, maxCoords);
            return step(0.5, getA(cellCoords.z, cellCoords.y, cellCoords.x));
        }

        vec4 computeEdgeHoles(ivec3 voxelCoords)
        {
            float v000 = cellShadowAt(voxelCoords + ivec3(${cellOffset(-1, -1, -1, axis, octant)}));
            float v010 = cellShadowAt(voxelCoords + ivec3(${cellOffset(-1,  0, -1, axis, octant)}));
            float v100 = cellShadowAt(voxelCoords + ivec3(${cellOffset( 0, -1, -1, axis, octant)}));
            float v110 = cellShadowAt(voxelCoords + ivec3(${cellOffset( 0,  0, -1, axis, octant)}));

            return vec4(v000, v010, v100, v110);
        }

        void main()
        {
            setOutput(computeEdgeHoles(outputCoords()));
        }
        `
    }
}

class PropagateEdgeMarginsProgram implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: [number, number, number, 2, 2]
    userCode: string

    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        shape: [number, number, number, 2, 2],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++'
    ) {
        const [depth, height, width, ] = shape

        this.outputShape = [depth, height, width, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec5 sliceCoords = getOutputCoords();
            return ivec3(sliceCoords.z, sliceCoords.y, sliceCoords.x);
        }

        vec4 currentSliceAt(ivec3 sliceCoords)
        {
            sliceCoords = clamp(sliceCoords, minCoords, maxCoords);
            return getA(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0);
        }

        vec4 previousSliceAt(ivec3 sliceCoords)
        {
            sliceCoords = clamp(sliceCoords, minCoords, maxCoords);
            return getB(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0);
        }

        float min4(vec4 v)
        {
            return min(min(min(v.x, v.y), v.z), v.w);
        }

        vec4 propagateEdgeMargins(ivec3 sliceCoords)
        {
            vec4 v111 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0,  0, axis, octant)}));
            vec4 v110 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0, -1, axis, octant)}));
            vec4 v100 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1, -1, axis, octant)}));
            vec4 v010 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0, -1, axis, octant)}));
            vec4 v000 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1, -1, axis, octant)}));

            vec4 bottlenecks = vec4(min4(v000), min4(v010), min4(v100), min4(v110));
            v111 += max(bottlenecks, 0.0);

            return v111;
        }

        void main()
        {
            setOutput(propagateEdgeMargins(outputCoords()));
        }
        `
    }
}

class PropagateHollowEdgeMarginsProgram implements GPGPUProgram 
{
    variableNames = ['A', 'B', 'C']
    outputShape: [number, number, number, 2, 2]
    userCode: string

    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        shape: [number, number, number, 2, 2],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++'
    ) {
        const [depth, height, width, ] = shape

        this.outputShape = [depth, height, width, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec5 sliceCoords = getOutputCoords();
            return ivec3(sliceCoords.z, sliceCoords.y, sliceCoords.x);
        }

        vec4 currentSliceAt(ivec3 sliceCoords)
        {
            sliceCoords = clamp(sliceCoords, minCoords, maxCoords);
            return getA(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0);
        }

        vec4 previousSliceAt(ivec3 sliceCoords)
        {
            sliceCoords = clamp(sliceCoords, minCoords, maxCoords);
            return getB(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0);
        }

        vec4 edgeHoles(ivec3 sliceCoords)
        {
            ivec3 voxelCoords = sliceCoords;
            voxelCoords.${axis} = slice;

            return step(0.5, getC(voxelCoords.z, voxelCoords.y, voxelCoords.x, 0, 0));
        }

        float min4(vec4 v)
        {
            return min(min(min(v.x, v.y), v.z), v.w);
        }

        vec4 propagateHollowEdgeMargins(ivec3 sliceCoords)
        {
            vec4 holes = edgeHoles(sliceCoords);

            vec4 v111 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0,  0, axis, octant)}));
            vec4 v110 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0, -1, axis, octant)}));
            vec4 v100 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1, -1, axis, octant)}));
            vec4 v010 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0, -1, axis, octant)}));
            vec4 v000 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1, -1, axis, octant)}));

            vec4 bottlenecks = vec4(min4(v000), min4(v010), min4(v100), min4(v110));
            v111 += mix(max(bottlenecks, 0.0), bottlenecks, holes);

            return v111;
        }

        void main()
        {
            setOutput(propagateHollowEdgeMargins(outputCoords()));
        }
        `
    }
}

class ComputeVertexShadowsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = true
    packedOutput = false
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(shape: [number, number, number, 2, 2]) 
    {
        const [depth, height, width, ] = shape
    
        this.outputShape = [depth, height, width]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec3 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        vec4 edgeMarginsAt(ivec3 voxelCoords)
        {
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x, 0, 0);
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        bool computeVertexShadow(ivec3 voxelCoords)
        {
            vec4 edgeMargins = edgeMarginsAt(voxelCoords);
            return all(greaterThan(edgeMargins, vec4(-tolerance)));
        }

        void main()
        {
            setOutput(float(computeVertexShadow(outputCoords())));
        }
        `
    }
}

class ComputeCellShadowsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth + 1, height + 1, width + 1]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});
            
        ivec3 outputCoords()
        {
            ivec3 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        bool vertexShadowAt(ivec3 voxelCoords)
        {
            voxelCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x) > 0.5;
        }

        bool computeCellShadow(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            return (
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(1, 1, 1, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(1, 0, 1, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(0, 1, 1, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(0, 0, 1, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(1, 1, 0, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(1, 0, 0, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(0, 1, 0, axis, octant)})) 
            );
        }

        void main()
        {
            setOutput(float(computeCellShadow(outputCoords())));
        }
        `
    }
}

function computeEdgeMargins(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = volume.shape
    const program = new ComputeEdgeMarginsProgram(shape, axis, octant)
    const edgeMargins = runWebGLProgram(program, [volume], 'float32', [], true) 
    if (verbose) logMean('edgeMargins', edgeMargins)

    return edgeMargins as tf.Tensor5D
}

function computeEdgeHoles(
    cellShadows: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = cellShadows.shape
    const program = new computeEdgeHolesProgram(shape, axis, octant)
    const edgeHoles = runWebGLProgram(program, [cellShadows], 'bool', [], true) 
    if (verbose) logMean('edgeHoles', edgeHoles)

    return edgeHoles as tf.Tensor5D
}

function propagateEdgeMargins(
    edgeMargins: tf.Tensor5D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    // const dimension = su.axisToDimension(axis)
    // const sign = su.getOctantSign(octant, dimension)
    // const backwards = sign === '-'

    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)
    const dimension = permute[0]
    const backwards = reverse.includes(dimension)

    const slices = unstack3dPacked(edgeMargins, dimension)
    const shape = slices[0].shape as [number, number, number, 2, 2]
    const propagate = new PropagateEdgeMarginsProgram(shape, axis, octant)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram( propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    const propagatedEdgeMargins = stack3dPacked(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean('propagatedEdgeMargins', propagatedEdgeMargins)

    return propagatedEdgeMargins as tf.Tensor5D
}

function propagateHollowEdgeMargins(
    edgeMargins: tf.Tensor5D,
    edgeHoles: tf.Tensor5D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    // const dimension = su.axisToDimension(axis)
    // const sign = su.getOctantSign(octant, dimension)
    // const backwards = sign === '-'

    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)
    const dimension = permute[0]
    const backwards = reverse.includes(dimension)

    const slices = unstack3dPacked(edgeMargins, dimension)
    const shape = slices[0].shape as [number, number, number, 2, 2]
    const propagate = new PropagateHollowEdgeMarginsProgram(shape, axis, octant)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram( propagate, [slices[i], slices[i-step], edgeHoles], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    const propagatedEdgeMargins = stack3dPacked(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean('propagatedEdgeMargins', propagatedEdgeMargins)

    return propagatedEdgeMargins as tf.Tensor5D
}

function computeVertexShadows(
    edgeMargins: tf.Tensor5D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = edgeMargins.shape as [number, number, number, 2, 2]
    const program = new ComputeVertexShadowsProgram(shape)
    const vertexShadows = runWebGLProgram(program, [edgeMargins], 'bool', [[tolerance]], true) 
    if (verbose) logMean('vertexShadows', vertexShadows)

    return vertexShadows as tf.Tensor3D
}

function computeCellShadows(
    vertexShadows: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = vertexShadows.shape
    const program = new ComputeCellShadowsProgram(shape, axis, octant)
    const cellShadows = runWebGLProgram(program, [vertexShadows], 'bool', [], true) 
    if (verbose) logMean('cellShadows', cellShadows)

    return cellShadows as tf.Tensor3D
}

export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const edgeMargins = computeEdgeMargins(volume, axis, octant, verbose)
    const propagatedEdgeMargins = propagateEdgeMargins(edgeMargins, axis, octant)
    tf.dispose(edgeMargins)
    
    const vertexShadows = computeVertexShadows(propagatedEdgeMargins, tolerance)
    tf.dispose(propagatedEdgeMargins)

    const cellShadows = computeCellShadows(vertexShadows, axis, octant, verbose)
    tf.dispose(vertexShadows)

    return cellShadows
}

export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    // Forward
    const forwardOctant = octant
    const forwardEdgeMargins = computeEdgeMargins(volume, axis, forwardOctant, verbose)
    const forwardPropagatedEdgeMargins = propagateEdgeMargins(forwardEdgeMargins, axis, forwardOctant, verbose)
    tf.dispose(forwardEdgeMargins)
    
    const forwardVertexShadows = computeVertexShadows(forwardPropagatedEdgeMargins, tolerance, verbose)
    tf.dispose(forwardPropagatedEdgeMargins)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, axis, forwardOctant, verbose)
    tf.dispose(forwardVertexShadows)
    if (verbose) logMean('forwardCellShadows', forwardCellShadows)

    // Backward
    const backwardOctant = su.reverseOctant(forwardOctant)
    const backwardEdgeHoles = computeEdgeHoles(forwardCellShadows, axis, backwardOctant, verbose)

    const backwardEdgeMargins = computeEdgeMargins(volume, axis, backwardOctant, verbose)
    const backwardPropagatedEdgeMargins = propagateHollowEdgeMargins(backwardEdgeMargins, backwardEdgeHoles, axis, backwardOctant, verbose)
    tf.dispose([backwardEdgeMargins, backwardEdgeHoles])

    const backwardVertexShadows = computeVertexShadows(backwardPropagatedEdgeMargins, tolerance, verbose)
    tf.dispose(backwardPropagatedEdgeMargins)

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, axis, backwardOctant, verbose)
    tf.dispose(backwardVertexShadows)
    if (verbose) logMean('backwardCellShadows', backwardCellShadows)

    // Bidirectional 
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows)
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean('bidirectionalCellShadows', bidirectionalCellShadows)

    return bidirectionalCellShadows as tf.Tensor3D
}

export function computeBidirectionalBlockShadowMap(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadows = computeBidirectionalShadowMap(volume, axis, octant, tolerance, verbose)
    if (blockSize === 1) return shadows

    const blockShadows = minPool3d(shadows, blockSize, blockSize, 'same') as tf.Tensor3D
    tf.dispose(shadows)
    if (verbose) logMean('blockShadowsMask', blockShadows)

    return blockShadows
}

export function computeBidirectionalBlockShadowMap2(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    // Forward
    const forwardOctant = octant
    const forwardEdgeMargins = computeEdgeMargins(volume, axis, forwardOctant, verbose)
    const forwardMinEdgeMargins = minPool3dPacked(forwardEdgeMargins, blockSize, blockSize, 'valid')
    tf.dispose(forwardEdgeMargins)

    const forwardPropagatedEdgeMargins = propagateEdgeMargins(forwardMinEdgeMargins, axis, forwardOctant, verbose)
    tf.dispose(forwardMinEdgeMargins)
    
    const forwardVertexShadows = computeVertexShadows(forwardPropagatedEdgeMargins, tolerance, verbose)
    tf.dispose(forwardPropagatedEdgeMargins)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, axis, forwardOctant, verbose)
    tf.dispose(forwardVertexShadows)
    if (verbose) logMean('forwardCellShadows', forwardCellShadows)

    // Backward
    const backwardOctant = su.reverseOctant(forwardOctant)
    const backwardEdgeHoles = computeEdgeHoles(forwardCellShadows, axis, backwardOctant, verbose)

    const backwardEdgeMargins = computeEdgeMargins(volume, axis, backwardOctant, verbose)
    const backwardMinEdgeMargins = minPool3dPacked(backwardEdgeMargins, blockSize, blockSize, 'valid')
    tf.dispose(backwardEdgeMargins)

    const backwardPropagatedEdgeMargins = propagateHollowEdgeMargins(backwardMinEdgeMargins, backwardEdgeHoles, axis, backwardOctant, verbose)
    tf.dispose([backwardMinEdgeMargins, backwardEdgeHoles])

    const backwardVertexShadows = computeVertexShadows(backwardPropagatedEdgeMargins, tolerance, verbose)
    tf.dispose(backwardPropagatedEdgeMargins)

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, axis, backwardOctant, verbose)
    tf.dispose(backwardVertexShadows)
    if (verbose) logMean('backwardCellShadows', backwardCellShadows)

    // Bidirectional 
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows)
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean('bidirectionalCellShadows', bidirectionalCellShadows)

    return bidirectionalCellShadows as tf.Tensor3D
}

export function computeBidirectionalBlockShadowMap3(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    // Reduce
    const vertexValues = minPool3d(volume, blockSize, blockSize, 'valid')

    // Forward
    const forwardOctant = octant
    const forwardEdgeMargins = computeEdgeMargins(vertexValues, axis, forwardOctant, verbose)
    const forwardPropagatedEdgeMargins = propagateEdgeMargins(forwardEdgeMargins, axis, forwardOctant, verbose)
    tf.dispose(forwardEdgeMargins)
    
    const forwardVertexShadows = computeVertexShadows(forwardPropagatedEdgeMargins, tolerance, verbose)
    tf.dispose(forwardPropagatedEdgeMargins)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, axis, forwardOctant, verbose)
    tf.dispose(forwardVertexShadows)
    if (verbose) logMean('forwardCellShadows', forwardCellShadows)

    // Backward
    const backwardOctant = su.reverseOctant(forwardOctant)
    const backwardEdgeHoles = computeEdgeHoles(forwardCellShadows, axis, backwardOctant, verbose)

    const backwardEdgeMargins = computeEdgeMargins(vertexValues, axis, backwardOctant, verbose)
    tf.dispose(vertexValues)

    const backwardPropagatedEdgeMargins = propagateHollowEdgeMargins(backwardEdgeMargins, backwardEdgeHoles, axis, backwardOctant, verbose)
    tf.dispose([backwardEdgeMargins, backwardEdgeHoles])

    const backwardVertexShadows = computeVertexShadows(backwardPropagatedEdgeMargins, tolerance, verbose)
    tf.dispose(backwardPropagatedEdgeMargins)

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, axis, backwardOctant, verbose)
    tf.dispose(backwardVertexShadows)
    if (verbose) logMean('backwardCellShadows', backwardCellShadows)

    // Bidirectional 
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows)
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean('bidirectionalCellShadows', bidirectionalCellShadows)

    return bidirectionalCellShadows as tf.Tensor3D
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
