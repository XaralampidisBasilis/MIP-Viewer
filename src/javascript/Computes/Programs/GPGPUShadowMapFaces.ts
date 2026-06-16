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
import * as su from '../../Utils/ShadowMapUtils'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstack3dPacked } from './unstack3dPacked'
import { stack3dPacked } from './stack3dPacked'
import { minPool3d } from './pool3d'
import { type Sign, type Octant, type Axis } from '../../Utils/ShadowMapUtils'

function xyz(zyx: [number, number, number]): string
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
    directionOctant: Octant
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(dominantAxis, directionOctant)

    const zyx = su.applyPermutation([z, y, x], permute)

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
    directionOctant: Octant
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(dominantAxis, directionOctant)

    const zyx = su.applyPermutation([z, y, x], permute)

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
    directionOctant: Octant
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(dominantAxis, directionOctant)

    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

    zyx[permute[0]] = 0

    return xyz(zyx)
}

/**
 * Logs the mean over spatial axes without downloading the full tensor.
 */
function logMean3d(name: string, tensor: tf.Tensor): void
{
    console.log(name, tf.tidy(() => tensor.mean([0, 1, 2]).dataSync()))
}

/**
 * Runs a custom WebGL program and wraps the backend TensorInfo as a Tensor.
 */
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

class HollowFaceMinValuesProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(shape: [number, number, number, 2, 2]) 
    {
        const [depth, height, width, ] = shape

        this.outputShape = shape
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validCellCoords(ivec3 cellCoords)
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

        vec3 faceMinValueAt(ivec3 cellCoords)
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

            vec3 faceMinValues = faceMinValueAt(cellCoords);
            bvec3 faceHoles    = faceHolesAt(cellCoords);
            
            faceMinValues.x = faceHoles.x ? NEG_INF : faceMinValues.x;
            faceMinValues.y = faceHoles.y ? NEG_INF : faceMinValues.y;
            faceMinValues.z = faceHoles.z ? NEG_INF : faceMinValues.z;

            setOutput(vec4(faceMinValues, 0.0));
        }
        `
    }
}

class ComputeFaceMinValuesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: [number, number, number],
        dominantAxis: Axis = 'z',
        directionOctant: Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth + 1, height + 1, width + 1, 2, 2]
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validVoxelCoords(ivec3 voxelCoords)
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

        float voxelValueAt(ivec3 voxelCoords)
        {
            voxelCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x);
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

        CellValues sampleCellValues(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            CellValues c;

            c.v000 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(0, 0, 0, dominantAxis, directionOctant)}));
            c.v100 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(1, 0, 0, dominantAxis, directionOctant)}));
            c.v010 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(0, 1, 0, dominantAxis, directionOctant)}));
            c.v001 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(0, 0, 1, dominantAxis, directionOctant)}));
            c.v011 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(0, 1, 1, dominantAxis, directionOctant)}));
            c.v101 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(1, 0, 1, dominantAxis, directionOctant)}));
            c.v110 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(1, 1, 0, dominantAxis, directionOctant)}));
            c.v111 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(1, 1, 1, dominantAxis, directionOctant)}));

            return c;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float computeXFaceMinValue(CellValues c)
        {
            return min4(c.v100, c.v110, c.v101, c.v111);
        }

        float computeYFaceMinValue(CellValues c)
        {
            return min4(c.v010, c.v110, c.v011, c.v111);
        }

        float computeZFaceMinValue(CellValues c)
        {
            return min4(c.v001, c.v011, c.v101, c.v111);
        }

        vec3 computeFaceMinValues(ivec3 cellCoords)
        {
            CellValues c = sampleCellValues(cellCoords);

            return vec3(computeXFaceMinValue(c), computeYFaceMinValue(c), computeZFaceMinValue(c));
        }

        void main()
        {
            setOutput(vec4(computeFaceMinValues(outputCoords()), 0.0));
        }
        `
    }
}

class ComputeFaceMaxValuesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: [number, number, number],
        dominantAxis: Axis = 'z',
        directionOctant: Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth + 1, height + 1, width + 1, 2, 2]
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validVoxelCoords(ivec3 voxelCoords)
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

        float voxelValueAt(ivec3 voxelCoords)
        {
            voxelCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x);
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

        CellValues sampleCellValues(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            CellValues c;

            c.v000 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(0, 0, 0, dominantAxis, directionOctant)}));
            c.v100 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(1, 0, 0, dominantAxis, directionOctant)}));
            c.v010 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(0, 1, 0, dominantAxis, directionOctant)}));
            c.v001 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(0, 0, 1, dominantAxis, directionOctant)}));
            c.v011 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(0, 1, 1, dominantAxis, directionOctant)}));
            c.v101 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(1, 0, 1, dominantAxis, directionOctant)}));
            c.v110 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(1, 1, 0, dominantAxis, directionOctant)}));
            c.v111 = voxelValueAt(voxelCoords + ivec3(${voxelOffset(1, 1, 1, dominantAxis, directionOctant)}));

            return c;
        }

        float max4(float a, float b, float c, float d)
        {
            return max(max(max(a, b), c), d);
        }

        float computeXFaceMaxValue(CellValues c)
        {
            return max4(c.v100, c.v110, c.v101, c.v111);
        }

        float computeYFaceMaxValue(CellValues c)
        {
            return max4(c.v010, c.v110, c.v011, c.v111);
        }

        float computeZFaceMaxValue(CellValues c)
        {
            return max4(c.v001, c.v011, c.v101, c.v111);
        }

        vec3 computeFaceMaxValues(ivec3 cellCoords)
        {
            CellValues c = sampleCellValues(cellCoords);

            return vec3(computeXFaceMaxValue(c), computeYFaceMaxValue(c), computeZFaceMaxValue(c));
        }

        void main()
        {
            setOutput(vec4(computeFaceMaxValues(outputCoords()), 0.0));
        }
        `
    }
}

class PropagateFaceMinmaxValuesProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        shape: [number, number, number, 2, 2],
        dominantAxis: Axis = 'z',
        directionOctant: Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = shape
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validSliceCoords(ivec3 sliceCoords)
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
            return validSliceCoords(sliceCoords) ? 
                getA(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0).xyz : vec3(NEG_INF);
        }

        vec3 previousSliceAt(ivec3 sliceCoords)
        {
            return validSliceCoords(sliceCoords) ? 
                getB(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0).xyz : vec3(NEG_INF);
        }

        float computeXFaceMinmaxValue(vec3 c111, vec3 c110, vec3 c101, vec3 c100)
        {
            float minValues = min(c110.z, max(c101.y, c100.z));
            return max(c111.x, minValues);
        }

        float computeYFaceMinmaxValue(vec3 c111, vec3 c110, vec3 c011, vec3 c010)
        {
            float minValues = min(c110.z, max(c011.x, c010.z));
            return max(c111.y, minValues);
        }

        float computeZFaceMinmaxValue(vec3 c111, vec3 c110, vec3 c101, vec3 c011)
        {
            float minValues = min(c110.z, min(c101.y, c011.x));
            return max(c111.z, minValues);
        }

        vec3 computeFaceMinmaxValues(ivec3 sliceCoords)
        {
            vec3 c111 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0,  0, dominantAxis, directionOctant)}));
            vec3 c011 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0,  0, dominantAxis, directionOctant)}));
            vec3 c101 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1,  0, dominantAxis, directionOctant)}));
            vec3 c001 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1,  0, dominantAxis, directionOctant)}));
            vec3 c110 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0, -1, dominantAxis, directionOctant)}));
            vec3 c010 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0, -1, dominantAxis, directionOctant)}));
            vec3 c100 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1, -1, dominantAxis, directionOctant)}));
            vec3 c000 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1, -1, dominantAxis, directionOctant)}));

            c011.x = computeXFaceMinmaxValue(c011, c010, c001, c000);
            c101.y = computeYFaceMinmaxValue(c101, c100, c001, c000);

            c111.x = computeXFaceMinmaxValue(c111, c110, c101, c100);
            c111.y = computeYFaceMinmaxValue(c111, c110, c011, c010);
            c111.z = computeZFaceMinmaxValue(c111, c110, c101, c011);

            return c111;
        }

        void main()
        {
            setOutput(vec4(computeFaceMinmaxValues(outputCoords()), 0.0));
        }
        `
    }
}

class IterateFaceMinmaxValuesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        dominantAxis: Axis = 'z',
        directionOctant: Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = shape
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validCellCoords(ivec3 cellCoords)
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

        vec3 faceMinmaxValuesAt(ivec3 cellCoords)
        {
            return validCellCoords(cellCoords) ? 
                getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).xyz : vec3(NEG_INF);
        }

        float computeXFaceMinmaxValue(vec3 c111, vec3 c110, vec3 c101)
        {
            float minValues = min(c110.z, c101.y);
            return max(c111.x, minValues);
        }

        float computeYFaceMinmaxValue(vec3 c111, vec3 c110, vec3 c011)
        {
            float minValues = min(c110.z, c011.x);
            return max(c111.y, minValues);
        }

        float computeZFaceMinmaxValue(vec3 c111, vec3 c110, vec3 c101, vec3 c011)
        {
            float minValues = min(c110.z, min(c101.y, c011.x));
            return max(c111.z, minValues);
        }

        vec3 computeFaceMinmaxValues(ivec3 cellCoords)
        {
            vec3 c111 = faceMinmaxValuesAt(cellCoords + ivec3(${cellOffset( 0,  0,  0, dominantAxis, directionOctant)}));
            vec3 c011 = faceMinmaxValuesAt(cellCoords + ivec3(${cellOffset(-1,  0,  0, dominantAxis, directionOctant)}));
            vec3 c101 = faceMinmaxValuesAt(cellCoords + ivec3(${cellOffset( 0, -1,  0, dominantAxis, directionOctant)}));
            vec3 c110 = faceMinmaxValuesAt(cellCoords + ivec3(${cellOffset( 0,  0, -1, dominantAxis, directionOctant)}));

            float xFaceMinmaxValue = computeXFaceMinmaxValue(c111, c110, c101);
            float yFaceMinmaxValue = computeYFaceMinmaxValue(c111, c110, c011);
            float zFaceMinmaxValue = computeZFaceMinmaxValue(c111, c110, c101, c011);

            return vec3(
                xFaceMinmaxValue, 
                yFaceMinmaxValue, 
                zFaceMinmaxValue
            );
        }

        void main()
        {
            setOutput(vec4(computeFaceMinmaxValues(outputCoords()), 0.0));
        }
        `
    }
}

class ComputeFaceShadowsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        shape: [number, number, number, 2, 2],
        dominantAxis: Axis = 'z',
        directionOctant: Octant = '+++'
    ) {
        const [depth, height, width, ] = shape

        this.outputShape = shape
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validCellCoords(ivec3 cellCoords)
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

        vec3 faceMaxValuesAt(ivec3 cellCoords)
        {
            return getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).xyz;
        }

        vec3 faceMinmaxValuesAt(ivec3 cellCoords)
        {
            return validCellCoords(cellCoords) ? 
                getB(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).xyz : vec3(NEG_INF);
        }

        bvec3 computeFaceShadows(ivec3 cellCoords)
        {
            vec3 c111 =    faceMaxValuesAt(cellCoords + ivec3(${cellOffset( 0,  0,  0, dominantAxis, directionOctant)}));
            vec3 c011 = faceMinmaxValuesAt(cellCoords + ivec3(${cellOffset(-1,  0,  0, dominantAxis, directionOctant)}));
            vec3 c101 = faceMinmaxValuesAt(cellCoords + ivec3(${cellOffset( 0, -1,  0, dominantAxis, directionOctant)}));
            vec3 c110 = faceMinmaxValuesAt(cellCoords + ivec3(${cellOffset( 0,  0, -1, dominantAxis, directionOctant)}));

            vec3 faceMargins = c111 - vec3(c011.x, c101.y, c110.z);

            return lessThanEqual(faceMargins, vec3(tolerance));
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
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: [number, number, number],
        dominantAxis: Axis = 'z',
        directionOctant: Octant = '+++'
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth, height, width, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validCellCoords(ivec3 cellCoords)
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
            bool c000 = cellShadow(cellCoords + ivec3(${cellOffset(0, 0, 0, dominantAxis, directionOctant)}));
            return bvec3(c000);
        }

        void main()
        {
            setOutput(vec4(computeFaceHoles(outputCoords()), 1.0));
        }
        `
    }
}

class ComputeCellShadowsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = true
    packedOutput = false

    constructor(
        shape: [number, number, number, 2, 2],
        dominantAxis: Axis = 'z',
        directionOctant: Octant = '+++'
    ) {
        const [depth, height, width, ] = shape

        this.outputShape = [depth, height, width]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validCellCoords(ivec3 cellCoords)
        {
            return 
                cellCoords.x >= minCoords.x && cellCoords.x <= maxCoords.x &&
                cellCoords.y >= minCoords.y && cellCoords.y <= maxCoords.y &&
                cellCoords.z >= minCoords.z && cellCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec3 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        bvec3 faceShadowsAt(ivec3 cellCoords)
        {
            return greaterThan(getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).xyz, vec3(0.5));
        }

        bool computeCellShadow(ivec3 cellCoords)
        {
            bvec3 faceShadows = faceShadowsAt(cellCoords);
            return all(faceShadows);
        }

        void main()
        {   
            setOutput(float(computeCellShadow(outputCoords())));
        }
        `
    }
}

function hollowFaceMinValues(
    faceMinValues: tf.Tensor5D,
    faceHoles: tf.Tensor5D,
    verbose: boolean = false
): tf.Tensor5D  
{
    const shape = faceMinValues.shape as [number, number, number, 2, 2]
    const program = new HollowFaceMinValuesProgram(shape)
    const hollowFaceMinima = runWebGLProgram(program, [faceMinValues, faceHoles], 'float32', [], true) 
    if (verbose) logMean3d('hollowFaceMinima', hollowFaceMinima)

    return hollowFaceMinima as tf.Tensor5D
}

function computeFaceMinValues(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = volume.shape
    const program = new ComputeFaceMinValuesProgram(shape, dominantAxis, directionOctant)
    const faceMinValues = runWebGLProgram(program, [volume], 'float32', [], true) 
    if (verbose) logMean3d('faceMinValues', faceMinValues)

    return faceMinValues as tf.Tensor5D
}

function propagateFaceMinmaxValues(
    faceMinValues: tf.Tensor5D,
    dominantAxis: Axis,
    directionOctant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(dominantAxis, directionOctant)
    const dimension = permute[0]
    const backwards = reverse.includes(dimension)

    const slices = unstack3dPacked(faceMinValues, dimension)
    const shape = slices[0].shape as [number, number, number, 2, 2]
    const propagate = new PropagateFaceMinmaxValuesProgram(shape, dominantAxis, directionOctant)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    const faceMinmaxValues = stack3dPacked(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean3d('faceMinmaxValues', faceMinmaxValues)

    return faceMinmaxValues as tf.Tensor5D
}

function iterateFaceMinmaxValues(
    faceMinValues: tf.Tensor5D,
    dominantAxis: Axis,
    directionOctant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{    
    const shape = faceMinValues.shape as [number, number, number, 2, 2]
    const dimension = su.axisToDimension(dominantAxis)
    const iterations = shape[dimension]

    const iterate = new IterateFaceMinmaxValuesProgram(shape, dominantAxis, directionOctant)
    let prev = faceMinValues.clone() 

    for (let i = 0; i < iterations; i += 1)
    {
        const next = runWebGLProgram(iterate, [prev], 'float32', [], true)
        tf.dispose(prev)
        prev = next as tf.Tensor5D
    }

    const faceMinmaxValues = prev
    if (verbose) logMean3d('faceMinmaxValues', faceMinmaxValues)

    return faceMinmaxValues as tf.Tensor5D
}

function computeFaceMaxValues(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = volume.shape
    const program = new ComputeFaceMaxValuesProgram(shape, dominantAxis, directionOctant)
    const faceMaxValues = runWebGLProgram(program, [volume], 'float32', [], true) 
    if (verbose) logMean3d('faceMaxValues', faceMaxValues)

    return faceMaxValues as tf.Tensor5D
}

function computeFaceShadows(
    faceMaxValues: tf.Tensor5D,
    faceMinmaxValues: tf.Tensor5D,
    dominantAxis: Axis,
    directionOctant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = faceMaxValues.shape as [number, number, number, 2, 2]
    const program = new ComputeFaceShadowsProgram(shape, dominantAxis, directionOctant)
    const faceShadows = runWebGLProgram(program, [faceMaxValues, faceMinmaxValues], 'bool', [[tolerance]], true) 
    if (verbose) logMean3d('faceShadows', faceShadows)

    return faceShadows as tf.Tensor5D
}

function computeCellShadows(
    faceShadows: tf.Tensor5D,
    dominantAxis: Axis,
    directionOctant: Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = faceShadows.shape as [number, number, number, 2, 2]
    const program = new ComputeCellShadowsProgram(shape, dominantAxis, directionOctant)
    const cellShadows = runWebGLProgram(program, [faceShadows], 'bool', [], true) 
    if (verbose) logMean3d('cellShadows', cellShadows)

    return cellShadows as tf.Tensor3D
}

function computeFaceHoles(
    cellShadows: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = cellShadows.shape
    const program = new ComputeFaceHolesProgram(shape, dominantAxis, directionOctant)
    const faceHoles = runWebGLProgram(program, [cellShadows], 'bool', [], true) 
    if (verbose) logMean3d('faceHoles', faceHoles)

    return faceHoles as tf.Tensor5D
}

export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    errorTolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const faceMinValues = computeFaceMinValues(volume, dominantAxis, directionOctant, verbose)
    const faceMinmaxValues = propagateFaceMinmaxValues(faceMinValues, dominantAxis, directionOctant, verbose)
    tf.dispose(faceMinValues)
    
    const faceMaxValues = computeFaceMaxValues(volume, dominantAxis, directionOctant, verbose)
    const faceShadows = computeFaceShadows(faceMaxValues, faceMinmaxValues, dominantAxis, directionOctant, errorTolerance, verbose)
    tf.dispose([faceMinmaxValues, faceMaxValues])

    const cellShadows = computeCellShadows(faceShadows, dominantAxis, directionOctant, verbose)
    tf.dispose(faceShadows)

    if (blockSize === 1) 
        return cellShadows

    const blockShadows = minPool3d(cellShadows, blockSize, blockSize, 0, 'ceil') 
    tf.dispose(cellShadows)
    if (verbose) logMean3d('blockShadows', blockShadows)
    
    return blockShadows
}

export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    errorTolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    // Forward
    const forwardOctant = directionOctant
    const forwardFaceMinValues = computeFaceMinValues(volume, dominantAxis, forwardOctant, verbose)
    const forwardFaceMinmaxValues = propagateFaceMinmaxValues(forwardFaceMinValues, dominantAxis, forwardOctant, verbose)
    tf.dispose(forwardFaceMinValues)
    
    const forwardFaceMaxValues = computeFaceMaxValues(volume, dominantAxis, forwardOctant, verbose)
    const forwardFaceShadows = computeFaceShadows(forwardFaceMaxValues, forwardFaceMinmaxValues, dominantAxis, forwardOctant, errorTolerance, verbose)
    tf.dispose([forwardFaceMinmaxValues, forwardFaceMaxValues])

    const forwardCellShadows = computeCellShadows(forwardFaceShadows, dominantAxis, forwardOctant, verbose)
    tf.dispose(forwardFaceShadows)

    // Backward
    const backwardOctant = su.reverseOctant(directionOctant)
    const backwardFaceHoles = computeFaceHoles(forwardCellShadows, dominantAxis, backwardOctant, verbose)

    const backwardFaceMinValues = computeFaceMinValues(volume, dominantAxis, backwardOctant, verbose)
    const backwardHollowFaceMinValues = hollowFaceMinValues(backwardFaceMinValues, backwardFaceHoles, verbose)
    tf.dispose([backwardFaceMinValues, backwardFaceHoles])

    const backwardFaceMinmaxValues = propagateFaceMinmaxValues(backwardHollowFaceMinValues, dominantAxis, backwardOctant, verbose)
    tf.dispose(backwardHollowFaceMinValues)
    
    const backwardFaceMaxValues = computeFaceMaxValues(volume, dominantAxis, backwardOctant, verbose)
    const backwardFaceShadows = computeFaceShadows(backwardFaceMaxValues, backwardFaceMinmaxValues, dominantAxis, backwardOctant, errorTolerance, verbose)
    tf.dispose([backwardFaceMinmaxValues, backwardFaceMaxValues])

    const backwardCellShadows = computeCellShadows(backwardFaceShadows, dominantAxis, backwardOctant, verbose)
    tf.dispose(backwardFaceShadows)

    // Bidirectional 
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows) as tf.Tensor3D
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean3d('bidirectionalCellShadows', bidirectionalCellShadows)

    if (blockSize === 1) 
        return bidirectionalCellShadows

    const bidirectionalBlockShadows = minPool3d(bidirectionalCellShadows, blockSize, blockSize, 0, 'ceil') 
    tf.dispose(bidirectionalCellShadows)
    if (verbose) logMean3d('bidirectionalBlockShadows', bidirectionalBlockShadows)
    
    return bidirectionalBlockShadows
}
