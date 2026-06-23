import { createStorageBuffer, readBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { runComputeProgram } from '../../WebGPU/WebGPUComputeUtils'

// Number of values each compute workgroup reduces at once.
// One workgroup reads up to 256 float values and outputs one vec2:
// [minimum, maximum].
const REDUCE_WORKGROUP_SIZE = 256

export async function minMax3dWebGPU(input, options = {})
{
    const { awaitCompletion = false } = options

    if (input.dtype !== 'float32')
    {
        throw new Error(`minMax3dWebGPU only supports float32 tensors, got "${input.dtype}".`)
    }

    if (input.size <= 0)
    {
        throw new Error('minMax3dWebGPU expected a non-empty tensor.')
    }

    // Special case:
    // If the tensor contains only one scalar, the min and max are the same value.
    // We must not read this as vec2<f32>, because the original tensor contains
    // only one float.
    if (input.size === 1)
    {
        const result = await readBuffer(
            input.device,
            input.buffer,
            Float32Array.BYTES_PER_ELEMENT,
            Float32Array,
        )

        return [result[0], result[0]]
    }

    // Start from the original tensor buffer.
    // We never destroy this buffer because it is owned by the caller.
    let inputBuffer = input.buffer

    // Total number of scalar values/pairs to reduce.
    // First pass: number of scalar float32 values.
    // Later passes: number of min/max pairs.
    let inputCount = input.size

    const tempBuffers = []

    // The first pass reads raw float values.
    // Later passes read vec2<f32> pairs: [minimum, maximum].
    let firstPass = true

    // Keep reducing until only one min/max pair remains.
    while (inputCount > 1)
    {
        // Each output pair is the min/max of up to REDUCE_WORKGROUP_SIZE inputs.
        const outputCount = Math.ceil(inputCount / REDUCE_WORKGROUP_SIZE)

        // Calculate how many compute workgroups are needed.
        const dispatch = reduceDispatch(input.device, outputCount)

        // Allocate a temporary GPU buffer to store this pass's reduced pairs.
        //
        // Each output item is a vec2<f32>, so it uses 2 float32 values.
        const outputBuffer = createStorageBuffer(
            input.device,
            outputCount * 2 * Float32Array.BYTES_PER_ELEMENT,
            'minmax3d-output',
        )

        await runComputeProgram(input.device, {
            label: firstPass ? 'minmax3d-values' : 'minmax3d-pairs',
            code: firstPass ?
                minMax3dValuesWGSL(inputCount, outputCount, dispatch[0]) :
                minMax3dPairsWGSL(inputCount, outputCount, dispatch[0]),
            bindings: [
                { buffer: inputBuffer },
                { buffer: outputBuffer },
            ],
            dispatch,
            awaitCompletion,
        })

        // The output of this pass becomes the input of the next pass.
        inputBuffer = outputBuffer
        inputCount = outputCount
        tempBuffers.push(outputBuffer)
        firstPass = false
    }

    // Read the final vec2<f32> pair back to the CPU.
    const result = await readBuffer(
        input.device,
        inputBuffer,
        2 * Float32Array.BYTES_PER_ELEMENT,
        Float32Array,
    )

    for (const buffer of tempBuffers)
    {
        buffer.destroy()
    }

    // Return [minimum, maximum].
    return [result[0], result[1]]
}

function reduceDispatch(device, outputCount)
{
    // WebGPU limits how many workgroups can be dispatched per dimension.
    const maxWorkgroupsPerDimension =
        device.limits?.maxComputeWorkgroupsPerDimension ?? 65535

    // Use as many workgroups as possible in X, up to the device limit.
    const x = Math.min(outputCount, maxWorkgroupsPerDimension)

    // If outputCount is larger than X can hold, spill into Y.
    const y = Math.ceil(outputCount / x)

    // Fail clearly if even a 2D dispatch cannot fit the required workgroups.
    if (y > maxWorkgroupsPerDimension)
    {
        throw new Error(
            `minMax3dWebGPU needs dispatch [${x}, ${y}, 1], but this WebGPU device allows only ${maxWorkgroupsPerDimension} workgroups per dimension.`,
        )
    }

    return [x, y, 1]
}

function minMax3dValuesWGSL(inputCount, outputCount, dispatchX)
{
    return /* wgsl */ `
    const INPUT_COUNT: u32 = ${inputCount}u;
    const OUTPUT_COUNT: u32 = ${outputCount}u;
    const WORKGROUP_SIZE: u32 = ${REDUCE_WORKGROUP_SIZE}u;
    const WORKGROUPS_X: u32 = ${dispatchX}u;

    // Largest and smallest finite f32 values.
    // Used as neutral values for min/max reduction.
    const POS_INF: f32 = 3.402823466e+38;
    const NEG_INF: f32 = -3.402823466e+38;

    // Raw input values to reduce.
    @group(0) @binding(0) var<storage, read> input_values: array<f32>;

    // One output [min, max] pair per workgroup.
    @group(0) @binding(1) var<storage, read_write> output_pairs: array<vec2<f32>>;

    // Shared memory for values inside one workgroup.
    var<workgroup> mins: array<f32, ${REDUCE_WORKGROUP_SIZE}>;
    var<workgroup> maxs: array<f32, ${REDUCE_WORKGROUP_SIZE}>;

    @compute @workgroup_size(${REDUCE_WORKGROUP_SIZE})
    fn main(
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
    )
    {
        let local_index = lid.x;

        // Flatten the 2D workgroup id into one output index.
        let output_index = wid.y * WORKGROUPS_X + wid.x;

        // Some dispatches may include extra workgroups because of rounding.
        if (output_index >= OUTPUT_COUNT)
        {
            return;
        }

        // Each invocation handles one input value for this output group.
        let input_index = output_index * WORKGROUP_SIZE + local_index;

        // Out-of-bounds lanes use neutral values so they do not affect the result.
        var min_value = POS_INF;
        var max_value = NEG_INF;

        if (input_index < INPUT_COUNT)
        {
            let raw = input_values[input_index];

            // Ignore NaN values.
            // For min, NaN becomes +infinity.
            // For max, NaN becomes -infinity.
            min_value = select(raw, POS_INF, raw != raw);
            max_value = select(raw, NEG_INF, raw != raw);
        }

        // Store this invocation's candidates into workgroup memory.
        mins[local_index] = min_value;
        maxs[local_index] = max_value;

        // Make sure all invocations have written before reduction begins.
        workgroupBarrier();

        // Parallel reduction:
        // 256 values -> 128 -> 64 -> 32 -> ... -> 1.
        var stride = WORKGROUP_SIZE / 2u;

        loop
        {
            if (stride == 0u)
            {
                break;
            }

            if (local_index < stride)
            {
                mins[local_index] = min(mins[local_index], mins[local_index + stride]);
                maxs[local_index] = max(maxs[local_index], maxs[local_index + stride]);
            }

            // Wait after each reduction step before the next stride reads results.
            workgroupBarrier();

            stride = stride / 2u;
        }

        // The first invocation writes the workgroup's final min/max pair.
        if (local_index == 0u)
        {
            output_pairs[output_index] = vec2<f32>(mins[0], maxs[0]);
        }
    }
`
}

function minMax3dPairsWGSL(inputCount, outputCount, dispatchX)
{
    return /* wgsl */ `
    const INPUT_COUNT: u32 = ${inputCount}u;
    const OUTPUT_COUNT: u32 = ${outputCount}u;
    const WORKGROUP_SIZE: u32 = ${REDUCE_WORKGROUP_SIZE}u;
    const WORKGROUPS_X: u32 = ${dispatchX}u;

    // Largest and smallest finite f32 values.
    // Used as neutral values for min/max reduction.
    const POS_INF: f32 = 3.402823466e+38;
    const NEG_INF: f32 = -3.402823466e+38;

    // Input min/max pairs from the previous reduction pass.
    @group(0) @binding(0) var<storage, read> input_pairs: array<vec2<f32>>;

    // One output [min, max] pair per workgroup.
    @group(0) @binding(1) var<storage, read_write> output_pairs: array<vec2<f32>>;

    // Shared memory for values inside one workgroup.
    var<workgroup> mins: array<f32, ${REDUCE_WORKGROUP_SIZE}>;
    var<workgroup> maxs: array<f32, ${REDUCE_WORKGROUP_SIZE}>;

    @compute @workgroup_size(${REDUCE_WORKGROUP_SIZE})
    fn main(
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
    )
    {
        let local_index = lid.x;

        // Flatten the 2D workgroup id into one output index.
        let output_index = wid.y * WORKGROUPS_X + wid.x;

        // Some dispatches may include extra workgroups because of rounding.
        if (output_index >= OUTPUT_COUNT)
        {
            return;
        }

        // Each invocation handles one input pair for this output group.
        let input_index = output_index * WORKGROUP_SIZE + local_index;

        // Out-of-bounds lanes use neutral values so they do not affect the result.
        var pair = vec2<f32>(POS_INF, NEG_INF);

        if (input_index < INPUT_COUNT)
        {
            pair = input_pairs[input_index];
        }

        // Store this invocation's candidates into workgroup memory.
        mins[local_index] = pair.x;
        maxs[local_index] = pair.y;

        // Make sure all invocations have written before reduction begins.
        workgroupBarrier();

        // Parallel reduction:
        // 256 values -> 128 -> 64 -> 32 -> ... -> 1.
        var stride = WORKGROUP_SIZE / 2u;

        loop
        {
            if (stride == 0u)
            {
                break;
            }

            if (local_index < stride)
            {
                mins[local_index] = min(mins[local_index], mins[local_index + stride]);
                maxs[local_index] = max(maxs[local_index], maxs[local_index + stride]);
            }

            // Wait after each reduction step before the next stride reads results.
            workgroupBarrier();

            stride = stride / 2u;
        }

        // The first invocation writes the workgroup's final min/max pair.
        if (local_index == 0u)
        {
            output_pairs[output_index] = vec2<f32>(mins[0], maxs[0]);
        }
    }
`
}
