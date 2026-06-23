import { createStorageBuffer, readBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { runComputeProgram } from '../../WebGPU/WebGPUComputeUtils'

// Number of values each compute workgroup reduces at once.
// One workgroup reads up to 256 float values and outputs one reduced value.
const REDUCE_WORKGROUP_SIZE = 256

export async function max3dWebGPU(input, options = {})
{
    const { awaitCompletion = false } = options

    if (input.dtype !== 'float32')
    {
        throw new Error(`max3dWebGPU only supports float32 tensors, got "${input.dtype}".`)
    }

    if (input.size <= 0)
    {
        throw new Error('max3dWebGPU expected a non-empty tensor.')
    }

    // Start from the original tensor buffer.
    // We never destroy this buffer because it is owned by the caller.
    let inputBuffer = input.buffer

    // Total number of scalar float32 values to reduce.
    let inputCount = input.size

    const tempBuffers = []

    // Keep reducing until only one value remains.
    while (inputCount > 1)
    {
        // Each output value is the maximum of up to REDUCE_WORKGROUP_SIZE values.
        const outputCount = Math.ceil(inputCount / REDUCE_WORKGROUP_SIZE)

        // Calculate how many compute workgroups are needed.
        const dispatch = reduceDispatch(input.device, outputCount)

        // Allocate a temporary GPU buffer for this pass.
        const outputBuffer = createStorageBuffer(
            input.device,
            outputCount * Float32Array.BYTES_PER_ELEMENT,
            'reduce-max-output',
        )

        await runComputeProgram(input.device, {
            label: 'reduce-max',
            code: max3dWGSL(inputCount, outputCount, dispatch[0]),
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
    }

    // Read the final single float value back to the CPU.
    const result = await readBuffer(
        input.device,
        inputBuffer,
        Float32Array.BYTES_PER_ELEMENT,
        Float32Array,
    )

    for (const buffer of tempBuffers)
    {
        buffer.destroy()
    }

    return result[0]
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
            `max3dWebGPU needs dispatch [${x}, ${y}, 1], but this WebGPU device allows only ${maxWorkgroupsPerDimension} workgroups per dimension.`,
        )
    }

    return [x, y, 1]
}

function max3dWGSL(inputCount, outputCount, dispatchX)
{
    return /* wgsl */ `
    const INPUT_COUNT: u32 = ${inputCount}u;
    const OUTPUT_COUNT: u32 = ${outputCount}u;
    const WORKGROUP_SIZE: u32 = ${REDUCE_WORKGROUP_SIZE}u;
    const WORKGROUPS_X: u32 = ${dispatchX}u;

    // Smallest finite f32 value.
    // Used as the neutral value for max reduction.
    const NEG_INF: f32 = -3.402823466e+38;

    // Raw input values to reduce.
    @group(0) @binding(0) var<storage, read> input_values: array<f32>;

    // One output maximum value per workgroup.
    @group(0) @binding(1) var<storage, read_write> output_values: array<f32>;

    // Shared memory for values inside one workgroup.
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

        // Out-of-bounds lanes use NEG_INF so they do not affect the maximum.
        var value = NEG_INF;

        if (input_index < INPUT_COUNT)
        {
            let raw = input_values[input_index];

            // Ignore NaN values by treating them as -infinity.
            // In WGSL, NaN is the only value where raw != raw is true.
            value = select(raw, NEG_INF, raw != raw);
        }

        // Store this invocation's value into workgroup memory.
        maxs[local_index] = value;

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
                maxs[local_index] = max(maxs[local_index], maxs[local_index + stride]);
            }

            // Wait after each reduction step before the next stride reads results.
            workgroupBarrier();

            stride = stride / 2u;
        }

        // The first invocation writes the workgroup's final maximum.
        if (local_index == 0u)
        {
            output_values[output_index] = maxs[0];
        }
    }
`
}
