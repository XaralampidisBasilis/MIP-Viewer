import { createStorageBuffer, readBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { runComputeProgram } from '../../WebGPU/WebGPUComputeUtils'

const REDUCE_WORKGROUP_SIZE = 256

export async function mean3dWebGPU( input, options = {} ) 
{
	const { awaitCompletion = false } = options

	if ( input.dtype !== 'float32' ) {
		throw new Error( `mean3dWebGPU only supports float32 tensors, got "${input.dtype}".` )
	}

	if ( input.size <= 0 ) {
		throw new Error( 'mean3dWebGPU expected a non-empty tensor.' )
	}

	let inputBuffer = input.buffer
	let inputCount = input.size
	const tempBuffers = []

	while ( inputCount > 1 ) 
	{
		const outputCount = Math.ceil( inputCount / REDUCE_WORKGROUP_SIZE )
		const dispatch = reduceDispatch( input.device, outputCount )

		const outputBuffer = createStorageBuffer(
			input.device,
			outputCount * Float32Array.BYTES_PER_ELEMENT,
			'reduce-sum-output',
		)

		await runComputeProgram( input.device, {
			label: 'reduce-sum',
			code: sum3dWGSL( inputCount, outputCount, dispatch[ 0 ] ),
			bindings: [
				{ buffer: inputBuffer },
				{ buffer: outputBuffer },
			],
			dispatch,
			awaitCompletion,
		} )

		inputBuffer = outputBuffer
		inputCount = outputCount
		tempBuffers.push( outputBuffer )
	}

	const result = await readBuffer(
		input.device,
		inputBuffer,
		Float32Array.BYTES_PER_ELEMENT,
		Float32Array,
	)

	for ( const buffer of tempBuffers ) {
		buffer.destroy()
	}

	return result[ 0 ] / input.size
}

function reduceDispatch( device, outputCount ) 
{
	const maxWorkgroupsPerDimension = device.limits?.maxComputeWorkgroupsPerDimension ?? 65535

	const x = Math.min( outputCount, maxWorkgroupsPerDimension )
	const y = Math.ceil( outputCount / x )

	if ( y > maxWorkgroupsPerDimension ) {
		throw new Error(
			`mean3dWebGPU needs dispatch [${x}, ${y}, 1], but this WebGPU device allows only ${maxWorkgroupsPerDimension} workgroups per dimension.`,
		)
	}

	return [ x, y, 1 ]
}

function sum3dWGSL( inputCount, outputCount, dispatchX ) 
{
	return /* wgsl */ `

    const INPUT_COUNT: u32 = ${inputCount}u;
    const OUTPUT_COUNT: u32 = ${outputCount}u;
    const WORKGROUP_SIZE: u32 = ${REDUCE_WORKGROUP_SIZE}u;
    const WORKGROUPS_X: u32 = ${dispatchX}u;

    @group(0) @binding(0) var<storage, read> input_values: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output_values: array<f32>;

    var<workgroup> sums: array<f32, ${REDUCE_WORKGROUP_SIZE}>;

    @compute @workgroup_size(${REDUCE_WORKGROUP_SIZE})
    fn main(
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
    )
    {
        let local_index = lid.x;

        let output_index = wid.y * WORKGROUPS_X + wid.x;

        if (output_index >= OUTPUT_COUNT) { return; }

        let input_index = output_index * WORKGROUP_SIZE + local_index;

        var value = 0.0;

        if (input_index < INPUT_COUNT) {
            value = input_values[input_index];
        }

        sums[local_index] = value;

        workgroupBarrier();

        var stride = WORKGROUP_SIZE / 2u;

        loop
        {
            if (stride == 0u) {
                break;
            }

            if (local_index < stride) {
                sums[local_index] = sums[local_index] + sums[local_index + stride];
            }

            workgroupBarrier();

            stride = stride / 2u;
        }

        if (local_index == 0u) {
            output_values[output_index] = sums[0];
        }
    }
    `
}
