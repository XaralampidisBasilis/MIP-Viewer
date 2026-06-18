import { createStorageBuffer, readBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { runComputeProgram } from './WebGPUComputeRunner'
export {
	map3dInPlaceWebGPU,
	map3dWebGPU,
	resizeTrilinearWebGPU,
} from './WebGPUKernelUtils'

const REDUCE_WORKGROUP_SIZE = 256

export async function reduceMinMaxWebGPU( input ) {

	let inputBuffer = input.buffer
	let inputCount = input.size
	let ownsInputBuffer = false
	let firstPass = true

	while ( inputCount > 1 ) {

		const outputCount = Math.ceil( inputCount / REDUCE_WORKGROUP_SIZE )
		const dispatch = reduceDispatch( input.device, outputCount )
		const outputBuffer = createStorageBuffer(
			input.device,
			outputCount * 2 * Float32Array.BYTES_PER_ELEMENT,
			'reduce-minmax-output',
		)

		await runComputeProgram( input.device, {
			label: firstPass ? 'reduce-minmax-values' : 'reduce-minmax-pairs',
			code: firstPass ?
				reduceMinMaxValuesWGSL( inputCount, outputCount, dispatch[ 0 ] ) :
				reduceMinMaxPairsWGSL( inputCount, outputCount, dispatch[ 0 ] ),
			bindings: [
				{ buffer: inputBuffer },
				{ buffer: outputBuffer },
			],
			dispatch,
		} )

		if ( ownsInputBuffer ) {
			inputBuffer.destroy()
		}

		inputBuffer = outputBuffer
		inputCount = outputCount
		ownsInputBuffer = true
		firstPass = false
	}

	const result = await readBuffer(
		input.device,
		inputBuffer,
		2 * Float32Array.BYTES_PER_ELEMENT,
		Float32Array,
	)

	if ( ownsInputBuffer ) {
		inputBuffer.destroy()
	}

	return [ result[ 0 ], result[ 1 ] ]
}

function reduceDispatch( device, outputCount ) {

	const maxWorkgroupsPerDimension = device.limits?.maxComputeWorkgroupsPerDimension ?? 65535
	const x = Math.min( outputCount, maxWorkgroupsPerDimension )
	const y = Math.ceil( outputCount / x )

	if ( y > maxWorkgroupsPerDimension ) {
		throw new Error(
			`reduceMinMaxWebGPU needs dispatch [${x}, ${y}, 1], but this WebGPU device allows only ${maxWorkgroupsPerDimension} workgroups per dimension.`,
		)
	}

	return [ x, y, 1 ]
}

function reduceMinMaxValuesWGSL( inputCount, outputCount, dispatchX ) {

	return /* wgsl */ `
	const INPUT_COUNT: u32 = ${inputCount}u;
	const OUTPUT_COUNT: u32 = ${outputCount}u;
	const WORKGROUP_SIZE: u32 = ${REDUCE_WORKGROUP_SIZE}u;
	const WORKGROUPS_X: u32 = ${dispatchX}u;
	const POS_INF: f32 = 3.402823466e+38;
	const NEG_INF: f32 = -3.402823466e+38;

	@group(0) @binding(0) var<storage, read> input_values: array<f32>;
	@group(0) @binding(1) var<storage, read_write> output_pairs: array<vec2<f32>>;

	var<workgroup> mins: array<f32, ${REDUCE_WORKGROUP_SIZE}>;
	var<workgroup> maxs: array<f32, ${REDUCE_WORKGROUP_SIZE}>;

	@compute @workgroup_size(${REDUCE_WORKGROUP_SIZE})
	fn main(
	    @builtin(local_invocation_id) lid: vec3<u32>,
	    @builtin(workgroup_id) wid: vec3<u32>,
	)
	{
	    let local_index = lid.x;
	    let output_index = wid.y * WORKGROUPS_X + wid.x;

	    if (output_index >= OUTPUT_COUNT) {
	        return;
	    }

	    let input_index = output_index * WORKGROUP_SIZE + local_index;
	    let in_bounds = input_index < INPUT_COUNT;

	    var value = POS_INF;

	    if (in_bounds) {
	        value = input_values[input_index];
	    }

	    mins[local_index] = value;
	    maxs[local_index] = select(NEG_INF, value, in_bounds);

	    workgroupBarrier();

	    var stride = WORKGROUP_SIZE / 2u;

	    loop {
	        if (stride == 0u) {
	            break;
	        }

	        if (local_index < stride) {
	            mins[local_index] = min(mins[local_index], mins[local_index + stride]);
	            maxs[local_index] = max(maxs[local_index], maxs[local_index + stride]);
	        }

	        workgroupBarrier();

	        stride = stride / 2u;
	    }

	    if (local_index == 0u) {
	        output_pairs[output_index] = vec2<f32>(mins[0], maxs[0]);
	    }
	}
`
}

function reduceMinMaxPairsWGSL( pairCount, outputCount, dispatchX ) {

	return /* wgsl */ `
	const INPUT_COUNT: u32 = ${pairCount}u;
	const OUTPUT_COUNT: u32 = ${outputCount}u;
	const WORKGROUP_SIZE: u32 = ${REDUCE_WORKGROUP_SIZE}u;
	const WORKGROUPS_X: u32 = ${dispatchX}u;
	const POS_INF: f32 = 3.402823466e+38;
	const NEG_INF: f32 = -3.402823466e+38;

	@group(0) @binding(0) var<storage, read> input_pairs: array<vec2<f32>>;
	@group(0) @binding(1) var<storage, read_write> output_pairs: array<vec2<f32>>;

	var<workgroup> mins: array<f32, ${REDUCE_WORKGROUP_SIZE}>;
	var<workgroup> maxs: array<f32, ${REDUCE_WORKGROUP_SIZE}>;

	@compute @workgroup_size(${REDUCE_WORKGROUP_SIZE})
	fn main(
	    @builtin(local_invocation_id) lid: vec3<u32>,
	    @builtin(workgroup_id) wid: vec3<u32>,
	)
	{
	    let local_index = lid.x;
	    let output_index = wid.y * WORKGROUPS_X + wid.x;

	    if (output_index >= OUTPUT_COUNT) {
	        return;
	    }

	    let input_index = output_index * WORKGROUP_SIZE + local_index;

	    var pair = vec2<f32>(POS_INF, NEG_INF);

	    if (input_index < INPUT_COUNT) {
	        pair = input_pairs[input_index];
	    }

	    mins[local_index] = pair.x;
	    maxs[local_index] = pair.y;

	    workgroupBarrier();

	    var stride = WORKGROUP_SIZE / 2u;

	    loop {
	        if (stride == 0u) {
	            break;
	        }

	        if (local_index < stride) {
	            mins[local_index] = min(mins[local_index], mins[local_index + stride]);
	            maxs[local_index] = max(maxs[local_index], maxs[local_index + stride]);
	        }

	        workgroupBarrier();

	        stride = stride / 2u;
	    }

	    if (local_index == 0u) {
	        output_pairs[output_index] = vec2<f32>(mins[0], maxs[0]);
	    }
	}
`
}
