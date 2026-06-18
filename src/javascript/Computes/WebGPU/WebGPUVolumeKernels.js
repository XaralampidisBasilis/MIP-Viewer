import { createStorageBuffer, createUniformBuffer, readBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { dispatchForShape, runComputeProgram } from './WebGPUComputeRunner'
import { WebGPUTensor3D } from './WebGPUTensor3D'

const REDUCE_WORKGROUP_SIZE = 256
const WORKGROUP_SIZE_3D = [ 8, 8, 4 ]

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

export async function map3dWebGPU( input, minValue, maxValue ) {

	const output = WebGPUTensor3D.empty( input.device, input.shape, 'float32', 'map3d-output' )
	const invRange = 1 / ( maxValue - minValue )

	const params = new Float32Array( [
		minValue,
		invRange,
		0,
		0,
	] )

	const paramsBuffer = createUniformBuffer( input.device, params, 'map3d-params' )

	await runComputeProgram( input.device, {
		label: 'map3d',
		code: map3dWGSL( input.shape ),
		bindings: [
			{ buffer: input.buffer },
			{ buffer: output.buffer },
			{ buffer: paramsBuffer },
		],
		dispatch: dispatchForShape( input.shape, WORKGROUP_SIZE_3D ),
	} )

	paramsBuffer.destroy()

	return output
}

export async function map3dInPlaceWebGPU( input, minValue, maxValue ) {

	const invRange = 1 / ( maxValue - minValue )

	const params = new Float32Array( [
		minValue,
		invRange,
		0,
		0,
	] )

	const paramsBuffer = createUniformBuffer( input.device, params, 'map3d-in-place-params' )

	await runComputeProgram( input.device, {
		label: 'map3d-in-place',
		code: map3dInPlaceWGSL( input.shape ),
		bindings: [
			{ buffer: input.buffer },
			{ buffer: paramsBuffer },
		],
		dispatch: dispatchForShape( input.shape, WORKGROUP_SIZE_3D ),
	} )

	paramsBuffer.destroy()

	return input
}
export async function resizeTrilinearWebGPU( input, outputShape, alignCorners = false, halfPixelCenters = true ) {

	const output = WebGPUTensor3D.empty( input.device, outputShape, 'float32', 'resize-trilinear-output' )
	const [ inDepth, inHeight, inWidth ] = input.shape
	const [ outDepth, outHeight, outWidth ] = outputShape

	const effectiveInSize = [
		alignCorners && outDepth > 1 ? inDepth - 1 : inDepth,
		alignCorners && outHeight > 1 ? inHeight - 1 : inHeight,
		alignCorners && outWidth > 1 ? inWidth - 1 : inWidth,
	]

	const effectiveOutSize = [
		alignCorners && outDepth > 1 ? outDepth - 1 : outDepth,
		alignCorners && outHeight > 1 ? outHeight - 1 : outHeight,
		alignCorners && outWidth > 1 ? outWidth - 1 : outWidth,
	]

	const scaleFactors = effectiveInSize.map( ( size, i ) => size / effectiveOutSize[ i ] )
	const offsetFactor = halfPixelCenters ? 0.5 : 0.0

	const params = new Float32Array( [
		scaleFactors[ 0 ],
		scaleFactors[ 1 ],
		scaleFactors[ 2 ],
		offsetFactor,
	] )

	const paramsBuffer = createUniformBuffer( input.device, params, 'resize-trilinear-params' )

	await runComputeProgram( input.device, {
		label: 'resize-trilinear',
		code: resizeTrilinearWGSL( input.shape, outputShape ),
		bindings: [
			{ buffer: input.buffer },
			{ buffer: output.buffer },
			{ buffer: paramsBuffer },
		],
		dispatch: dispatchForShape( outputShape, WORKGROUP_SIZE_3D ),
	} )

	paramsBuffer.destroy()

	return output
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
function map3dWGSL( shape ) {

	const [ depth, height, width ] = shape

	return /* wgsl */ `
	const DEPTH: u32 = ${depth}u;
	const HEIGHT: u32 = ${height}u;
	const WIDTH: u32 = ${width}u;
	const PLANE_SIZE: u32 = HEIGHT * WIDTH;

	struct Params {
	    min_value: f32,
	    inv_range: f32,
	    _pad0: f32,
	    _pad1: f32,
	};

	@group(0) @binding(0) var<storage, read> input_values: array<f32>;
	@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
	@group(0) @binding(2) var<uniform> params: Params;

	fn index3(gid: vec3<u32>) -> u32
	{
	    return gid.z * PLANE_SIZE + gid.y * WIDTH + gid.x;
	}

	@compute @workgroup_size(8, 8, 4)
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) {
	        return;
	    }

	    let idx = index3(gid);
	    let mapped = (input_values[idx] - params.min_value) * params.inv_range;

	    output_values[idx] = clamp(mapped, 0.0, 1.0);
	}
`
}

function map3dInPlaceWGSL( shape ) {

	const [ depth, height, width ] = shape

	return /* wgsl */ `
	const DEPTH: u32 = ${depth}u;
	const HEIGHT: u32 = ${height}u;
	const WIDTH: u32 = ${width}u;
	const PLANE_SIZE: u32 = HEIGHT * WIDTH;

	struct Params {
	    min_value: f32,
	    inv_range: f32,
	    _pad0: f32,
	    _pad1: f32,
	};

	@group(0) @binding(0) var<storage, read_write> values: array<f32>;
	@group(0) @binding(1) var<uniform> params: Params;

	fn index3(gid: vec3<u32>) -> u32
	{
	    return gid.z * PLANE_SIZE + gid.y * WIDTH + gid.x;
	}

	@compute @workgroup_size(8, 8, 4)
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) {
	        return;
	    }

	    let idx = index3(gid);
	    let mapped = (values[idx] - params.min_value) * params.inv_range;

	    values[idx] = clamp(mapped, 0.0, 1.0);
	}
    `
}

function resizeTrilinearWGSL( inputShape, outputShape ) {

	const [ inDepth, inHeight, inWidth ] = inputShape
	const [ outDepth, outHeight, outWidth ] = outputShape

	return /* wgsl */ `
	const IN_DEPTH: u32 = ${inDepth}u;
	const IN_HEIGHT: u32 = ${inHeight}u;
	const IN_WIDTH: u32 = ${inWidth}u;
	const IN_PLANE_SIZE: u32 = IN_HEIGHT * IN_WIDTH;

	const OUT_DEPTH: u32 = ${outDepth}u;
	const OUT_HEIGHT: u32 = ${outHeight}u;
	const OUT_WIDTH: u32 = ${outWidth}u;
	const OUT_PLANE_SIZE: u32 = OUT_HEIGHT * OUT_WIDTH;

	struct Params {
	    scale_depth: f32,
	    scale_height: f32,
	    scale_width: f32,
	    offset_factor: f32,
	};

	@group(0) @binding(0) var<storage, read> input_values: array<f32>;
	@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
	@group(0) @binding(2) var<uniform> params: Params;

	fn input_index(z: u32, y: u32, x: u32) -> u32
	{
	    return z * IN_PLANE_SIZE + y * IN_WIDTH + x;
	}

	fn output_index(z: u32, y: u32, x: u32) -> u32
	{
	    return z * OUT_PLANE_SIZE + y * OUT_WIDTH + x;
	}

	fn clamp_index(v: i32, hi: u32) -> u32
	{
	    return u32(clamp(v, 0, i32(hi) - 1));
	}

	@compute @workgroup_size(8, 8, 4)
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    let x = gid.x;
	    let y = gid.y;
	    let z = gid.z;

	    if (x >= OUT_WIDTH || y >= OUT_HEIGHT || z >= OUT_DEPTH) {
	        return;
	    }

	    let source_z = (f32(z) + params.offset_factor) * params.scale_depth - params.offset_factor;
	    let source_y = (f32(y) + params.offset_factor) * params.scale_height - params.offset_factor;
	    let source_x = (f32(x) + params.offset_factor) * params.scale_width - params.offset_factor;

	    let floor_z_i = i32(floor(source_z));
	    let floor_y_i = i32(floor(source_y));
	    let floor_x_i = i32(floor(source_x));

	    let floor_z = clamp_index(floor_z_i, IN_DEPTH);
	    let floor_y = clamp_index(floor_y_i, IN_HEIGHT);
	    let floor_x = clamp_index(floor_x_i, IN_WIDTH);

	    let ceil_z = clamp_index(floor_z_i + 1, IN_DEPTH);
	    let ceil_y = clamp_index(floor_y_i + 1, IN_HEIGHT);
	    let ceil_x = clamp_index(floor_x_i + 1, IN_WIDTH);

	    let fz = source_z - floor(source_z);
	    let fy = source_y - floor(source_y);
	    let fx = source_x - floor(source_x);

        let c000 = input_values[input_index(floor_z, floor_y, floor_x)];
	    let c001 = input_values[input_index(floor_z, floor_y, ceil_x)];
	    let c010 = input_values[input_index(floor_z, ceil_y, floor_x)];
	    let c011 = input_values[input_index(floor_z, ceil_y, ceil_x)];
	    let c100 = input_values[input_index(ceil_z, floor_y, floor_x)];
	    let c101 = input_values[input_index(ceil_z, floor_y, ceil_x)];
	    let c110 = input_values[input_index(ceil_z, ceil_y, floor_x)];
	    let c111 = input_values[input_index(ceil_z, ceil_y, ceil_x)];

	    let c00 = mix(c000, c001, fx);
	    let c01 = mix(c010, c011, fx);
	    let c10 = mix(c100, c101, fx);
	    let c11 = mix(c110, c111, fx);

	    let c0 = mix(c00, c01, fy);
	    let c1 = mix(c10, c11, fy);

	    output_values[output_index(z, y, x)] = mix(c0, c1, fz);
	}
`
}