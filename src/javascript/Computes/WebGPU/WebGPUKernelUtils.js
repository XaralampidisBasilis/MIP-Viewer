import { createUniformBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { dispatchForShape, runComputeProgram } from '../../WebGPU/WebGPUComputeUtils'
import { WebGPUTensor3D } from '../../WebGPU/WebGPUTensor3D'

export const WORKGROUP_SIZE_3D = [ 8, 8, 4 ]
export const WORKGROUP_SIZE_2D = [ 16, 16, 1 ]

export function workgroupSizeWGSL( workgroupSize ) {
	return `@compute @workgroup_size(${workgroupSize[ 0 ]}, ${workgroupSize[ 1 ]}, ${workgroupSize[ 2 ]})`
}

export function commonTensor3DWGSL( shape ) {
	const [ width, height, depth ] = shape
	return /* wgsl */ `
	const DEPTH: u32 = ${depth}u;
	const HEIGHT: u32 = ${height}u;
	const WIDTH: u32 = ${width}u;

	fn index3(coords: vec3<i32>) -> u32
	{
	    return u32(coords.z) * HEIGHT * WIDTH + u32(coords.y) * WIDTH + u32(coords.x);
	}

	fn index3u(coords: vec3<u32>) -> u32
	{
	    return coords.z * HEIGHT * WIDTH + coords.y * WIDTH + coords.x;
	}

	fn valid3(coords: vec3<i32>) -> bool
	{
	    return coords.x >= 0 && coords.x < i32(WIDTH) &&
	           coords.y >= 0 && coords.y < i32(HEIGHT) &&
	           coords.z >= 0 && coords.z < i32(DEPTH);
	}

	fn clamp3(coords: vec3<i32>) -> vec3<i32>
	{
	    return clamp(coords, vec3<i32>(0), vec3<i32>(i32(WIDTH) - 1, i32(HEIGHT) - 1, i32(DEPTH) - 1));
	}
`
}

export function commonOutputWGSL( shape, prefix = 'OUT' ) {
	const [ width, height, depth ] = shape
	const lowerPrefix = prefix.toLowerCase()
	return /* wgsl */ `
	const ${prefix}_DEPTH: u32 = ${depth}u;
	const ${prefix}_HEIGHT: u32 = ${height}u;
	const ${prefix}_WIDTH: u32 = ${width}u;

	fn ${lowerPrefix}_index(coords: vec3<i32>) -> u32
	{
	    return u32(coords.z) * ${prefix}_HEIGHT * ${prefix}_WIDTH + u32(coords.y) * ${prefix}_WIDTH + u32(coords.x);
	}

	fn ${lowerPrefix}_index_u(coords: vec3<u32>) -> u32
	{
	    return coords.z * ${prefix}_HEIGHT * ${prefix}_WIDTH + coords.y * ${prefix}_WIDTH + coords.x;
	}

	fn ${lowerPrefix}_in_bounds(gid: vec3<u32>) -> bool
	{
	    return gid.x < ${prefix}_WIDTH && gid.y < ${prefix}_HEIGHT && gid.z < ${prefix}_DEPTH;
	}
`
}

export function axisSize( shape, axis ) {
	const [ width, height, depth ] = shape
	if ( axis === 'x' ) return width
	if ( axis === 'y' ) return height
	return depth
}

export function axisStride( shape, axis ) {
	const [ width, height ] = shape
	if ( axis === 'x' ) return 1
	if ( axis === 'y' ) return width
	return height * width
}

export function sampleIndexWGSL( baseIndex, radius, sign ) {
	const offset = `u32(${radius}) * AXIS_STRIDE`
	return sign === '-' ? `${baseIndex} - ${offset}` : `${baseIndex} + ${offset}`
}

export async function logTensor3DMeanWebGPU( name, tensor ) {
	const data = await tensor.read()
	let sum = 0
	for ( let i = 0; i < data.length; i += 1 ) sum += Number( data[ i ] )
	console.log( name, sum / Math.max( data.length, 1 ) )
}

export async function logicalOrInPlaceWebGPU( a, b ) {
	await runComputeProgram( a.device, {
		label: 'logical-or-in-place',
		code: logicalOrInPlaceWGSL( a.shape ),
		bindings: [ { buffer: a.buffer }, { buffer: b.buffer } ],
		dispatch: dispatchForShape( a.shape, WORKGROUP_SIZE_3D ),
	} )
	return a
}

export async function minPoolBool3dWebGPU( input, blockSize ) {
	const outputShape = input.shape.map( ( size ) => Math.ceil( size / blockSize ) )
	const output = WebGPUTensor3D.empty( input.device, outputShape, 'uint32', 'webgpu-min-pool-bool3d' )

	await runComputeProgram( input.device, {
		label: 'min-pool-bool3d',
		code: minPoolBool3dWGSL( input.shape, outputShape, blockSize ),
		bindings: [ { buffer: input.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

export async function map3dWebGPU( input, minValue, maxValue ) {
	const output = WebGPUTensor3D.empty( input.device, input.shape, 'float32', 'map3d-output' )
	const paramsBuffer = map3dParamsBuffer( input, minValue, maxValue, 'map3d-params' )

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
	const paramsBuffer = map3dParamsBuffer( input, minValue, maxValue, 'map3d-in-place-params' )

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
	const [ inWidth, inHeight, inDepth ] = input.shape
	const [ outWidth, outHeight, outDepth ] = outputShape

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
	const paramsBuffer = createUniformBuffer(
		input.device,
		new Float32Array( [ scaleFactors[ 0 ], scaleFactors[ 1 ], scaleFactors[ 2 ], offsetFactor ] ),
		'resize-trilinear-params',
	)

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

function map3dParamsBuffer( input, minValue, maxValue, label ) {
	const invRange = 1 / ( maxValue - minValue )
	return createUniformBuffer(
		input.device,
		new Float32Array( [ minValue, invRange, 0, 0 ] ),
		label,
	)
}

function logicalOrInPlaceWGSL( shape ) {
	return /* wgsl */ `
	${commonTensor3DWGSL( shape )}

	@group(0) @binding(0) var<storage, read_write> a_values: array<u32>;
	@group(0) @binding(1) var<storage, read> b_values: array<u32>;

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

	    let idx = index3u(gid);
	    a_values[idx] = a_values[idx] | b_values[idx];
	}
`
}

function minPoolBool3dWGSL( inputShape, outputShape, blockSize ) {
	return /* wgsl */ `
	${commonTensor3DWGSL( inputShape )}
	${commonOutputWGSL( outputShape )}

	const BLOCK_SIZE: i32 = ${blockSize};

	@group(0) @binding(0) var<storage, read> input_values: array<u32>;
	@group(0) @binding(1) var<storage, read_write> output_values: array<u32>;

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!out_in_bounds(gid)) { return; }

	    let out_coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
	    let start = out_coords * vec3<i32>(BLOCK_SIZE);
	    let end = min(start + vec3<i32>(BLOCK_SIZE), vec3<i32>(i32(WIDTH), i32(HEIGHT), i32(DEPTH)));

	    var value = 1u;

	    for (var dz = start.z; dz < end.z && value != 0u; dz = dz + 1) {
	        for (var dy = start.y; dy < end.y && value != 0u; dy = dy + 1) {
	            for (var dx = start.x; dx < end.x; dx = dx + 1) {
	                value = value & input_values[index3(vec3<i32>(dx, dy, dz))];
	                if (value == 0u) { break; }
	            }
	        }
	    }

	    output_values[out_index_u(gid)] = value;
	}
`
}

function map3dWGSL( shape ) {
	return /* wgsl */ `
	${commonTensor3DWGSL( shape )}

	struct Params {
	    min_value: f32,
	    inv_range: f32,
	    _pad0: f32,
	    _pad1: f32,
	};

	@group(0) @binding(0) var<storage, read> input_values: array<f32>;
	@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
	@group(0) @binding(2) var<uniform> params: Params;

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

	    let idx = index3u(gid);
	    let mapped = (input_values[idx] - params.min_value) * params.inv_range;
	    output_values[idx] = clamp(mapped, 0.0, 1.0);
	}
`
}

function map3dInPlaceWGSL( shape ) {
	return /* wgsl */ `
	${commonTensor3DWGSL( shape )}

	struct Params {
	    min_value: f32,
	    inv_range: f32,
	    _pad0: f32,
	    _pad1: f32,
	};

	@group(0) @binding(0) var<storage, read_write> values: array<f32>;
	@group(0) @binding(1) var<uniform> params: Params;

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

	    let idx = index3u(gid);
	    let mapped = (values[idx] - params.min_value) * params.inv_range;
	    values[idx] = clamp(mapped, 0.0, 1.0);
	}
`
}

function resizeTrilinearWGSL( inputShape, outputShape ) {
	const [ inWidth, inHeight, inDepth ] = inputShape
	const [ outWidth, outHeight, outDepth ] = outputShape

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

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    let x = gid.x;
	    let y = gid.y;
	    let z = gid.z;

	    if (x >= OUT_WIDTH || y >= OUT_HEIGHT || z >= OUT_DEPTH) { return; }

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
