import * as su from '../../Utils/ShadowMapUtils'
import { createUniformBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { dispatchForShape, runComputeProgram, runComputeProgramSequence } from './WebGPUComputeRunner'
import { WebGPUTensor3D } from './WebGPUTensor3D'

const WORKGROUP_SIZE_3D = [ 8, 8, 4 ]
const WORKGROUP_SIZE_2D = [ 16, 16, 1 ]

function workgroupSizeWGSL( workgroupSize ) {
    
	return `@compute @workgroup_size(${workgroupSize[ 0 ]}, ${workgroupSize[ 1 ]}, ${workgroupSize[ 2 ]})`
}

export async function computeBidirectionalShadowMapWebGPU( volume, dominantAxis, directionOctant, tolerance, blockSize ) {

	const forwardOctant = directionOctant

	let forwardVertexMinmax = null
	let forwardVertexShadows = null
	let forwardCellShadows = null
	let backwardVertexMinmax = null
	let backwardVertexShadows = null
	let backwardCellShadows = null
	let keepForwardCellShadows = false

	try {

		forwardVertexMinmax = await computeVertexValues( volume )
		await propagateVertexMinmaxValuesInPlace( forwardVertexMinmax, dominantAxis, forwardOctant )

		forwardVertexShadows = await computeVertexShadowsFromVolume(
			volume,
			forwardVertexMinmax,
			dominantAxis,
			forwardOctant,
			tolerance,
		)

		forwardVertexMinmax.dispose()
		forwardVertexMinmax = null

		forwardCellShadows = await computeCellShadows( forwardVertexShadows, dominantAxis, forwardOctant )

		forwardVertexShadows.dispose()
		forwardVertexShadows = null

		const backwardOctant = su.reverseOctant( forwardOctant )

		backwardVertexMinmax = await computeVertexValuesWithHoles(
			volume,
			forwardCellShadows,
			dominantAxis,
			backwardOctant,
		)

		await propagateVertexMinmaxValuesInPlace( backwardVertexMinmax, dominantAxis, backwardOctant )

		backwardVertexShadows = await computeVertexShadowsFromVolumeHoles(
			volume,
			forwardCellShadows,
			backwardVertexMinmax,
			dominantAxis,
			backwardOctant,
			tolerance,
		)

		backwardVertexMinmax.dispose()
		backwardVertexMinmax = null

		backwardCellShadows = await computeCellShadows( backwardVertexShadows, dominantAxis, backwardOctant )

		backwardVertexShadows.dispose()
		backwardVertexShadows = null

		await logicalOrInPlace( forwardCellShadows, backwardCellShadows )

		backwardCellShadows.dispose()
		backwardCellShadows = null

		if ( blockSize === 1 ) 
        {
			keepForwardCellShadows = true
			return forwardCellShadows
		}

		const blockShadows = await minPoolBool3d( forwardCellShadows, blockSize )

		forwardCellShadows.dispose()
		forwardCellShadows = null

		return blockShadows

	} finally {

		forwardVertexMinmax?.dispose()
		forwardVertexShadows?.dispose()

		if ( ! keepForwardCellShadows ) {
			forwardCellShadows?.dispose()
		}

		backwardVertexMinmax?.dispose()
		backwardVertexShadows?.dispose()
		backwardCellShadows?.dispose()
	}
}

export async function computeUnidirectionalDistanceMapWebGPU( shadowMap, dominantAxis, directionOctant, maxDistance ) {

    const sweepAxes = [ 'x', 'y', 'z' ].filter( ( axis ) => axis !== dominantAxis )
	const sweepSigns = sweepAxes.map( ( axis ) => su.getOctantSign( directionOctant, axis ) )
	const dominantSign = su.getOctantSign( directionOctant, dominantAxis )
    
	const distances0d = await initialChebyshevDistanceInPlace( shadowMap, maxDistance )

	const distances1d = await anisotropicChebyshevDistance( distances0d, sweepAxes[ 0 ], sweepSigns[ 0 ], maxDistance )
	distances0d.dispose()

	const distances2d = await anisotropicChebyshevDistance( distances1d, sweepAxes[ 1 ], sweepSigns[ 1 ], maxDistance )
	distances1d.dispose()

	const distances3d = await extendedChebyshevDistance( distances2d, dominantAxis, dominantSign, maxDistance )
	distances2d.dispose()

	return distances3d
}

async function computeVertexValues( volume ) {

	const [ depth, height, width ] = volume.shape
	const output = WebGPUTensor3D.empty(
		volume.device,
		[ depth + 2, height + 2, width + 2 ],
		'float32',
		'webgpu-vertex-values',
	)

	await runComputeProgram( volume.device, {
		label: 'compute-vertex-values',
		code: computeVertexValuesWGSL( volume.shape, output.shape ),
		bindings: [ { buffer: volume.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function computeVertexValuesWithHoles( volume, cellShadows, axis, octant ) {

	const [ depth, height, width ] = volume.shape
	const output = WebGPUTensor3D.empty(
		volume.device,
		[ depth + 2, height + 2, width + 2 ],
		'float32',
		'webgpu-vertex-values-holes',
	)

	await runComputeProgram( volume.device, {
		label: `compute-vertex-values-holes-${axis}-${octant}`,
		code: computeVertexValuesWithHolesWGSL( volume.shape, cellShadows.shape, output.shape, axis, octant ),
		bindings: [ { buffer: volume.buffer }, { buffer: cellShadows.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function propagateVertexMinmaxValuesInPlace( vertexValues, axis, octant ) {

	const dimension = su.axisToDimension( axis )
	const sign = su.getOctantSign( octant, axis )
	const backwards = sign === '-'
	const slices = vertexValues.shape[ dimension ]

	const start = backwards ? slices - 2 : 1
	const end = backwards ? -1 : slices
	const step = backwards ? -1 : 1
	const steps = []
	const paramsBuffers = []
	const dispatch = dispatchForPlane( vertexValues.shape, axis )

	for ( let slice = start; slice !== end; slice += step ) {

		const paramsBuffer = createUniformBuffer(
			vertexValues.device,
			new Int32Array( [ slice, 0, 0, 0 ] ),
			`propagate-vertex-minmax-${axis}-${octant}-${slice}:params`,
		)

		paramsBuffers.push( paramsBuffer )
		steps.push( {
			bindings: [ { buffer: vertexValues.buffer }, { buffer: paramsBuffer } ],
			dispatch,
		} )
	}

	await runComputeProgramSequence( vertexValues.device, {
		label: `propagate-vertex-minmax-${axis}-${octant}`,
		code: propagateVertexMinmaxInPlaceWGSL( vertexValues.shape, axis, octant, step ),
		steps,
		disposeAfterSubmit: paramsBuffers,
	} )

	return vertexValues
}

async function computeVertexShadowsFromVolume( volume, vertexMinmax, axis, octant, tolerance ) {

	const output = WebGPUTensor3D.empty( volume.device, vertexMinmax.shape, 'uint32', 'webgpu-vertex-shadows' )

	await runComputeProgram( volume.device, {
		label: `compute-vertex-shadows-${axis}-${octant}`,
		code: computeVertexShadowsFromVolumeWGSL( volume.shape, vertexMinmax.shape, axis, octant, tolerance ),
		bindings: [ { buffer: volume.buffer }, { buffer: vertexMinmax.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function computeVertexShadowsFromVolumeHoles( volume, cellShadows, vertexMinmax, axis, octant, tolerance ) {

	const output = WebGPUTensor3D.empty( volume.device, vertexMinmax.shape, 'uint32', 'webgpu-vertex-shadows-holes' )
    	await runComputeProgram( volume.device, {
		label: `compute-vertex-shadows-holes-${axis}-${octant}`,
		code: computeVertexShadowsFromVolumeHolesWGSL(
			volume.shape,
			cellShadows.shape,
			vertexMinmax.shape,
			axis,
			octant,
			tolerance,
		),
		bindings: [
			{ buffer: volume.buffer },
			{ buffer: cellShadows.buffer },
			{ buffer: vertexMinmax.buffer },
			{ buffer: output.buffer },
		],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function computeCellShadows( vertexShadows, axis, octant ) {

	const [ depth, height, width ] = vertexShadows.shape
	const output = WebGPUTensor3D.empty(
		vertexShadows.device,
		[ depth - 1, height - 1, width - 1 ],
		'uint32',
		'webgpu-cell-shadows',
	)

	await runComputeProgram( vertexShadows.device, {
		label: `compute-cell-shadows-${axis}-${octant}`,
		code: computeCellShadowsWGSL( vertexShadows.shape, output.shape, axis, octant ),
		bindings: [ { buffer: vertexShadows.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function logicalOrInPlace( a, b ) {

	await runComputeProgram( a.device, {
		label: 'logical-or-in-place',
		code: logicalOrInPlaceWGSL( a.shape ),
		bindings: [ { buffer: a.buffer }, { buffer: b.buffer } ],
		dispatch: dispatchForShape( a.shape, WORKGROUP_SIZE_3D ),
	} )

	return a
}

async function minPoolBool3d( input, blockSize ) {

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

async function initialChebyshevDistanceInPlace( shadowMap, maxDistance ) {

	await runComputeProgram( shadowMap.device, {
		label: 'initial-chebyshev-distance-in-place',
		code: initialChebyshevDistanceInPlaceWGSL( shadowMap.shape, maxDistance ),
		bindings: [ { buffer: shadowMap.buffer } ],
		dispatch: dispatchForShape( shadowMap.shape, WORKGROUP_SIZE_3D ),
	} )

	return shadowMap
}

async function anisotropicChebyshevDistance( distances, axis, sign, maxDistance ) {

	const output = WebGPUTensor3D.empty(
		distances.device,
		distances.shape,
		'uint32',
		`webgpu-anisotropic-distance-${axis}-${sign}`,
	)

	await runComputeProgram( distances.device, {
		label: `anisotropic-chebyshev-distance-${axis}-${sign}`,
		code: anisotropicChebyshevDistanceWGSL( distances.shape, axis, sign, maxDistance ),
		bindings: [ { buffer: distances.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function extendedChebyshevDistance( distances, axis, sign, maxDistance ) {

	const output = WebGPUTensor3D.empty(
		distances.device,
		distances.shape,
		'uint32',
		`webgpu-extended-distance-${axis}-${sign}`,
	)

	await runComputeProgram( distances.device, {
		label: `extended-chebyshev-distance-${axis}-${sign}`,
		code: extendedChebyshevDistanceWGSL( distances.shape, axis, sign, maxDistance ),
		bindings: [ { buffer: distances.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

function commonFloatTensorWGSL( shape ) {

	const [ depth, height, width ] = shape

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

function commonOutputWGSL( shape, prefix = 'OUT' ) {

	const [ depth, height, width ] = shape
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

function computeVertexValuesWGSL( inputShape, outputShape ) {

	const [ depth, height, width ] = inputShape

	return /* wgsl */ `
	${commonFloatTensorWGSL( inputShape )}
	${commonOutputWGSL( outputShape )}

	@group(0) @binding(0) var<storage, read> volume: array<f32>;
	@group(0) @binding(1) var<storage, read_write> vertex_values: array<f32>;

	fn volume_at(coords: vec3<i32>) -> f32
	{
	    let clamped = clamp(coords, vec3<i32>(0), vec3<i32>(${width - 1}, ${height - 1}, ${depth - 1}));
	    return volume[index3(clamped)];
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!out_in_bounds(gid)) { return; }

	    let vertex_coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
	    let voxel_coords = vertex_coords - vec3<i32>(1);

	    vertex_values[out_index_u(gid)] = volume_at(voxel_coords);
	}
`
}

function computeVertexValuesWithHolesWGSL( inputShape, cellShape, outputShape, axis, octant ) {

	const [ depth, height, width ] = inputShape
	const [ cellDepth, cellHeight, cellWidth ] = cellShape

	return /* wgsl */ `
	${commonOutputWGSL( outputShape )}

	const VOLUME_DEPTH: u32 = ${depth}u;
	const VOLUME_HEIGHT: u32 = ${height}u;
	const VOLUME_WIDTH: u32 = ${width}u;
    	const CELL_DEPTH: u32 = ${cellDepth}u;
	const CELL_HEIGHT: u32 = ${cellHeight}u;
	const CELL_WIDTH: u32 = ${cellWidth}u;

	@group(0) @binding(0) var<storage, read> volume: array<f32>;
	@group(0) @binding(1) var<storage, read> cell_shadows: array<u32>;
	@group(0) @binding(2) var<storage, read_write> vertex_values: array<f32>;

	fn volume_index(coords: vec3<i32>) -> u32
	{
	    return u32(coords.z) * VOLUME_HEIGHT * VOLUME_WIDTH + u32(coords.y) * VOLUME_WIDTH + u32(coords.x);
	}

	fn cell_index(coords: vec3<i32>) -> u32
	{
	    return u32(coords.z) * CELL_HEIGHT * CELL_WIDTH + u32(coords.y) * CELL_WIDTH + u32(coords.x);
	}

	fn volume_at(coords: vec3<i32>) -> f32
	{
	    let clamped = clamp(coords, vec3<i32>(0), vec3<i32>(${width - 1}, ${height - 1}, ${depth - 1}));
	    return volume[volume_index(clamped)];
	}

	fn cell_shadow_at(coords: vec3<i32>) -> bool
	{
	    let clamped = clamp(coords, vec3<i32>(0), vec3<i32>(${cellWidth - 1}, ${cellHeight - 1}, ${cellDepth - 1}));
	    return cell_shadows[cell_index(clamped)] != 0u;
	}

	fn vertex_value_at(vertex_coords: vec3<i32>) -> f32
	{
	    let hole = cell_shadow_at(vertex_coords - ${vec3(cellOffset(0, 0, 0, axis, octant))});
	    return select(volume_at(vertex_coords - vec3<i32>(1)), 0.0, hole);
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!out_in_bounds(gid)) { return; }

	    let vertex_coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
	    vertex_values[out_index_u(gid)] = vertex_value_at(vertex_coords);
	}
`
}

function propagateVertexMinmaxInPlaceWGSL( shape, axis, octant, step ) {

	const prevAxis = axisDelta( axis, -step )
	const [ planeU, planeV ] = planeSize( shape, axis )

	return /* wgsl */ `
	${commonFloatTensorWGSL( shape )}

	struct Params {
	    slice: i32,
	    _pad0: i32,
	    _pad1: i32,
	    _pad2: i32,
	};

	const NEG_INF: f32 = -3.402823466e+38;
	const PLANE_U: u32 = ${planeU}u;
	const PLANE_V: u32 = ${planeV}u;

	@group(0) @binding(0) var<storage, read_write> propagated_values: array<f32>;
	@group(0) @binding(1) var<uniform> params: Params;

	fn plane_coords(gid: vec3<u32>) -> vec3<i32>
	{
	    ${planeCoordsWGSL(axis)}
	}

	fn raw_at(coords: vec3<i32>) -> f32
	{
	    return propagated_values[index3(coords)];
	}

	fn propagated_at(coords: vec3<i32>) -> f32
	{
	    if (!valid3(coords)) {
	        return NEG_INF;
	    }

	    return propagated_values[index3(coords)];
	}

	fn min4(a: f32, b: f32, c: f32, d: f32) -> f32
	{
	    return min(min(min(a, b), c), d);
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_2D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= PLANE_U || gid.y >= PLANE_V) { return; }

	    let coords = plane_coords(gid);
	    let previous = ${vec3(prevAxis)};

	    let v111 = raw_at(coords + ${vec3(sliceOffset(0, 0, 0, axis, octant))});
	    let v110 = propagated_at(coords + previous + ${vec3(sliceOffset(0, 0, -1, axis, octant))});
	    let v100 = propagated_at(coords + previous + ${vec3(sliceOffset(0, -1, -1, axis, octant))});
	    let v010 = propagated_at(coords + previous + ${vec3(sliceOffset(-1, 0, -1, axis, octant))});
	    let v000 = propagated_at(coords + previous + ${vec3(sliceOffset(-1, -1, -1, axis, octant))});

	    propagated_values[index3(coords)] = max(v111, min4(v000, v010, v100, v110));
	}
`
}

function computeVertexShadowsFromVolumeWGSL( inputShape, vertexShape, axis, octant, tolerance ) {

	const [ depth, height, width ] = inputShape

	return /* wgsl */ `
	${commonFloatTensorWGSL( vertexShape )}

	const VOLUME_DEPTH: u32 = ${depth}u;
	const VOLUME_HEIGHT: u32 = ${height}u;
	const VOLUME_WIDTH: u32 = ${width}u;
	const NEG_INF: f32 = -3.402823466e+38;
	const TOLERANCE: f32 = ${tolerance};

	@group(0) @binding(0) var<storage, read> volume: array<f32>;
	@group(0) @binding(1) var<storage, read> vertex_minmax: array<f32>;
	@group(0) @binding(2) var<storage, read_write> vertex_shadows: array<u32>;

	fn volume_index(coords: vec3<i32>) -> u32
	{
	    return u32(coords.z) * VOLUME_HEIGHT * VOLUME_WIDTH + u32(coords.y) * VOLUME_WIDTH + u32(coords.x);
	}

	fn volume_at(coords: vec3<i32>) -> f32
	{
	    let clamped = clamp(coords, vec3<i32>(0), vec3<i32>(${width - 1}, ${height - 1}, ${depth - 1}));
	    return volume[volume_index(clamped)];
	}

	fn vertex_value_at(coords: vec3<i32>) -> f32
	{
	    return volume_at(coords - vec3<i32>(1));
	}

	fn vertex_minmax_at(coords: vec3<i32>) -> f32
	{
	    if (!valid3(coords)) {
	        return NEG_INF;
	    }

	    return vertex_minmax[index3(coords)];
	}

	fn min4(a: f32, b: f32, c: f32, d: f32) -> f32
	{
	    return min(min(min(a, b), c), d);
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

	    let coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
	    let out_index_value = index3u(gid);

	    let v111 = vertex_value_at(coords + ${vec3(vertexOffset(0, 0, 0, axis, octant))});
	    let v110 = vertex_minmax_at(coords + ${vec3(vertexOffset(0, 0, -1, axis, octant))});
	    let v100 = vertex_minmax_at(coords + ${vec3(vertexOffset(0, -1, -1, axis, octant))});
	    let v010 = vertex_minmax_at(coords + ${vec3(vertexOffset(-1, 0, -1, axis, octant))});
	    let v000 = vertex_minmax_at(coords + ${vec3(vertexOffset(-1, -1, -1, axis, octant))});
	    let margin = v111 - min4(v000, v010, v100, v110);

	    vertex_shadows[out_index_value] = select(0u, 1u, margin <= TOLERANCE);
	}
`
}

function computeVertexShadowsFromVolumeHolesWGSL( inputShape, cellShape, vertexShape, axis, octant, tolerance ) {

	const [ depth, height, width ] = inputShape
	const [ cellDepth, cellHeight, cellWidth ] = cellShape

	return /* wgsl */ `
	${commonFloatTensorWGSL( vertexShape )}

	const VOLUME_DEPTH:  u32 = ${depth}u;
	const VOLUME_HEIGHT: u32 = ${height}u;
	const VOLUME_WIDTH:  u32 = ${width}u;

	const CELL_DEPTH:  u32 = ${cellDepth}u;
	const CELL_HEIGHT: u32 = ${cellHeight}u;
	const CELL_WIDTH:  u32 = ${cellWidth}u;
    
	const NEG_INF: f32 = -3.402823466e+38;
	const TOLERANCE: f32 = ${tolerance};

	@group(0) @binding(0) var<storage, read> volume: array<f32>;
	@group(0) @binding(1) var<storage, read> cell_shadows: array<u32>;
	@group(0) @binding(2) var<storage, read> vertex_minmax: array<f32>;
	@group(0) @binding(3) var<storage, read_write> vertex_shadows: array<u32>;

	fn volume_index(coords: vec3<i32>) -> u32
	{
	    return u32(coords.z) * VOLUME_HEIGHT * VOLUME_WIDTH + u32(coords.y) * VOLUME_WIDTH + u32(coords.x);
	}

	fn cell_index(coords: vec3<i32>) -> u32
	{
	    return u32(coords.z) * CELL_HEIGHT * CELL_WIDTH + u32(coords.y) * CELL_WIDTH + u32(coords.x);
	}

	fn volume_at(coords: vec3<i32>) -> f32
	{
	    let clamped = clamp(coords, vec3<i32>(0), vec3<i32>(${width - 1}, ${height - 1}, ${depth - 1}));
	    return volume[volume_index(clamped)];
	}

	fn cell_shadow_at(coords: vec3<i32>) -> bool
	{
	    let clamped = clamp(coords, vec3<i32>(0), vec3<i32>(${cellWidth - 1}, ${cellHeight - 1}, ${cellDepth - 1}));
        	    return cell_shadows[cell_index(clamped)] != 0u;
	}

	fn vertex_value_at(coords: vec3<i32>) -> f32
	{
	    let hole = cell_shadow_at(coords - ${vec3(cellOffset(0, 0, 0, axis, octant))});
	    return select(volume_at(coords - vec3<i32>(1)), 0.0, hole);
	}

	fn vertex_minmax_at(coords: vec3<i32>) -> f32
	{
	    if (!valid3(coords)) {
	        return NEG_INF;
	    }

	    return vertex_minmax[index3(coords)];
	}

	fn min4(a: f32, b: f32, c: f32, d: f32) -> f32
	{
	    return min(min(min(a, b), c), d);
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

	    let coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
	    let out_index_value = index3u(gid);

	    let v111 =  vertex_value_at(coords + ${vec3(vertexOffset( 0,  0,  0, axis, octant))});
	    let v110 = vertex_minmax_at(coords + ${vec3(vertexOffset( 0,  0, -1, axis, octant))});
	    let v100 = vertex_minmax_at(coords + ${vec3(vertexOffset( 0, -1, -1, axis, octant))});
	    let v010 = vertex_minmax_at(coords + ${vec3(vertexOffset(-1,  0, -1, axis, octant))});
	    let v000 = vertex_minmax_at(coords + ${vec3(vertexOffset(-1, -1, -1, axis, octant))});

	    let margin = v111 - min4(v000, v010, v100, v110);

	    vertex_shadows[out_index_value] = select(0u, 1u, margin <= TOLERANCE);
	}
`
}

function computeCellShadowsWGSL( inputShape, outputShape, axis, octant ) {

	return /* wgsl */ `
	${commonFloatTensorWGSL( inputShape )}
	${commonOutputWGSL( outputShape )}

	@group(0) @binding(0) var<storage, read> vertex_shadows: array<u32>;
	@group(0) @binding(1) var<storage, read_write> cell_shadows: array<u32>;

	fn vertex_shadow_at(coords: vec3<i32>) -> u32
	{
	    return vertex_shadows[index3(coords)];
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!out_in_bounds(gid)) { return; }

	    let coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
	    let shadow =
	        vertex_shadow_at(coords + ${vec3(cellOffset(1, 1, 1, axis, octant))}) &
	        vertex_shadow_at(coords + ${vec3(cellOffset(1, 0, 1, axis, octant))}) &
	        vertex_shadow_at(coords + ${vec3(cellOffset(0, 1, 1, axis, octant))}) &
	        vertex_shadow_at(coords + ${vec3(cellOffset(0, 0, 1, axis, octant))}) &
	        vertex_shadow_at(coords + ${vec3(cellOffset(1, 1, 0, axis, octant))}) &
	        vertex_shadow_at(coords + ${vec3(cellOffset(1, 0, 0, axis, octant))}) &
	        vertex_shadow_at(coords + ${vec3(cellOffset(0, 1, 0, axis, octant))});

	    cell_shadows[out_index_u(gid)] = shadow;
	}
`
}

function logicalOrInPlaceWGSL( shape ) {

	return /* wgsl */ `
	${commonFloatTensorWGSL( shape )}

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
	${commonFloatTensorWGSL( inputShape )}
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

	                if (value == 0u) {
	                    break;
	                }
	            }
	        }
	    }

	    output_values[out_index_u(gid)] = value;
	}
`
}

function initialChebyshevDistanceInPlaceWGSL( shape, maxDistance ) {

	return /* wgsl */ `
	${commonFloatTensorWGSL( shape )}

	const MAX_DISTANCE: u32 = ${maxDistance}u;

	@group(0) @binding(0) var<storage, read_write> distances: array<u32>;

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

	    let idx = index3u(gid);
	    distances[idx] = select(0u, MAX_DISTANCE, distances[idx] != 0u);
	}
`
}

function anisotropicChebyshevDistanceWGSL( shape, axis, sign, maxDistance ) {

	const axisField = axis
	const signOperator = sign
	const sampleIndexExpression = sampleIndexWGSL( 'out_index_value', 'radius', axis, sign )

	return /* wgsl */ `
	${commonFloatTensorWGSL( shape )}

	const MAX_DISTANCE: u32 = ${maxDistance}u;
	const MAX_RADIUS: i32 = min(i32(MAX_DISTANCE), i32(${axisSize(shape, axis)}u) - 1);
	const AXIS_LIMIT: i32 = i32(${axisSize(shape, axis)}u);
	const AXIS_STRIDE: u32 = ${axisStride(shape, axis)}u;

	@group(0) @binding(0) var<storage, read> input_distances: array<u32>;
	@group(0) @binding(1) var<storage, read_write> output_distances: array<u32>;

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

	    let out_index_value = index3u(gid);
	    let out_axis = i32(gid.${axisField});
	    var min_distance = input_distances[out_index_value];

	    if (min_distance == 0u) {
	        output_distances[out_index_value] = min_distance;
	        return;
	    }

	    for (var radius = 1; radius <= MAX_RADIUS; radius = radius + 1) {
	        let sample_axis = out_axis ${signOperator} radius;

	        if (sample_axis < 0 || sample_axis >= AXIS_LIMIT) {
	            break;
	        }

	        let sample_distance = input_distances[${sampleIndexExpression}];

	        if (sample_distance == 0u) {
	            min_distance = u32(radius);
	            break;
	        }

	        let candidate = max(sample_distance, u32(radius));
	        min_distance = min(min_distance, candidate);

	        if (min_distance <= u32(radius)) {
	            break;
	        }
	    }

	    output_distances[out_index_value] = min_distance;
	}
`
}

function extendedChebyshevDistanceWGSL( shape, axis, sign, maxDistance ) {

	const axisField = axis
	const signOperator = sign
	const sampleIndexExpression = sampleIndexWGSL( 'out_index_value', 'radius', axis, sign )

	return /* wgsl */ `
    	${commonFloatTensorWGSL( shape )}

	const MAX_DISTANCE: u32 = ${maxDistance}u;
	const MAX_RADIUS: i32 = min(i32(MAX_DISTANCE), i32(${axisSize(shape, axis)}u) - 1);
	const AXIS_LIMIT: i32 = i32(${axisSize(shape, axis)}u);
	const AXIS_STRIDE: u32 = ${axisStride(shape, axis)}u;

	@group(0) @binding(0) var<storage, read> input_distances: array<u32>;
	@group(0) @binding(1) var<storage, read_write> output_distances: array<u32>;

	fn candidate_distance(sample_distance: u32, radius: u32) -> u32
	{
	    return select(MAX_DISTANCE, radius, sample_distance <= radius);
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

	    let out_index_value = index3u(gid);
	    let out_axis = i32(gid.${axisField});
	    var min_distance = MAX_DISTANCE;

	    for (var radius = 0; radius <= MAX_RADIUS; radius = radius + 1) {
	        let sample_axis = out_axis ${signOperator} radius;

	        if (sample_axis < 0 || sample_axis >= AXIS_LIMIT) {
	            break;
	        }

	        let sample_distance = input_distances[${sampleIndexExpression}];

	        if (sample_distance <= u32(radius)) {
	            min_distance = u32(radius);
	            break;
	        }

	        min_distance = min(min_distance, candidate_distance(sample_distance, u32(radius)));

	        if (min_distance <= u32(radius)) {
	            break;
	        }
	    }

	    output_distances[out_index_value] = min_distance;
	}
`
}

function dispatchForPlane( shape, axis ) {

	const [ u, v ] = planeSize( shape, axis )

	return [
		Math.ceil(u / WORKGROUP_SIZE_2D[ 0 ]),
		Math.ceil(v / WORKGROUP_SIZE_2D[ 1 ]),
		1,
	]
}

function planeSize( shape, axis ) {

	const [ depth, height, width ] = shape

	if ( axis === 'x' ) return [ height, depth ]
	if ( axis === 'y' ) return [ width, depth ]
	return [ width, height ]
}

function planeCoordsWGSL( axis ) {

	if ( axis === 'x' ) {
		return 'return vec3<i32>(params.slice, i32(gid.x), i32(gid.y));'
	}

	if ( axis === 'y' ) {
		return 'return vec3<i32>(i32(gid.x), params.slice, i32(gid.y));'
	}

	return 'return vec3<i32>(i32(gid.x), i32(gid.y), params.slice);'
}

function axisSize( shape, axis ) {

	const [ depth, height, width ] = shape

	if ( axis === 'x' ) return width
	if ( axis === 'y' ) return height
	return depth
}

function axisStride( shape, axis ) {

	const [ depth, height, width ] = shape

	if ( axis === 'x' ) return 1
	if ( axis === 'y' ) return width
	return height * width
}

function sampleIndexWGSL( baseIndex, radius, axis, sign ) {

	const offset = `u32(${radius}) * AXIS_STRIDE`

	if ( sign === '-' ) {
		return `${baseIndex} - ${offset}`
	}

	return `${baseIndex} + ${offset}`
}

function axisDelta( axis, value ) {

	if ( axis === 'x' ) return [ value, 0, 0 ]
	if ( axis === 'y' ) return [ 0, value, 0 ]
	return [ 0, 0, value ]
}

function vec3( xyz ) {

	return `vec3<i32>(${xyz[0]}, ${xyz[1]}, ${xyz[2]})`
}

function toXYZ( zyx ) {

	return [ zyx[ 2 ], zyx[ 1 ], zyx[ 0 ] ]
}

function vertexOffset( x, y, z, axis, octant ) {

	const { permute, reverse } = su.dominantAxisOctantToPermuteReverse( axis, octant )
	const zyx = su.applyPermutation( [ z, y, x ], permute )

	for ( const dimension of reverse ) {
		zyx[ dimension ] = -zyx[ dimension ]
	}

	return toXYZ( zyx )
}

function sliceOffset( x, y, z, axis, octant ) {

	const { permute, reverse } = su.dominantAxisOctantToPermuteReverse( axis, octant )
	const zyx = su.applyPermutation( [ z, y, x ], permute )

	for ( const dimension of reverse ) {
		zyx[ dimension ] = -zyx[ dimension ]
	}

	zyx[ permute[ 0 ] ] = 0

	return toXYZ( zyx )
}

function cellOffset( x, y, z, axis, octant ) {

	const { permute, reverse } = su.dominantAxisOctantToPermuteReverse( axis, octant )
	const zyx = su.applyPermutation( [ z, y, x ], permute )

	for ( const dimension of reverse ) {
		zyx[ dimension ] = 1 - zyx[ dimension ]
	}

	return toXYZ( zyx )
}