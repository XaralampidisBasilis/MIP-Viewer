import { createStorageBuffer, createUniformBuffer, readBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { dispatchForShape, runComputeProgram, runComputeProgramSequence } from '../../WebGPU/WebGPUComputeUtils'
import { WebGPUTensor3D } from '../../WebGPU/WebGPUTensor3D'
import {
	axisDelta,
	axisToShapeIndex,
	getOctantSign,
	planeCoordsWGSL,
	planeSize,
	reverseOctant,
	signedOffset,
	sliceOffset as shadowSliceOffset,
	unitOffset,
	vec3,
} from './WebGPUShadowMapUtils'
import {
	WORKGROUP_SIZE_2D,
	WORKGROUP_SIZE_3D,
	commonTensor3DWGSL,
	logicalOrInPlaceWebGPU,
	logTensor3DMeanWebGPU,
	minPoolBool3dWebGPU,
	workgroupSizeWGSL,
} from './WebGPUKernelUtils'

const FACE_LANES = 3
const NEG_INF = '-3.402823466e+38'

class WebGPUFaceTensor3D {
	constructor( device, shape, dtype, buffer, label = 'face-tensor3d' ) {
		this.device = device
		this.shape = shape
		this.dtype = dtype
		this.buffer = buffer
		this.label = label
		this.width = shape[ 0 ]
		this.height = shape[ 1 ]
		this.depth = shape[ 2 ]
		this.cellCount = shape[ 0 ] * shape[ 1 ] * shape[ 2 ]
		this.size = this.cellCount * FACE_LANES
		this.byteLength = this.size * 4
	}

	static empty( device, shape, dtype = 'float32', label = 'empty-face-tensor3d' ) {
		const bytes = shape[ 0 ] * shape[ 1 ] * shape[ 2 ] * FACE_LANES * 4
		const buffer = createStorageBuffer( device, bytes, label )
		return new WebGPUFaceTensor3D( device, shape, dtype, buffer, label )
	}

	async read() {
		return readBuffer( this.device, this.buffer, this.byteLength, this.dtype === 'uint32' ? Uint32Array : Float32Array )
	}

	clone( label = `${this.label}:clone` ) {
		const clone = WebGPUFaceTensor3D.empty( this.device, this.shape, this.dtype, label )
		const encoder = this.device.createCommandEncoder()
		encoder.copyBufferToBuffer( this.buffer, 0, clone.buffer, 0, this.byteLength )
		this.device.queue.submit( [ encoder.finish() ] )
		return clone
	}

	dispose() {
		this.buffer?.destroy()
		this.buffer = null
	}
}

export async function computeBidirectionalShadowMapFacesWebGPU(
	volume,
	dominantAxis,
	directionOctant,
	tolerance,
	blockSize,
	verbose = false,
) {
	const forwardOctant = directionOctant

	let forwardFaceMinValues = null
	let forwardFaceMinmaxValues = null
	let forwardFaceMaxValues = null
	let forwardFaceShadows = null
	let forwardCellShadows = null
	let backwardFaceHoles = null
	let backwardFaceMinValues = null
	let backwardFaceMinmaxValues = null
	let backwardFaceMaxValues = null
	let backwardFaceShadows = null
	let backwardCellShadows = null
	let keepForwardCellShadows = false

	try {
		forwardFaceMinValues = await computeFaceMinValues( volume, dominantAxis, forwardOctant )
		if ( verbose ) await logFaceTensorMeanWebGPU( 'forwardFaceMinValues@WebGPUFaces', forwardFaceMinValues )

		forwardFaceMinmaxValues = await propagateFaceMinmaxValuesInPlace( forwardFaceMinValues, dominantAxis, forwardOctant )
		forwardFaceMinValues = null
		if ( verbose ) await logFaceTensorMeanWebGPU( 'forwardFaceMinmaxValues@WebGPUFaces', forwardFaceMinmaxValues )

		forwardFaceMaxValues = await computeFaceMaxValues( volume, dominantAxis, forwardOctant )
		forwardFaceShadows = await computeFaceShadows( forwardFaceMaxValues, forwardFaceMinmaxValues, dominantAxis, forwardOctant, tolerance )
		if ( verbose ) await logFaceTensorMeanWebGPU( 'forwardFaceShadows@WebGPUFaces', forwardFaceShadows )

		forwardFaceMinmaxValues.dispose()
		forwardFaceMinmaxValues = null
		forwardFaceMaxValues.dispose()
		forwardFaceMaxValues = null

		forwardCellShadows = await computeCellShadows( forwardFaceShadows )
		if ( verbose ) await logTensor3DMeanWebGPU( 'forwardCellShadows@WebGPUFaces', forwardCellShadows )

		forwardFaceShadows.dispose()
		forwardFaceShadows = null

		const backwardOctant = reverseOctant( directionOctant )

		backwardFaceHoles = await computeFaceHoles( forwardCellShadows, dominantAxis, backwardOctant )
		backwardFaceMinValues = await computeFaceMinValues( volume, dominantAxis, backwardOctant )
		await hollowFaceMinValuesInPlace( backwardFaceMinValues, backwardFaceHoles )
		if ( verbose ) await logFaceTensorMeanWebGPU( 'backwardHollowFaceMinValues@WebGPUFaces', backwardFaceMinValues )

		backwardFaceHoles.dispose()
		backwardFaceHoles = null

		backwardFaceMinmaxValues = await propagateFaceMinmaxValuesInPlace( backwardFaceMinValues, dominantAxis, backwardOctant )
		backwardFaceMinValues = null
		if ( verbose ) await logFaceTensorMeanWebGPU( 'backwardFaceMinmaxValues@WebGPUFaces', backwardFaceMinmaxValues )

		backwardFaceMaxValues = await computeFaceMaxValues( volume, dominantAxis, backwardOctant )
		backwardFaceShadows = await computeFaceShadows( backwardFaceMaxValues, backwardFaceMinmaxValues, dominantAxis, backwardOctant, tolerance )
		if ( verbose ) await logFaceTensorMeanWebGPU( 'backwardFaceShadows@WebGPUFaces', backwardFaceShadows )

		backwardFaceMinmaxValues.dispose()
		backwardFaceMinmaxValues = null
		backwardFaceMaxValues.dispose()
		backwardFaceMaxValues = null

		backwardCellShadows = await computeCellShadows( backwardFaceShadows )
		if ( verbose ) await logTensor3DMeanWebGPU( 'backwardCellShadows@WebGPUFaces', backwardCellShadows )

		backwardFaceShadows.dispose()
		backwardFaceShadows = null

		await logicalOrInPlaceWebGPU( forwardCellShadows, backwardCellShadows )
		if ( verbose ) await logTensor3DMeanWebGPU( 'bidirectionalCellShadows@WebGPUFaces', forwardCellShadows )

		backwardCellShadows.dispose()
		backwardCellShadows = null

		if ( blockSize === 1 ) {
			keepForwardCellShadows = true
			return forwardCellShadows
		}

		const blockShadows = await minPoolBool3dWebGPU( forwardCellShadows, blockSize )
		if ( verbose ) await logTensor3DMeanWebGPU( 'bidirectionalBlockShadows@WebGPUFaces', blockShadows )

		forwardCellShadows.dispose()
		forwardCellShadows = null

		return blockShadows
	} finally {
		forwardFaceMinValues?.dispose()
		forwardFaceMinmaxValues?.dispose()
		forwardFaceMaxValues?.dispose()
		forwardFaceShadows?.dispose()
		if ( ! keepForwardCellShadows ) forwardCellShadows?.dispose()
		backwardFaceHoles?.dispose()
		backwardFaceMinValues?.dispose()
		backwardFaceMinmaxValues?.dispose()
		backwardFaceMaxValues?.dispose()
		backwardFaceShadows?.dispose()
		backwardCellShadows?.dispose()
	}
}

export async function computeUnidirectionalShadowMapFacesWebGPU(
	volume,
	dominantAxis,
	directionOctant,
	tolerance,
	blockSize,
	verbose = false,
) {
	let faceMinValues = null
	let faceMinmaxValues = null
	let faceMaxValues = null
	let faceShadows = null
	let cellShadows = null
	let keepCellShadows = false

	try {
		faceMinValues = await computeFaceMinValues( volume, dominantAxis, directionOctant )
		faceMinmaxValues = await propagateFaceMinmaxValuesInPlace( faceMinValues, dominantAxis, directionOctant )
		faceMinValues = null

		faceMaxValues = await computeFaceMaxValues( volume, dominantAxis, directionOctant )
		faceShadows = await computeFaceShadows( faceMaxValues, faceMinmaxValues, dominantAxis, directionOctant, tolerance )

		faceMinmaxValues.dispose()
		faceMinmaxValues = null
		faceMaxValues.dispose()
		faceMaxValues = null

		cellShadows = await computeCellShadows( faceShadows )
		if ( verbose ) await logTensor3DMeanWebGPU( 'cellShadows@WebGPUFaces', cellShadows )

		faceShadows.dispose()
		faceShadows = null

		if ( blockSize === 1 ) {
			keepCellShadows = true
			return cellShadows
		}

		const blockShadows = await minPoolBool3dWebGPU( cellShadows, blockSize )
		cellShadows.dispose()
		cellShadows = null
		if ( verbose ) await logTensor3DMeanWebGPU( 'blockShadows@WebGPUFaces', blockShadows )
		return blockShadows
	} finally {
		faceMinValues?.dispose()
		faceMinmaxValues?.dispose()
		faceMaxValues?.dispose()
		faceShadows?.dispose()
		if ( ! keepCellShadows ) cellShadows?.dispose()
	}
}

async function computeFaceMinValues( volume, axis, octant ) {
	return computeFaceExtremaValues( volume, axis, octant, 'min' )
}

async function computeFaceMaxValues( volume, axis, octant ) {
	return computeFaceExtremaValues( volume, axis, octant, 'max' )
}

async function computeFaceExtremaValues( volume, axis, octant, mode ) {
	const [ width, height, depth ] = volume.shape
	const output = WebGPUFaceTensor3D.empty(
		volume.device,
		[ width + 1, height + 1, depth + 1 ],
		'float32',
		`webgpu-face-${mode}-values`,
	)

	await runComputeProgram( volume.device, {
		label: `compute-face-${mode}-values-${axis}-${octant}`,
		code: computeFaceExtremaValuesWGSL( volume.shape, output.shape, axis, octant, mode ),
		bindings: [ { buffer: volume.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function propagateFaceMinmaxValuesInPlace( faceValues, axis, octant ) {
	const rawFaceValues = faceValues.clone( `${faceValues.label}:raw` )
	const dimension = axisToShapeIndex( axis )
	const sign = getOctantSign( octant, axis )
	const backwards = sign === '-'
	const slices = faceValues.shape[ dimension ]
	const start = backwards ? slices - 2 : 1
	const end = backwards ? -1 : slices
	const step = backwards ? -1 : 1
	const steps = []
	const paramsBuffers = []
	const dispatch = dispatchForPlane( faceValues.shape, axis )

	for ( let slice = start; slice !== end; slice += step ) {
		const paramsBuffer = createUniformBuffer(
			faceValues.device,
			new Int32Array( [ slice, 0, 0, 0 ] ),
			`propagate-face-minmax-${axis}-${octant}-${slice}:params`,
		)

		paramsBuffers.push( paramsBuffer )
		steps.push( {
			bindings: [
				{ buffer: rawFaceValues.buffer },
				{ buffer: faceValues.buffer },
				{ buffer: paramsBuffer },
			],
			dispatch,
		} )
	}

	await runComputeProgramSequence( faceValues.device, {
		label: `propagate-face-minmax-${axis}-${octant}`,
		code: propagateFaceMinmaxInPlaceWGSL( faceValues.shape, axis, octant, step ),
		steps,
		awaitCompletion: true,
		disposeAfterSubmit: paramsBuffers,
	} )

	rawFaceValues.dispose()
	return faceValues
}

export async function iterateFaceMinmaxValuesWebGPU( faceValues, axis, octant, iterations = null ) {
	const totalIterations = iterations ?? Math.max( 0, faceValues.width + faceValues.height + faceValues.depth - 3 )
	const code = iterateFaceMinmaxWGSL( faceValues.shape, axis, octant )
	let prev = faceValues.clone( `${faceValues.label}:iterate-prev` )
	let next = WebGPUFaceTensor3D.empty( faceValues.device, faceValues.shape, faceValues.dtype, `${faceValues.label}:iterate-next` )

	if ( totalIterations === 0 ) {
		next.dispose()
		return prev
	}

	try {
		for ( let i = 0; i < totalIterations; i += 1 ) {
			await runComputeProgram( faceValues.device, {
				label: `iterate-face-minmax-${axis}-${octant}`,
				code,
				bindings: [
					{ buffer: prev.buffer },
					{ buffer: next.buffer },
				],
				dispatch: dispatchForShape( faceValues.shape, WORKGROUP_SIZE_3D ),
			} )

			const tmp = prev
			prev = next
			next = tmp
		}

		next.dispose()
		next = null
		return prev
	} finally {
		next?.dispose()
	}
}

async function computeFaceShadows( faceMaxValues, faceMinmaxValues, axis, octant, tolerance ) {
	const output = WebGPUFaceTensor3D.empty( faceMaxValues.device, faceMaxValues.shape, 'uint32', 'webgpu-face-shadows' )

	await runComputeProgram( faceMaxValues.device, {
		label: `compute-face-shadows-${axis}-${octant}`,
		code: computeFaceShadowsWGSL( faceMaxValues.shape, axis, octant, tolerance ),
		bindings: [
			{ buffer: faceMaxValues.buffer },
			{ buffer: faceMinmaxValues.buffer },
			{ buffer: output.buffer },
		],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function computeCellShadows( faceShadows ) {
	const output = WebGPUTensor3D.empty( faceShadows.device, faceShadows.shape, 'uint32', 'webgpu-face-cell-shadows' )

	await runComputeProgram( faceShadows.device, {
		label: 'compute-cell-shadows-from-faces',
		code: computeCellShadowsWGSL( faceShadows.shape ),
		bindings: [ { buffer: faceShadows.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function computeFaceHoles( cellShadows, axis, octant ) {
	const output = WebGPUFaceTensor3D.empty( cellShadows.device, cellShadows.shape, 'uint32', 'webgpu-face-holes' )

	await runComputeProgram( cellShadows.device, {
		label: `compute-face-holes-${axis}-${octant}`,
		code: computeFaceHolesWGSL( cellShadows.shape, axis, octant ),
		bindings: [ { buffer: cellShadows.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )

	return output
}

async function hollowFaceMinValuesInPlace( faceMinValues, faceHoles ) {
	await runComputeProgram( faceMinValues.device, {
		label: 'hollow-face-min-values-in-place',
		code: hollowFaceMinValuesInPlaceWGSL( faceMinValues.shape ),
		bindings: [
			{ buffer: faceMinValues.buffer },
			{ buffer: faceHoles.buffer },
		],
		dispatch: dispatchForShape( faceMinValues.shape, WORKGROUP_SIZE_3D ),
	} )

	return faceMinValues
}

async function logFaceTensorMeanWebGPU( name, tensor ) {
	const data = await tensor.read()
	let sum = 0
	for ( let i = 0; i < data.length; i += 1 ) sum += Number( data[ i ] )
	console.log( name, sum / Math.max( data.length, 1 ) )
}

function computeFaceExtremaValuesWGSL( volumeShape, faceShape, axis, octant, mode ) {
	const [ width, height, depth ] = volumeShape
	const extremaFn = mode === 'min' ? 'min4' : 'max4'

	return /* wgsl */ `
	${commonTensor3DWGSL( volumeShape )}
	${commonFaceTensorWGSL( faceShape )}

	@group(0) @binding(0) var<storage, read> volume: array<f32>;
	@group(0) @binding(1) var<storage, read_write> face_values: array<f32>;

	fn volume_at(coords: vec3<i32>) -> f32
	{
	    let clamped = clamp(coords, vec3<i32>(0), vec3<i32>(${width - 1}, ${height - 1}, ${depth - 1}));
	    return volume[index3(clamped)];
	}

	struct CellValues {
	    v000: f32,
	    v100: f32,
	    v010: f32,
	    v001: f32,
	    v011: f32,
	    v101: f32,
	    v110: f32,
	    v111: f32,
	};

	fn sample_cell_values(cell_coords: vec3<i32>) -> CellValues
	{
	    let voxel_coords = cell_coords - vec3<i32>(1);

	    return CellValues(
	        volume_at(voxel_coords + ${vec3(voxelOffset(0, 0, 0, axis, octant))}),
	        volume_at(voxel_coords + ${vec3(voxelOffset(1, 0, 0, axis, octant))}),
	        volume_at(voxel_coords + ${vec3(voxelOffset(0, 1, 0, axis, octant))}),
	        volume_at(voxel_coords + ${vec3(voxelOffset(0, 0, 1, axis, octant))}),
	        volume_at(voxel_coords + ${vec3(voxelOffset(0, 1, 1, axis, octant))}),
	        volume_at(voxel_coords + ${vec3(voxelOffset(1, 0, 1, axis, octant))}),
	        volume_at(voxel_coords + ${vec3(voxelOffset(1, 1, 0, axis, octant))}),
	        volume_at(voxel_coords + ${vec3(voxelOffset(1, 1, 1, axis, octant))})
	    );
	}

	fn min4(a: f32, b: f32, c: f32, d: f32) -> f32 { return min(min(min(a, b), c), d); }
	fn max4(a: f32, b: f32, c: f32, d: f32) -> f32 { return max(max(max(a, b), c), d); }

	fn compute_face_values(cell_coords: vec3<i32>) -> vec3<f32>
	{
	    let c = sample_cell_values(cell_coords);
	    return vec3<f32>(
	        ${extremaFn}(c.v100, c.v110, c.v101, c.v111),
	        ${extremaFn}(c.v010, c.v110, c.v011, c.v111),
	        ${extremaFn}(c.v001, c.v011, c.v101, c.v111)
	    );
	}

	fn write_face_f32(gid: vec3<u32>, value: vec3<f32>)
	{
	    let base = face_base_u(gid);
	    face_values[base] = value.x;
	    face_values[base + 1u] = value.y;
	    face_values[base + 2u] = value.z;
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!face_in_bounds(gid)) { return; }
	    let values = compute_face_values(vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)));
	    write_face_f32(gid, values);
	}
`
}

function propagateFaceMinmaxInPlaceWGSL( shape, axis, octant, step ) {
	const prevAxis = axisDelta( axis, -step )
	const [ planeU, planeV ] = planeSize( shape, axis )

	return /* wgsl */ `
	${commonFaceTensorWGSL( shape )}

	struct Params {
	    slice: i32,
	    _pad0: i32,
	    _pad1: i32,
	    _pad2: i32,
	};

	@group(0) @binding(0) var<storage, read> raw_face_values: array<f32>;
	@group(0) @binding(1) var<storage, read_write> propagated_values: array<f32>;
	@group(0) @binding(2) var<uniform> params: Params;

	fn plane_coords(gid: vec3<u32>) -> vec3<i32>
	{
	    ${planeCoordsWGSL(axis)}
	}

	fn raw_at(coords: vec3<i32>) -> vec3<f32>
	{
	    if (!face_valid(coords)) { return vec3<f32>(${NEG_INF}); }
	    let base = face_base(coords);
	    return vec3<f32>(raw_face_values[base], raw_face_values[base + 1u], raw_face_values[base + 2u]);
	}

	fn propagated_at(coords: vec3<i32>) -> vec3<f32>
	{
	    if (!face_valid(coords)) { return vec3<f32>(${NEG_INF}); }
	    let base = face_base(coords);
	    return vec3<f32>(propagated_values[base], propagated_values[base + 1u], propagated_values[base + 2u]);
	}

	fn compute_x_face_minmax(c111: vec3<f32>, c110: vec3<f32>, c101: vec3<f32>, c100: vec3<f32>) -> f32
	{
	    let min_values = min(c110.z, max(c101.y, c100.z));
	    return max(c111.x, min_values);
	}

	fn compute_y_face_minmax(c111: vec3<f32>, c110: vec3<f32>, c011: vec3<f32>, c010: vec3<f32>) -> f32
	{
	    let min_values = min(c110.z, max(c011.x, c010.z));
	    return max(c111.y, min_values);
	}

	fn compute_z_face_minmax(c111: vec3<f32>, c110: vec3<f32>, c101: vec3<f32>, c011: vec3<f32>) -> f32
	{
	    let min_values = min(c110.z, min(c101.y, c011.x));
	    return max(c111.z, min_values);
	}

	fn compute_face_minmax(coords: vec3<i32>) -> vec3<f32>
	{
	    var c111 = raw_at(coords + ${vec3(sliceOffset( 0,  0,  0, axis, octant))});
	    var c011 = raw_at(coords + ${vec3(sliceOffset(-1,  0,  0, axis, octant))});
	    var c101 = raw_at(coords + ${vec3(sliceOffset( 0, -1,  0, axis, octant))});
	    let c001 = raw_at(coords + ${vec3(sliceOffset(-1, -1,  0, axis, octant))});
	    let c110 = propagated_at(coords + ${vec3(prevAxis)} + ${vec3(sliceOffset( 0,  0, -1, axis, octant))});
	    let c010 = propagated_at(coords + ${vec3(prevAxis)} + ${vec3(sliceOffset(-1,  0, -1, axis, octant))});
	    let c100 = propagated_at(coords + ${vec3(prevAxis)} + ${vec3(sliceOffset( 0, -1, -1, axis, octant))});
	    let c000 = propagated_at(coords + ${vec3(prevAxis)} + ${vec3(sliceOffset(-1, -1, -1, axis, octant))});

	    c111.x = compute_x_face_minmax(c111, c110, c101, c100);
	    c111.y = compute_y_face_minmax(c111, c110, c011, c010);
	    c011.x = compute_x_face_minmax(c011, c010, c001, c000);
	    c101.y = compute_y_face_minmax(c101, c100, c001, c000);
	    c111.z = compute_z_face_minmax(c111, c110, c101, c011);

	    return c111;
	}

	fn write_face_i32_f32(coords: vec3<i32>, value: vec3<f32>)
	{
	    let base = face_base(coords);
	    propagated_values[base] = value.x;
	    propagated_values[base + 1u] = value.y;
	    propagated_values[base + 2u] = value.z;
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_2D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= ${planeU}u || gid.y >= ${planeV}u) { return; }
	    let coords = plane_coords(gid);
	    write_face_i32_f32(coords, compute_face_minmax(coords));
	}
`
}

function iterateFaceMinmaxWGSL( shape, axis, octant ) {
	return /* wgsl */ `
	${commonFaceTensorWGSL( shape )}

	@group(0) @binding(0) var<storage, read> input_face_values: array<f32>;
	@group(0) @binding(1) var<storage, read_write> output_face_values: array<f32>;

	fn face_minmax_at(coords: vec3<i32>) -> vec3<f32>
	{
	    if (!face_valid(coords)) { return vec3<f32>(${NEG_INF}); }
	    let base = face_base(coords);
	    return vec3<f32>(input_face_values[base], input_face_values[base + 1u], input_face_values[base + 2u]);
	}

	fn compute_x_face_minmax(c111: vec3<f32>, c110: vec3<f32>, c101: vec3<f32>) -> f32
	{
	    let min_values = min(c110.z, c101.y);
	    return max(c111.x, min_values);
	}

	fn compute_y_face_minmax(c111: vec3<f32>, c110: vec3<f32>, c011: vec3<f32>) -> f32
	{
	    let min_values = min(c110.z, c011.x);
	    return max(c111.y, min_values);
	}

	fn compute_z_face_minmax(c111: vec3<f32>, c110: vec3<f32>, c101: vec3<f32>, c011: vec3<f32>) -> f32
	{
	    let min_values = min(c110.z, min(c101.y, c011.x));
	    return max(c111.z, min_values);
	}

	fn compute_face_minmax(coords: vec3<i32>) -> vec3<f32>
	{
	    let c111 = face_minmax_at(coords + ${vec3(cellOffset( 0,  0,  0, axis, octant))});
	    let c011 = face_minmax_at(coords + ${vec3(cellOffset(-1,  0,  0, axis, octant))});
	    let c101 = face_minmax_at(coords + ${vec3(cellOffset( 0, -1,  0, axis, octant))});
	    let c110 = face_minmax_at(coords + ${vec3(cellOffset( 0,  0, -1, axis, octant))});

	    return vec3<f32>(
	        compute_x_face_minmax(c111, c110, c101),
	        compute_y_face_minmax(c111, c110, c011),
	        compute_z_face_minmax(c111, c110, c101, c011)
	    );
	}

	fn write_output(gid: vec3<u32>, value: vec3<f32>)
	{
	    let base = face_base_u(gid);
	    output_face_values[base] = value.x;
	    output_face_values[base + 1u] = value.y;
	    output_face_values[base + 2u] = value.z;
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!face_in_bounds(gid)) { return; }
	    let coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
	    write_output(gid, compute_face_minmax(coords));
	}
`
}

function computeFaceShadowsWGSL( shape, axis, octant, tolerance ) {
	return /* wgsl */ `
	${commonFaceTensorWGSL( shape )}

	const TOLERANCE: f32 = ${tolerance};

	@group(0) @binding(0) var<storage, read> face_max_values: array<f32>;
	@group(0) @binding(1) var<storage, read> face_minmax_values: array<f32>;
	@group(0) @binding(2) var<storage, read_write> face_shadows: array<u32>;

	fn face_max_at(coords: vec3<i32>) -> vec3<f32>
	{
	    if (!face_valid(coords)) { return vec3<f32>(${NEG_INF}); }
	    let base = face_base(coords);
	    return vec3<f32>(face_max_values[base], face_max_values[base + 1u], face_max_values[base + 2u]);
	}

	fn face_minmax_at(coords: vec3<i32>) -> vec3<f32>
	{
	    if (!face_valid(coords)) { return vec3<f32>(${NEG_INF}); }
	    let base = face_base(coords);
	    return vec3<f32>(face_minmax_values[base], face_minmax_values[base + 1u], face_minmax_values[base + 2u]);
	}

	fn compute_face_shadows(coords: vec3<i32>) -> vec3<u32>
	{
	    let c111 = face_max_at(coords + ${vec3(cellOffset( 0,  0,  0, axis, octant))});
	    let c110 = face_minmax_at(coords + ${vec3(cellOffset( 0,  0, -1, axis, octant))});
	    let c101 = face_minmax_at(coords + ${vec3(cellOffset( 0, -1,  0, axis, octant))});
	    let c011 = face_minmax_at(coords + ${vec3(cellOffset(-1,  0,  0, axis, octant))});

	    let margins = vec3<f32>(
	        c111.x - min(c110.z, c101.y),
	        c111.y - min(c110.z, c011.x),
	        c111.z - min(c110.z, min(c101.y, c011.x))
	    );

	    return vec3<u32>(
	        select(0u, 1u, margins.x <= TOLERANCE),
	        select(0u, 1u, margins.y <= TOLERANCE),
	        select(0u, 1u, margins.z <= TOLERANCE)
	    );
	}

	fn write_face_shadows(gid: vec3<u32>, value: vec3<u32>)
	{
	    let base = face_base_u(gid);
	    face_shadows[base] = value.x;
	    face_shadows[base + 1u] = value.y;
	    face_shadows[base + 2u] = value.z;
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!face_in_bounds(gid)) { return; }
	    let values = compute_face_shadows(vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)));
	    write_face_shadows(gid, values);
	}
`
}

function computeCellShadowsWGSL( shape ) {
	return /* wgsl */ `
	${commonFaceTensorWGSL( shape )}
	${commonTensor3DWGSL( shape )}

	@group(0) @binding(0) var<storage, read> face_shadows: array<u32>;
	@group(0) @binding(1) var<storage, read_write> cell_shadows: array<u32>;

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!face_in_bounds(gid)) { return; }
	    let base = face_base_u(gid);
	    let shadow = face_shadows[base] & face_shadows[base + 1u] & face_shadows[base + 2u];
	    cell_shadows[index3u(gid)] = shadow;
	}
`
}

function computeFaceHolesWGSL( shape, axis, octant ) {
	return /* wgsl */ `
	${commonTensor3DWGSL( shape )}
	${commonFaceTensorWGSL( shape )}

	@group(0) @binding(0) var<storage, read> cell_shadows: array<u32>;
	@group(0) @binding(1) var<storage, read_write> face_holes: array<u32>;

	fn cell_shadow_at(coords: vec3<i32>) -> bool
	{
	    let clamped = clamp3(coords);
	    return cell_shadows[index3(clamped)] != 0u;
	}

	fn write_face_holes(gid: vec3<u32>, value: vec3<u32>)
	{
	    let base = face_base_u(gid);
	    face_holes[base] = value.x;
	    face_holes[base + 1u] = value.y;
	    face_holes[base + 2u] = value.z;
	}

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!face_in_bounds(gid)) { return; }
	    let coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
	    let hole = select(0u, 1u, cell_shadow_at(coords + ${vec3(cellOffset(0, 0, 0, axis, octant))}));
	    write_face_holes(gid, vec3<u32>(hole));
	}
`
}

function hollowFaceMinValuesInPlaceWGSL( shape ) {
	return /* wgsl */ `
	${commonFaceTensorWGSL( shape )}

	@group(0) @binding(0) var<storage, read_write> face_min_values: array<f32>;
	@group(0) @binding(1) var<storage, read> face_holes: array<u32>;

	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (!face_in_bounds(gid)) { return; }
	    let base = face_base_u(gid);
	    face_min_values[base] = select(face_min_values[base], ${NEG_INF}, face_holes[base] != 0u);
	    face_min_values[base + 1u] = select(face_min_values[base + 1u], ${NEG_INF}, face_holes[base + 1u] != 0u);
	    face_min_values[base + 2u] = select(face_min_values[base + 2u], ${NEG_INF}, face_holes[base + 2u] != 0u);
	}
`
}

function commonFaceTensorWGSL( shape ) {
	const [ width, height, depth ] = shape

	return /* wgsl */ `
	const FACE_WIDTH: u32 = ${width}u;
	const FACE_HEIGHT: u32 = ${height}u;
	const FACE_DEPTH: u32 = ${depth}u;

	fn face_index(coords: vec3<i32>) -> u32
	{
	    return ((u32(coords.z) * FACE_HEIGHT * FACE_WIDTH) + (u32(coords.y) * FACE_WIDTH) + u32(coords.x)) * 3u;
	}

	fn face_base(coords: vec3<i32>) -> u32
	{
	    return face_index(coords);
	}

	fn face_base_u(coords: vec3<u32>) -> u32
	{
	    return ((coords.z * FACE_HEIGHT * FACE_WIDTH) + (coords.y * FACE_WIDTH) + coords.x) * 3u;
	}

	fn face_valid(coords: vec3<i32>) -> bool
	{
	    return coords.x >= 0 && coords.x < i32(FACE_WIDTH) &&
	           coords.y >= 0 && coords.y < i32(FACE_HEIGHT) &&
	           coords.z >= 0 && coords.z < i32(FACE_DEPTH);
	}

	fn face_in_bounds(gid: vec3<u32>) -> bool
	{
	    return gid.x < FACE_WIDTH && gid.y < FACE_HEIGHT && gid.z < FACE_DEPTH;
	}

`
}

function dispatchForPlane( shape, axis ) {
	const [ u, v ] = planeSize( shape, axis )
	return [
		Math.ceil( u / WORKGROUP_SIZE_2D[ 0 ] ),
		Math.ceil( v / WORKGROUP_SIZE_2D[ 1 ] ),
		1,
	]
}

function voxelOffset( x, y, z, axis, octant ) {
	return unitOffset( x, y, z, axis, octant )
}

function cellOffset( x, y, z, axis, octant ) {
	return signedOffset( x, y, z, axis, octant )
}

function sliceOffset( x, y, z, axis, octant ) {
	return shadowSliceOffset( x, y, z, axis, octant )
}
