import { dispatchForShape, runComputeProgram } from './WebGPUComputeRunner'
import { WebGPUTensor3D } from '../../WebGPU/WebGPUTensor3D'
import { getOctantSign } from './WebGPUShadowMapUtils'
import {
	WORKGROUP_SIZE_3D,
	axisSize,
	axisStride,
	commonTensor3DWGSL,
	logTensor3DMeanWebGPU,
	sampleIndexWGSL,
	workgroupSizeWGSL,
} from './WebGPUKernelUtils'

export { logTensor3DMeanWebGPU }

export async function computeUnidirectionalDistanceMapWebGPU(
	shadowMap,
	dominantAxis,
	directionOctant,
	maxDistance,
	verbose = false,
) {
	const sweepAxes = [ 'x', 'y', 'z' ].filter( ( axis ) => axis !== dominantAxis )
	const sweepSigns = sweepAxes.map( ( axis ) => getOctantSign( directionOctant, axis ) )
	const dominantSign = getOctantSign( directionOctant, dominantAxis )

	const distances0d = await initialChebyshevDistanceInPlace( shadowMap, maxDistance )
	if ( verbose ) await logTensor3DMeanWebGPU( 'initialDistances@WebGPU', distances0d )

	const distances1d = await anisotropicChebyshevDistance( distances0d, sweepAxes[ 0 ], sweepSigns[ 0 ], maxDistance )
	distances0d.dispose()
	if ( verbose ) await logTensor3DMeanWebGPU( 'distances1d@WebGPU', distances1d )

	const distances2d = await anisotropicChebyshevDistance( distances1d, sweepAxes[ 1 ], sweepSigns[ 1 ], maxDistance )
	distances1d.dispose()
	if ( verbose ) await logTensor3DMeanWebGPU( 'distances2d@WebGPU', distances2d )

	const distances3d = await extendedChebyshevDistance( distances2d, dominantAxis, dominantSign, maxDistance )
	distances2d.dispose()
	if ( verbose ) await logTensor3DMeanWebGPU( 'distanceMap@WebGPU', distances3d )

	return distances3d
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
	const output = WebGPUTensor3D.empty( distances.device, distances.shape, 'uint32', `webgpu-anisotropic-distance-${axis}-${sign}` )
	await runComputeProgram( distances.device, {
		label: `anisotropic-chebyshev-distance-${axis}-${sign}`,
		code: anisotropicChebyshevDistanceWGSL( distances.shape, axis, sign, maxDistance ),
		bindings: [ { buffer: distances.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )
	return output
}

async function extendedChebyshevDistance( distances, axis, sign, maxDistance ) {
	const output = WebGPUTensor3D.empty( distances.device, distances.shape, 'uint32', `webgpu-extended-distance-${axis}-${sign}` )
	await runComputeProgram( distances.device, {
		label: `extended-chebyshev-distance-${axis}-${sign}`,
		code: extendedChebyshevDistanceWGSL( distances.shape, axis, sign, maxDistance ),
		bindings: [ { buffer: distances.buffer }, { buffer: output.buffer } ],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
	} )
	return output
}

function initialChebyshevDistanceInPlaceWGSL( shape, maxDistance ) {
	return /* wgsl */ `
	${commonTensor3DWGSL( shape )}
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
	const sampleIndexExpression = sampleIndexWGSL( 'out_index_value', 'radius', sign )
	return /* wgsl */ `
	${commonTensor3DWGSL( shape )}
	const MAX_DISTANCE: u32 = ${maxDistance}u;
	const MAX_RADIUS: i32 = min(i32(MAX_DISTANCE), i32(${axisSize( shape, axis )}u) - 1);
	const AXIS_LIMIT: i32 = i32(${axisSize( shape, axis )}u);
	const AXIS_STRIDE: u32 = ${axisStride( shape, axis )}u;
	@group(0) @binding(0) var<storage, read> input_distances: array<u32>;
	@group(0) @binding(1) var<storage, read_write> output_distances: array<u32>;
	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }
	    let out_index_value = index3u(gid);
	    let out_axis = i32(gid.${axis});
	    var min_distance = input_distances[out_index_value];
	    if (min_distance == 0u) {
	        output_distances[out_index_value] = min_distance;
	        return;
	    }
	    for (var radius = 1; radius <= MAX_RADIUS; radius = radius + 1) {
	        let sample_axis = out_axis ${sign} radius;
	        if (sample_axis < 0 || sample_axis >= AXIS_LIMIT) { break; }
	        let sample_distance = input_distances[${sampleIndexExpression}];
	        if (sample_distance == 0u) {
	            min_distance = u32(radius);
	            break;
	        }
	        min_distance = min(min_distance, max(sample_distance, u32(radius)));
	        if (min_distance <= u32(radius)) { break; }
	    }
	    output_distances[out_index_value] = min_distance;
	}
`
}

function extendedChebyshevDistanceWGSL( shape, axis, sign, maxDistance ) {
	const sampleIndexExpression = sampleIndexWGSL( 'out_index_value', 'radius', sign )
	return /* wgsl */ `
	${commonTensor3DWGSL( shape )}
	const MAX_DISTANCE: u32 = ${maxDistance}u;
	const MAX_RADIUS: i32 = min(i32(MAX_DISTANCE), i32(${axisSize( shape, axis )}u) - 1);
	const AXIS_LIMIT: i32 = i32(${axisSize( shape, axis )}u);
	const AXIS_STRIDE: u32 = ${axisStride( shape, axis )}u;
	@group(0) @binding(0) var<storage, read> input_distances: array<u32>;
	@group(0) @binding(1) var<storage, read_write> output_distances: array<u32>;
	${workgroupSizeWGSL( WORKGROUP_SIZE_3D )}
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }
	    let out_index_value = index3u(gid);
	    let out_axis = i32(gid.${axis});
	    var min_distance = MAX_DISTANCE;
	    for (var radius = 0; radius <= MAX_RADIUS; radius = radius + 1) {
	        let sample_axis = out_axis ${sign} radius;
	        if (sample_axis < 0 || sample_axis >= AXIS_LIMIT) { break; }
	        let sample_distance = input_distances[${sampleIndexExpression}];
	        if (sample_distance <= u32(radius)) {
	            min_distance = u32(radius);
	            break;
	        }
	    }
	    output_distances[out_index_value] = min_distance;
	}
`
}
