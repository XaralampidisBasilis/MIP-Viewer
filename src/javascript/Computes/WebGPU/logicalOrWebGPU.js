import { runComputeProgram, dispatchForShape } from '../../WebGPU/WebGPUComputeUtils'
import { WebGPUTensor3D } from '../../WebGPU/WebGPUTensor3D'

const WORKGROUP_SIZE_3D = [ 16, 16, 1 ]

export async function logicalOrWebGPU( a, b, options = {} ) 
{
	const {
		inPlace = false,
		awaitCompletion = false,
	} = options

	if ( a.dtype !== 'uint32' ) {
		throw new Error( `logicalOrWebGPU only supports uint32 tensors, got "${a.dtype}" for a.` )
	}

	if ( b.dtype !== 'uint32' ) {
		throw new Error( `logicalOrWebGPU only supports uint32 tensors, got "${b.dtype}" for b.` )
	}

	if ( a.size <= 0 || b.size <= 0 ) {
		throw new Error( 'logicalOrWebGPU expected non-empty tensors.' )
	}

	validateSameShape( a.shape, b.shape )

	const output = inPlace ?
		a :
		WebGPUTensor3D.empty( a.device, a.shape, 'uint32', 'logical-or-output' )

	await runComputeProgram( a.device, {
		label: inPlace ? 'logical-or-in-place' : 'logical-or',
		code: logicalOrWGSL( a.shape, inPlace, WORKGROUP_SIZE_3D ),
		bindings: inPlace ?
			[
				{ buffer: a.buffer },
				{ buffer: b.buffer },
			] :
			[
				{ buffer: a.buffer },
				{ buffer: b.buffer },
				{ buffer: output.buffer },
			],
		dispatch: dispatchForShape( a.shape, WORKGROUP_SIZE_3D ),
		awaitCompletion,
	} )

	return output
}

export async function logicalOrInPlaceWebGPU( a, b, options = {} ) 
{
	return logicalOrWebGPU( a, b, { ...options, inPlace: true } )
}

function validateSameShape( aShape, bShape ) 
{
	if (
		aShape[ 0 ] !== bShape[ 0 ] ||
		aShape[ 1 ] !== bShape[ 1 ] ||
		aShape[ 2 ] !== bShape[ 2 ]
	) {
		throw new Error(
			`logicalOrWebGPU expected tensors with the same shape, got [${aShape.join( ', ' )}] and [${bShape.join( ', ' )}].`,
		)
	}
}

function workgroupSizeWGSL( workgroupSize ) 
{
	return `@compute @workgroup_size(${workgroupSize[ 0 ]}, ${workgroupSize[ 1 ]}, ${workgroupSize[ 2 ]})`
}

function logicalOrWGSL( shape, inPlace, workgroupSize ) 
{
	const [ width, height, depth ] = shape

	if ( inPlace ) 
	{
		return /* wgsl */ `

        const DEPTH: u32 = ${depth}u;
        const HEIGHT: u32 = ${height}u;
        const WIDTH: u32 = ${width}u;

        @group(0) @binding(0) var<storage, read_write> a_values: array<u32>;
        @group(0) @binding(1) var<storage, read> b_values: array<u32>;

        fn index3u(coords: vec3<u32>) -> u32
        {
            return coords.z * HEIGHT * WIDTH + coords.y * WIDTH + coords.x;
        }

        ${workgroupSizeWGSL( workgroupSize )}
        fn main(@builtin(global_invocation_id) gid: vec3<u32>)
        {
            if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

            let idx = index3u(gid);
            a_values[idx] = a_values[idx] | b_values[idx];
        }
        `
	}

	return /* wgsl */ `

        const DEPTH: u32 = ${depth}u;
        const HEIGHT: u32 = ${height}u;
        const WIDTH: u32 = ${width}u;

        @group(0) @binding(0) var<storage, read> a_values: array<u32>;
        @group(0) @binding(1) var<storage, read> b_values: array<u32>;
        @group(0) @binding(2) var<storage, read_write> output_values: array<u32>;

        fn index3u(coords: vec3<u32>) -> u32
        {
            return coords.z * HEIGHT * WIDTH + coords.y * WIDTH + coords.x;
        }

        ${workgroupSizeWGSL( workgroupSize )}
        fn main(@builtin(global_invocation_id) gid: vec3<u32>)
        {
            if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

            let idx = index3u(gid);
            output_values[idx] = a_values[idx] | b_values[idx];
        }
        `
}
