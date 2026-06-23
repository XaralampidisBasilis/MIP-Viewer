import { runComputeProgram, dispatchForShape } from '../../WebGPU/WebGPUComputeUtils'
import { WebGPUTensor3D } from '../../WebGPU/WebGPUTensor3D'

const WORKGROUP_SIZE_3D = [ 8, 8, 4 ]

export async function maxPool3dWebGPU( input, blockSize, options = {} ) 
{
	const {
		disposeInput = false,
	} = options

	if ( input.dtype !== 'float32' && input.dtype !== 'uint32' && input.dtype !== 'int32' ) {
		throw new Error( `maxPool3dWebGPU only supports float32, uint32, and int32 tensors, got "${input.dtype}".` )
	}

	if ( input.size <= 0 ) {
		throw new Error( 'maxPool3dWebGPU expected a non-empty tensor.' )
	}

	if ( !Number.isInteger( blockSize ) || blockSize <= 0 ) {
		throw new Error( `maxPool3dWebGPU expected a positive integer blockSize, got ${blockSize}.` )
	}

	validateShape3D( input.shape, 'input.shape' )

	const outputShape = input.shape.map( ( size ) => Math.ceil( size / blockSize ) )

	const output = WebGPUTensor3D.empty(
		input.device,
		outputShape,
		input.dtype,
		'max-pool3d-output',
	)

	await runComputeProgram( input.device, {
		label: `max-pool3d-${input.dtype}`,
		code: maxPool3dWGSL( input.shape, outputShape, blockSize, input.dtype, WORKGROUP_SIZE_3D ),
		bindings: [
			{ buffer: input.buffer },
			{ buffer: output.buffer },
		],
		dispatch: dispatchForShape( output.shape, WORKGROUP_SIZE_3D ),
		awaitCompletion: disposeInput,
	} )

	if ( disposeInput ) {
		input.dispose()
	}

	return output
}

function validateShape3D( shape, label ) 
{
	if ( !Array.isArray( shape ) || shape.length !== 3 ) {
		throw new Error( `${label} expected [width, height, depth], got ${JSON.stringify( shape )}.` )
	}

	const [ width, height, depth ] = shape

	if (
		!Number.isInteger( width ) ||
		!Number.isInteger( height ) ||
		!Number.isInteger( depth ) ||
		width <= 0 ||
		height <= 0 ||
		depth <= 0
	) {
		throw new Error( `${label} values must be positive integers, got [${shape.join( ', ' )}].` )
	}
}

function workgroupSizeWGSL( workgroupSize ) 
{
	return `@compute @workgroup_size(${workgroupSize[ 0 ]}, ${workgroupSize[ 1 ]}, ${workgroupSize[ 2 ]})`
}

function dtypeInfoWGSL( dtype ) 
{
	if ( dtype === 'float32' ) {
		return {
			type: 'f32',
			neutralMax: '-3.402823466e+38',
		}
	}

	if ( dtype === 'uint32' ) {
		return {
			type: 'u32',
			neutralMax: '0u',
		}
	}

	if ( dtype === 'int32' ) {
		return {
			type: 'i32',
			neutralMax: '-2147483648',
		}
	}

	throw new Error( `Unsupported dtype "${dtype}".` )
}

function maxPool3dWGSL( inputShape, outputShape, blockSize, dtype, workgroupSize ) 
{
	const [ width, height, depth ] = inputShape
	const [ outWidth, outHeight, outDepth ] = outputShape

	const dtypeInfo = dtypeInfoWGSL( dtype )

	return /* wgsl */ `

        const DEPTH: u32 = ${depth}u;
        const HEIGHT: u32 = ${height}u;
        const WIDTH: u32 = ${width}u;

        const OUT_DEPTH: u32 = ${outDepth}u;
        const OUT_HEIGHT: u32 = ${outHeight}u;
        const OUT_WIDTH: u32 = ${outWidth}u;

        const BLOCK_SIZE: i32 = ${blockSize};

        const NEUTRAL_MAX: ${dtypeInfo.type} = ${dtypeInfo.neutralMax};

        @group(0) @binding(0) var<storage, read> input_values: array<${dtypeInfo.type}>;
        @group(0) @binding(1) var<storage, read_write> output_values: array<${dtypeInfo.type}>;

        fn index3(coords: vec3<i32>) -> u32
        {
            return u32(coords.z) * HEIGHT * WIDTH + u32(coords.y) * WIDTH + u32(coords.x);
        }

        fn out_index_u(coords: vec3<u32>) -> u32
        {
            return coords.z * OUT_HEIGHT * OUT_WIDTH + coords.y * OUT_WIDTH + coords.x;
        }

        ${workgroupSizeWGSL( workgroupSize )}
        fn main(@builtin(global_invocation_id) gid: vec3<u32>)
        {
            if (gid.x >= OUT_WIDTH || gid.y >= OUT_HEIGHT || gid.z >= OUT_DEPTH) { return; }

            let out_coords = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
            let start = out_coords * vec3<i32>(BLOCK_SIZE);

            let end = min(
                start + vec3<i32>(BLOCK_SIZE),
                vec3<i32>(i32(WIDTH), i32(HEIGHT), i32(DEPTH)),
            );

            var value = NEUTRAL_MAX;

            for (var dz = start.z; dz < end.z; dz = dz + 1) {
                for (var dy = start.y; dy < end.y; dy = dy + 1) {
                    for (var dx = start.x; dx < end.x; dx = dx + 1) {
                        let sample = input_values[index3(vec3<i32>(dx, dy, dz))];
                        value = max(value, sample);
                    }
                }
            }

            output_values[out_index_u(gid)] = value;
        }
        `
}