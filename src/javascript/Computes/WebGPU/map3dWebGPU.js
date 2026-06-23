import { createUniformBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { runComputeProgram, dispatchForShape } from '../../WebGPU/WebGPUComputeUtils'
import { WebGPUTensor3D } from '../../WebGPU/WebGPUTensor3D'

const WORKGROUP_SIZE_3D = [ 16, 16, 1 ]

export async function map3dWebGPU( input, minValue, maxValue, options = {} ) 
{
	const { inPlace = false } = options

	if ( input.dtype !== 'float32' ) {
		throw new Error( `map3dWebGPU only supports float32 tensors, got "${input.dtype}".` )
	}

	if ( input.size <= 0 ) {
		throw new Error( 'map3dWebGPU expected a non-empty tensor.' )
	}

	if ( !Number.isFinite( minValue ) || !Number.isFinite( maxValue ) ) {
		throw new Error( `map3dWebGPU expected finite minValue and maxValue, got ${minValue}, ${maxValue}.` )
	}

	const output = inPlace ?
		input :
		WebGPUTensor3D.empty( input.device, input.shape, 'float32', 'map3d-output' )

	const paramsBuffer = map3dParamsBuffer(
		input,
		minValue,
		maxValue,
		inPlace ? 'map3d-in-place-params' : 'map3d-params',
	)

	await runComputeProgram( input.device, {
		label: inPlace ? 'map3d-in-place' : 'map3d',
		code: map3dWGSL( input.shape, inPlace, WORKGROUP_SIZE_3D ),
		bindings: inPlace ?
			[
				{ buffer: input.buffer },
				{ buffer: paramsBuffer },
			] :
			[
				{ buffer: input.buffer },
				{ buffer: output.buffer },
				{ buffer: paramsBuffer },
			],
		dispatch: dispatchForShape( input.shape, WORKGROUP_SIZE_3D ),
		awaitCompletion: true,
	} )

	paramsBuffer.destroy()

	return output
}

export async function map3dInPlaceWebGPU( input, minValue, maxValue ) 
{
	return map3dWebGPU( input, minValue, maxValue, { inPlace: true } )
}

function map3dParamsBuffer( input, minValue, maxValue, label ) 
{
	const range = maxValue - minValue
	const invRange = range === 0 ? 0 : 1 / range

	return createUniformBuffer(
		input.device,
		new Float32Array( [ minValue, invRange, 0, 0 ] ),
		label,
	)
}

function workgroupSizeWGSL( workgroupSize ) 
{
	return `@compute @workgroup_size(${workgroupSize[ 0 ]}, ${workgroupSize[ 1 ]}, ${workgroupSize[ 2 ]})`
}

function map3dWGSL( shape, inPlace, workgroupSize ) 
{
	const [ width, height, depth ] = shape

	if ( inPlace ) 
	{
		return /* wgsl */ `

        const DEPTH: u32 = ${depth}u;
        const HEIGHT: u32 = ${height}u;
        const WIDTH: u32 = ${width}u;

        struct Params {
            min_value: f32,
            inv_range: f32,
            _pad0: f32,
            _pad1: f32,
        };

        @group(0) @binding(0) var<storage, read_write> values: array<f32>;
        @group(0) @binding(1) var<uniform> params: Params;

        fn index3u(coords: vec3<u32>) -> u32
        {
            return coords.z * HEIGHT * WIDTH + coords.y * WIDTH + coords.x;
        }

        ${workgroupSizeWGSL( workgroupSize )}
        fn main(@builtin(global_invocation_id) gid: vec3<u32>)
        {
            if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

            let idx = index3u(gid);
            let mapped = (values[idx] - params.min_value) * params.inv_range;
            values[idx] = clamp(mapped, 0.0, 1.0);
        }
        `
	}

	return /* wgsl */ `

        const DEPTH: u32 = ${depth}u;
        const HEIGHT: u32 = ${height}u;
        const WIDTH: u32 = ${width}u;

        struct Params {
            min_value: f32,
            inv_range: f32,
            _pad0: f32,
            _pad1: f32,
        };

        @group(0) @binding(0) var<storage, read> input_values: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
        @group(0) @binding(2) var<uniform> params: Params;

        fn index3u(coords: vec3<u32>) -> u32
        {
            return coords.z * HEIGHT * WIDTH + coords.y * WIDTH + coords.x;
        }

        ${workgroupSizeWGSL( workgroupSize )}
        fn main(@builtin(global_invocation_id) gid: vec3<u32>)
        {
            if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) { return; }

            let idx = index3u(gid);
            let mapped = (input_values[idx] - params.min_value) * params.inv_range;
            output_values[idx] = clamp(mapped, 0.0, 1.0);
        }
        `
}