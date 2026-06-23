import { createUniformBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { runComputeProgram, dispatchForShape } from '../../WebGPU/WebGPUComputeUtils'
import { WebGPUTensor3D } from '../../WebGPU/WebGPUTensor3D'

const WORKGROUP_SIZE_3D = [ 8, 8, 4 ]

export async function resizeTrilinearWebGPU( input, outputShape, options = {} ) 
{
	const {
		alignCorners = false,
		halfPixelCenters = true,
		disposeInput = false,
	} = options

	if ( input.dtype !== 'float32' ) {
		throw new Error( `resizeTrilinearWebGPU only supports float32 tensors, got "${input.dtype}".` )
	}

	if ( input.size <= 0 ) {
		throw new Error( 'resizeTrilinearWebGPU expected a non-empty tensor.' )
	}

	validateShape3D( input.shape, 'input.shape' )
	validateShape3D( outputShape, 'outputShape' )

	const output = WebGPUTensor3D.empty(
		input.device,
		outputShape,
		'float32',
		'resize-trilinear-output',
	)

	const [ inWidth, inHeight, inDepth ] = input.shape
	const [ outWidth, outHeight, outDepth ] = outputShape

	const scaleWidth = resizeScale( inWidth, outWidth, alignCorners )
	const scaleHeight = resizeScale( inHeight, outHeight, alignCorners )
	const scaleDepth = resizeScale( inDepth, outDepth, alignCorners )

	const offsetFactor = halfPixelCenters ? 0.5 : 0.0

	const paramsBuffer = createUniformBuffer(
		input.device,
		new Float32Array( [ scaleWidth, scaleHeight, scaleDepth, offsetFactor ] ),
		'resize-trilinear-params',
	)

	await runComputeProgram( input.device, {
		label: 'resize-trilinear',
		code: resizeTrilinearWGSL( input.shape, outputShape, WORKGROUP_SIZE_3D ),
		bindings: [
			{ buffer: input.buffer },
			{ buffer: output.buffer },
			{ buffer: paramsBuffer },
		],
		dispatch: dispatchForShape( outputShape, WORKGROUP_SIZE_3D ),
		awaitCompletion: true,
	} )

	paramsBuffer.destroy()

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

function resizeScale( inputSize, outputSize, alignCorners ) 
{
	if ( alignCorners && outputSize > 1 ) {
		return ( inputSize - 1 ) / ( outputSize - 1 )
	}

	return inputSize / outputSize
}

function workgroupSizeWGSL( workgroupSize ) 
{
	return `@compute @workgroup_size(${workgroupSize[ 0 ]}, ${workgroupSize[ 1 ]}, ${workgroupSize[ 2 ]})`
}

function resizeTrilinearWGSL( inputShape, outputShape, workgroupSize ) 
{
	const [ inWidth, inHeight, inDepth ] = inputShape
	const [ outWidth, outHeight, outDepth ] = outputShape

	return /* wgsl */ `

    const IN_WIDTH: u32 = ${inWidth}u;
    const IN_HEIGHT: u32 = ${inHeight}u;
    const IN_DEPTH: u32 = ${inDepth}u;
    const IN_PLANE_SIZE: u32 = IN_HEIGHT * IN_WIDTH;

    const OUT_WIDTH: u32 = ${outWidth}u;
    const OUT_HEIGHT: u32 = ${outHeight}u;
    const OUT_DEPTH: u32 = ${outDepth}u;
    const OUT_PLANE_SIZE: u32 = OUT_HEIGHT * OUT_WIDTH;

    struct Params {
        scale_width: f32,
        scale_height: f32,
        scale_depth: f32,
        offset_factor: f32,
    };

    @group(0) @binding(0) var<storage, read> input_values: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    fn input_index(x: u32, y: u32, z: u32) -> u32
    {
        return z * IN_PLANE_SIZE + y * IN_WIDTH + x;
    }

    fn output_index(x: u32, y: u32, z: u32) -> u32
    {
        return z * OUT_PLANE_SIZE + y * OUT_WIDTH + x;
    }

    fn clamp_index(value: i32, size: u32) -> u32
    {
        return u32(clamp(value, 0, i32(size) - 1));
    }

    ${workgroupSizeWGSL( workgroupSize )}
    fn main(@builtin(global_invocation_id) gid: vec3<u32>)
    {
        let x = gid.x;
        let y = gid.y;
        let z = gid.z;

        if (x >= OUT_WIDTH || y >= OUT_HEIGHT || z >= OUT_DEPTH) { return; }

        let source_x = (f32(x) + params.offset_factor) * params.scale_width - params.offset_factor;
        let source_y = (f32(y) + params.offset_factor) * params.scale_height - params.offset_factor;
        let source_z = (f32(z) + params.offset_factor) * params.scale_depth - params.offset_factor;

        let floor_x_f = floor(source_x);
        let floor_y_f = floor(source_y);
        let floor_z_f = floor(source_z);

        let floor_x_i = i32(floor_x_f);
        let floor_y_i = i32(floor_y_f);
        let floor_z_i = i32(floor_z_f);

        let x0 = clamp_index(floor_x_i, IN_WIDTH);
        let y0 = clamp_index(floor_y_i, IN_HEIGHT);
        let z0 = clamp_index(floor_z_i, IN_DEPTH);

        let x1 = clamp_index(floor_x_i + 1, IN_WIDTH);
        let y1 = clamp_index(floor_y_i + 1, IN_HEIGHT);
        let z1 = clamp_index(floor_z_i + 1, IN_DEPTH);

        let fx = source_x - floor_x_f;
        let fy = source_y - floor_y_f;
        let fz = source_z - floor_z_f;

        let c000 = input_values[input_index(x0, y0, z0)];
        let c001 = input_values[input_index(x1, y0, z0)];
        let c010 = input_values[input_index(x0, y1, z0)];
        let c011 = input_values[input_index(x1, y1, z0)];

        let c100 = input_values[input_index(x0, y0, z1)];
        let c101 = input_values[input_index(x1, y0, z1)];
        let c110 = input_values[input_index(x0, y1, z1)];
        let c111 = input_values[input_index(x1, y1, z1)];

        let c00 = mix(c000, c001, fx);
        let c01 = mix(c010, c011, fx);
        let c10 = mix(c100, c101, fx);
        let c11 = mix(c110, c111, fx);

        let c0 = mix(c00, c01, fy);
        let c1 = mix(c10, c11, fy);

        output_values[output_index(x, y, z)] = mix(c0, c1, fz);
    }
    `
}