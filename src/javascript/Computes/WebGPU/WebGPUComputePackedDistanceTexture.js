import * as THREE from 'three'
import { getWebGPUComputeContext } from '../../WebGPU/WebGPUDevice'
import { createStorageBuffer, readBuffer } from '../../WebGPU/WebGPUBufferUtils'
import { dispatchForShape, runComputeProgram } from './WebGPUComputeRunner'
import { WebGPUTensor3D } from './WebGPUTensor3D'
import { computeUnidirectionalDistanceMapWebGPU } from './WebGPUDistanceMap'
import { computeBidirectionalShadowMapFacesWebGPU } from './WebGPUShadowMapFaces'
import { computeBidirectionalShadowMapPathsWebGPU } from './WebGPUShadowMapPaths'

const PACK_WORKGROUP_SIZE = [ 8, 8, 4 ]

const DISTANCE_TARGETS = [
	{ axis: 'x', octant: '+++' },
	{ axis: 'x', octant: '+-+' },
	{ axis: 'x', octant: '++-' },
	{ axis: 'x', octant: '+--' },

	{ axis: 'y', octant: '+++' },
	{ axis: 'y', octant: '-++' },
	{ axis: 'y', octant: '++-' },
	{ axis: 'y', octant: '-+-' },

	{ axis: 'z', octant: '+++' },
	{ axis: 'z', octant: '+-+' },
	{ axis: 'z', octant: '-++' },
	{ axis: 'z', octant: '--+' },
]

const MAX_ENCODABLE_DISTANCES = {
	'1bit' : { x: 1,    y: 1,    z: 1    },
	'5bit' : { x: 31,   y: 31,   z: 63   },
	'8bit' : { x: 255,  y: 255,  z: 255  },
	'10bit': { x: 2047, y: 2047, z: 1023 },
}

// Keep compatible with the WebGPU render path. WebGPU has no rgb32uint format,
// so the 8bit layout uses rgba32uint with the fourth word unused.
const DISTANCE_TEXTURE_FORMATS = {
	'1bit' : { format: 'RED_INTEGER',  type: THREE.UnsignedShortType, internalFormat: 'R16UI'    },
	'5bit' : { format: 'RGBA_INTEGER', type: THREE.UnsignedShortType, internalFormat: 'RGBA16UI' },
	'8bit' : { format: 'RGBA_INTEGER', type: THREE.UnsignedIntType,   internalFormat: 'RGBA32UI' },
	'10bit': { format: 'RGBA_INTEGER', type: THREE.UnsignedIntType,   internalFormat: 'RGBA32UI' },
}

function maxEncodableDistance( encoding, axis ) {

	return MAX_ENCODABLE_DISTANCES[ encoding ][ axis ]
}

function computeTextureDimensionsFromShape( shape, blockSize ) {

	const blockShape = shape.map( ( x ) => Math.ceil( ( x + 1 ) / blockSize ) )
	const [ width, height, depth ] = blockShape

	return new THREE.Vector3( width, height, depth )
}

function packedWordsPerVoxel( encoding ) {

	if ( encoding === '1bit' ) return 1
	if ( encoding === '5bit' ) return 2
	if ( encoding === '8bit' ) return 4
	if ( encoding === '10bit' ) return 4

	throw new Error( `Unsupported distance texture encoding "${encoding}".` )
}

function distanceTextureWordsPerVoxel( encoding ) {

	if ( encoding === '1bit' ) return 1
	if ( encoding === '5bit' ) return 4
	if ( encoding === '8bit' ) return 4
	if ( encoding === '10bit' ) return 4

	throw new Error( `Unsupported distance texture encoding "${encoding}".` )
}

function packedWordCount( encoding, voxelCount ) {

	return voxelCount * packedWordsPerVoxel( encoding )
}

async function ensureWebGPUTensor( volume, device ) {

	if ( volume?.buffer && volume?.device && volume?.shape ) {
		return { tensor: volume, owned: false }
	}

	if ( ! volume?.shape || typeof volume.data !== 'function' ) {
		throw new Error( 'computePackedDistanceTextureWebGPU expected a WebGPUTensor3D or tf.Tensor3D-like volume.' )
	}

	const volumeData = await volume.data()
	const data = volumeData instanceof Float32Array ? volumeData : new Float32Array( volumeData )

	const tensor = WebGPUTensor3D.fromTypedArray(
		device,
		volume.shape.toReversed(),
		data,
		'float32',
		'webgpu-distance-volume',
	)

	return { tensor, owned: true }
}

function getShadowMapComputer( shadowBackend ) {
	if ( shadowBackend === 'faces' ) return computeBidirectionalShadowMapFacesWebGPU
	if ( shadowBackend === 'paths' ) return computeBidirectionalShadowMapPathsWebGPU
	throw new Error( `Unsupported WebGPU shadow backend "${shadowBackend}".` )
}

async function computeDistanceMapTensor( volume, variant, axis, octant, tolerance, blockSize, maxDistance, verbose, shadowBackend ) {

	if ( variant !== 'unidirectional' ) {
		throw new Error( `WebGPU distance texture currently supports the active "unidirectional" variant, got "${variant}".` )
	}

	if ( verbose ) {
		console.time( `webgpu-distance-${axis}-${octant}` )
	}

	const computeShadowMap = getShadowMapComputer( shadowBackend )
	const shadowMap = await computeShadowMap(
		volume,
		axis,
		octant,
		tolerance,
		blockSize,
		verbose,
	)

	let distanceMap = null

	try {

		distanceMap = await computeUnidirectionalDistanceMapWebGPU(
			shadowMap,
			axis,
			octant,
			maxDistance,
			verbose,
		)

		return distanceMap

	} catch ( error ) {

		distanceMap?.dispose()
		throw error

	} finally {

		shadowMap.dispose()

		if ( verbose ) {
			console.timeEnd( `webgpu-distance-${axis}-${octant}` )
		}
	}
}

async function packDistanceMapIntoBuffer( distanceMap, packedBuffer, encoding, targetIndex ) {

	await runComputeProgram( distanceMap.device, {
		label: `pack-distance-map-${encoding}-${targetIndex}`,
		code: packDistanceMapWGSL( distanceMap.shape, encoding, targetIndex ),
		bindings: [
			{ buffer: distanceMap.buffer },
			{ buffer: packedBuffer },
		],
		dispatch: dispatchForShape( distanceMap.shape, PACK_WORKGROUP_SIZE ),
	} )
}

export async function computePackedDistanceBufferWebGPU(
	volume,
	variant,
	tolerance,
	blockSize,
	encoding,
	verbose = false,
	options = {},
) {

	const { device } = await getWebGPUComputeContext()
	const { tensor, owned } = await ensureWebGPUTensor( volume, device )
	const dimensions = computeTextureDimensionsFromShape( tensor.shape, blockSize )
	const textureFormat = DISTANCE_TEXTURE_FORMATS[ encoding ]
	const shadowBackend = options.shadowBackend ?? 'paths'

	if ( ! textureFormat ) {
		throw new Error( `Unsupported distance texture encoding "${encoding}".` )
	}

	const voxelCount = dimensions.x * dimensions.y * dimensions.z
	const wordCount = packedWordCount( encoding, voxelCount )
	const packedBuffer = createStorageBuffer(
		device,
		wordCount * Uint32Array.BYTES_PER_ELEMENT,
		`packed-distance-buffer-${encoding}`,
	)

	if ( verbose ) {
		console.time( 'computePackedDistanceBufferWebGPU' )
	}

	try {

		for ( let targetIndex = 0; targetIndex < DISTANCE_TARGETS.length; targetIndex += 1 ) {

			const { axis, octant } = DISTANCE_TARGETS[ targetIndex ]
			const maxDistance = maxEncodableDistance( encoding, axis )

			const distanceMap = await computeDistanceMapTensor(
				tensor,
				variant,
				axis,
				octant,
				tolerance,
				blockSize,
				maxDistance,
				verbose,
				shadowBackend,
			)

			try {

				await packDistanceMapIntoBuffer(
					distanceMap,
					packedBuffer,
					encoding,
					targetIndex,
				)

			} finally {

				distanceMap.dispose()
			}
		}

		return {
			device,
			packedBuffer,
			wordCount,
			dimensions,
			encoding,
			textureFormat,
			dispose() {
				packedBuffer.destroy()
			},
		}

	} catch ( error ) {

		packedBuffer.destroy()
		throw error

	} finally {

		if ( owned ) {
			tensor.dispose()
		}

		if ( verbose ) {
			console.timeEnd( 'computePackedDistanceBufferWebGPU' )
		}
	}
}

export async function computePackedDistanceTextureWebGPU(
	volume,
	variant,
	tolerance,
	blockSize,
	encoding,
	verbose = false,
	options = {},
) {

	const textureFormat = DISTANCE_TEXTURE_FORMATS[ encoding ]

	if ( ! textureFormat ) {
		throw new Error( `Unsupported distance texture encoding "${encoding}".` )
	}

	if ( verbose ) {
		console.time( 'computePackedDistanceTextureWebGPU' )
	}

	const packed = await computePackedDistanceBufferWebGPU(
		volume,
		variant,
		tolerance,
		blockSize,
		encoding,
		verbose,
		options,
	)

	try {

		const voxelCount = packed.dimensions.x * packed.dimensions.y * packed.dimensions.z
		const data = await readPackedDistanceData(
			packed.device,
			packed.packedBuffer,
			encoding,
			voxelCount,
		)

		const texture = new THREE.Data3DTexture(
			data,
			packed.dimensions.x,
			packed.dimensions.y,
			packed.dimensions.z,
		)

		texture.format = textureFormat.format
		texture.type = textureFormat.type
		texture.internalFormat = textureFormat.internalFormat
		texture.minFilter = THREE.NearestFilter
		texture.magFilter = THREE.NearestFilter
		texture.generateMipmaps = false
		texture.unpackAlignment = 1
		texture.userData.distanceEncoding = encoding
		texture.userData.distanceWordsPerVoxel = distanceTextureWordsPerVoxel( encoding )
		texture.needsUpdate = true

		return {
			data,
			texture,
			dimensions: packed.dimensions,
			encoding,
		}

	} finally {

		packed.dispose()

		if ( verbose ) {
			console.timeEnd( 'computePackedDistanceTextureWebGPU' )
		}
	}
}

async function readPackedDistanceData( device, packedBuffer, encoding, voxelCount ) {

	const wordCount = packedWordCount( encoding, voxelCount )
	const packedWords = await readBuffer(
		device,
		packedBuffer,
		wordCount * Uint32Array.BYTES_PER_ELEMENT,
		Uint32Array,
	)

	if ( encoding === '1bit' ) {

		const data = new Uint16Array( voxelCount )

		for ( let i = 0; i < voxelCount; i += 1 ) {
			data[ i ] = packedWords[ i ] & 0xffff
		}

		return data
	}

	if ( encoding === '5bit' ) {
		return new Uint16Array(
			packedWords.buffer,
			packedWords.byteOffset,
			voxelCount * 4,
		)
	}

	if ( encoding === '8bit' ) {
		return packedWords
	}

	if ( encoding === '10bit' ) {
		return packedWords
	}

	throw new Error( `Unsupported distance texture encoding "${encoding}".` )
}

function packDistanceMapWGSL( shape, encoding, targetIndex ) {

	const [ width, height, depth ] = shape
	const targetGroup = Math.floor( targetIndex / 4 )
	const targetComponent = targetIndex % 4

	return /* wgsl */ `
	const DEPTH: u32 = ${depth}u;
	const HEIGHT: u32 = ${height}u;
	const WIDTH: u32 = ${width}u;
	const TARGET_INDEX: u32 = ${targetIndex}u;
	const TARGET_GROUP: u32 = ${targetGroup}u;
	const TARGET_COMPONENT: u32 = ${targetComponent}u;

	@group(0) @binding(0) var<storage, read> distance_values: array<u32>;
	@group(0) @binding(1) var<storage, read_write> packed_words: array<u32>;

	fn index3(gid: vec3<u32>) -> u32
	{
	    return gid.z * HEIGHT * WIDTH + gid.y * WIDTH + gid.x;
	}

	fn write_or_set(word_index: u32, value: u32, initialize: bool)
	{
	    if (initialize) {
	        packed_words[word_index] = value;
	        return;
	    }

	    packed_words[word_index] = packed_words[word_index] | value;
	}

	fn pack_1bit(index: u32, distance: u32)
	{
	    let packed = (distance & 0x1u) << TARGET_INDEX;
	    write_or_set(index, packed, TARGET_INDEX == 0u);
	}

	fn pack_5bit(index: u32, distance: u32)
	{
	    let mask = select(0x1fu, 0x3fu, TARGET_GROUP == 2u);
	    let bit_shift = select(select(0u, 5u, TARGET_GROUP == 1u), 10u, TARGET_GROUP == 2u);
	    let channel_value = (distance & mask) << bit_shift;

	    let word_index = index * 2u + TARGET_COMPONENT / 2u;
	    let half_shift = (TARGET_COMPONENT & 1u) * 16u;
	    let packed = channel_value << half_shift;
	    let initialize = TARGET_GROUP == 0u && (TARGET_COMPONENT == 0u || TARGET_COMPONENT == 2u);

	    write_or_set(word_index, packed, initialize);
	}

	fn pack_8bit(index: u32, distance: u32)
	{
	    let word_index = index * 4u + TARGET_GROUP;
	    let bit_shift = TARGET_COMPONENT * 8u;
	    let packed = (distance & 0xffu) << bit_shift;
	    let initialize = TARGET_COMPONENT == 0u;

	    write_or_set(word_index, packed, initialize);

	    if (TARGET_INDEX == 0u) {
	        packed_words[index * 4u + 3u] = 0u;
	    }
	}

	fn pack_10bit(index: u32, distance: u32)
	{
	    let mask = select(0x7ffu, 0x3ffu, TARGET_GROUP == 2u);
	    let bit_shift = select(select(0u, 11u, TARGET_GROUP == 1u), 22u, TARGET_GROUP == 2u);
	    let word_index = index * 4u + TARGET_COMPONENT;
	    let packed = (distance & mask) << bit_shift;
	    let initialize = TARGET_GROUP == 0u;

	    write_or_set(word_index, packed, initialize);
	}

	@compute @workgroup_size(${PACK_WORKGROUP_SIZE[ 0 ]}, ${PACK_WORKGROUP_SIZE[ 1 ]}, ${PACK_WORKGROUP_SIZE[ 2 ]})
	fn main(@builtin(global_invocation_id) gid: vec3<u32>)
	{
	    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) {
	        return;
	    }

	    let index = index3(gid);
	    let distance = distance_values[index];

	    ${packCallWGSL( encoding )}
	}
`
}

function packCallWGSL( encoding ) {

	if ( encoding === '1bit' ) return 'pack_1bit(index, distance);'
	if ( encoding === '5bit' ) return 'pack_5bit(index, distance);'
	if ( encoding === '8bit' ) return 'pack_8bit(index, distance);'
	if ( encoding === '10bit' ) return 'pack_10bit(index, distance);'

	throw new Error( `Unsupported distance texture encoding "${encoding}".` )
}
