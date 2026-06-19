import * as THREE from 'three/webgpu'
import { reference, screenCoordinate, storage, uniform, wgslFn } from 'three/tsl'
import fragmentWGSL from '../../../../shaders/mip_viewer/wgsl/fragment.wgsl?raw'

const emptyVolumeWords = new THREE.StorageBufferAttribute(new Uint32Array([0]), 1)
const emptyDistanceWords = new THREE.StorageBufferAttribute(new Uint32Array([0]), 1)
const volumeWordsCache = new WeakMap()
const distanceWordsCache = new WeakMap()

function toVolumeWords(texture)
{
    const source = texture?.image?.data
    if (!source) return emptyVolumeWords

    const cached = volumeWordsCache.get(texture)
    if (cached?.source === source) return cached.words

    const data = new Uint32Array(Math.ceil(source.length / 2))

    for (let i = 0; i < source.length; i += 2)
    {
        data[i >> 1] = source[i] | ((source[i + 1] ?? 0) << 16)
    }

    const words = new THREE.StorageBufferAttribute(data, 1)
    volumeWordsCache.set(texture, { source, words })

    return words
}

function getDistanceWordsPerVoxel(texture)
{
    if (texture?.internalFormat === 'R16UI') return 1
    if (texture?.internalFormat === 'RGBA16UI') return 4
    if (texture?.internalFormat === 'RGB32UI') return 3
    if (texture?.internalFormat === 'RGBA32UI') return 4
    return 1
}

function toDistanceWords(texture)
{
    const source = texture?.image?.data
    if (!source) return emptyDistanceWords

    const cached = distanceWordsCache.get(texture)
    if (cached?.source === source) return cached.words

    const data = source instanceof Uint32Array
        ? source
        : Uint32Array.from(source)

    const words = new THREE.StorageBufferAttribute(data, 1)
    distanceWordsCache.set(texture, { source, words })

    return words
}

export default function createWebGPUMaterial(uniforms, defines)
{
    const material = new THREE.NodeMaterial()
    material.side = THREE.BackSide
    material.blending = THREE.NoBlending
    material.depthTest = false
    material.depthWrite = false
    material.transparent = false

    material.isMIPWebGPUMaterial = true
    material.uniforms = uniforms
    material.defines = defines

    const volume = uniforms.u_volume.value
    const ray = uniforms.u_ray.value
    const box = uniforms.u_box.value
    const transform = uniforms.u_transform.value
    const shading = uniforms.u_shading.value
    const debug = uniforms.u_debug.value
    const textures = uniforms.u_textures.value

    const volumeWordsNode = storage(emptyVolumeWords, 'uint', emptyVolumeWords.count).toReadOnly()
    const distanceWordsNode = storage(emptyDistanceWords, 'uint', emptyDistanceWords.count).toReadOnly()
    const distanceWordsPerVoxelNode = uniform(1, 'uint')

    const mipFragment = wgslFn(fragmentWGSL)
    material.fragmentNode = mipFragment({
        volume_map: volumeWordsNode,
        distance_words: distanceWordsNode,

        frag_coord: screenCoordinate,
        resolution: reference('resolution', 'vec2', transform),
        inv_projection: reference('inv_projection', 'mat4', transform),
        inv_view: reference('inv_view', 'mat4', transform),
        inv_model: reference('inv_model', 'mat4', transform),

        volume_dimensions: reference('dimensions', 'ivec3', volume),
        volume_inv_dimensions: reference('inv_dimensions', 'vec3', volume),
        distance_dimensions: reference('blocked_dimensions', 'ivec3', volume),

        ray_direction: reference('direction', 'vec3', ray),
        ray_inv_direction: reference('inv_direction', 'vec3', ray),
        ray_sign_direction: reference('sign_direction', 'ivec3', ray),
        ray_step_distances: reference('step_distances', 'vec3', ray),
        ray_step_distance: reference('step_distance', 'float', ray),
        ray_dominant_axis: reference('dominant_axis', 'uint', ray),
        ray_quadrant_index: reference('quadrant_index', 'uint', ray),
        ray_group_index: reference('group_index', 'uint', ray),
        ray_reverse: reference('reverse', 'bool', ray),

        box_min_position: reference('min_position', 'vec3', box),
        box_max_position: reference('max_position', 'vec3', box),
        box_min_distance: reference('min_distance', 'float', box),
        box_max_distance: reference('max_distance', 'float', box),
        box_span_distance: reference('span_distance', 'float', box),

        shading_colormap: reference('colormap', 'int', shading),
        debug_option: reference('option', 'int', debug),

        distance_variation: reference('DISTANCE_VARIATION', 'int', defines),
        marching_method: reference('MARCHING_METHOD', 'int', defines),
        skipping_method: reference('SKIPPING_METHOD', 'int', defines),
        skipping_enabled: reference('SKIPPING_ENABLED', 'int', defines),
        block_size: reference('BLOCK_SIZE', 'int', defines),
        max_cells: reference('MAX_CELLS', 'int', defines),
        max_blocks: reference('MAX_BLOCKS', 'int', defines),
        max_cells_in_block: reference('MAX_CELLS_IN_BLOCK', 'int', defines),
        max_traces: reference('MAX_TRACES', 'int', defines),
        max_traces_in_cell: reference('MAX_TRACES_IN_CELL', 'int', defines),
        distance_words_per_voxel: distanceWordsPerVoxelNode,
    })

    material.syncWebGPUResources = () =>
    {
        const volumeWords = toVolumeWords(textures.volume_map)
        volumeWordsNode.value = volumeWords
        volumeWordsNode.bufferCount = volumeWords.count

        const distanceTexture = textures.distance_map
        const distanceWords = toDistanceWords(distanceTexture)
        distanceWordsNode.value = distanceWords
        distanceWordsNode.bufferCount = distanceWords.count
        distanceWordsPerVoxelNode.value = getDistanceWordsPerVoxel(distanceTexture)

        material.needsUpdate = true
    }

    return material
}
