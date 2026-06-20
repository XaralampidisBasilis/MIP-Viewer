import * as THREE from 'three/webgpu'
import { reference, sampler, screenCoordinate, storage, texture3D, uniform, wgslFn } from 'three/tsl'
import fragmentWGSL from '../../../../shaders/mip_viewer/wgsl/fragment/source'

const emptyDistanceWords = new THREE.StorageBufferAttribute(new Uint32Array([0]), 1)
const distanceWordsCache = new WeakMap()

class AddressOfNode extends THREE.Node
{
    constructor(node, nodeType)
    {
        super(nodeType)
        this.node = node
    }

    generate(builder)
    {
        return `&${this.node.build(builder)}`
    }
}

function addressOf(node, nodeType)
{
    return new AddressOfNode(node, nodeType)
}

function createEmptyVolumeTexture()
{
    const texture = new THREE.Data3DTexture(new Uint16Array([0]), 1, 1, 1)
    texture.format = THREE.RedFormat
    texture.type = THREE.HalfFloatType
    texture.minFilter = THREE.LinearFilter
    texture.magFilter = THREE.LinearFilter
    texture.generateMipmaps = false
    texture.unpackAlignment = 2
    texture.needsUpdate = true

    return texture
}

function getDistanceWordsPerVoxel(texture)
{
    if (texture?.internalFormat === 'R16UI') return 1
    if (texture?.internalFormat === 'RGBA16UI') return 4
    // Compatibility for legacy/WebGL-generated 8bit maps. WebGPU compute emits RGBA32UI.
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
    material.side = THREE.DoubleSide
    material.blending = THREE.NoBlending
    material.depthTest = false
    material.depthWrite = false
    material.transparent = false
    material.toneMapped = false

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

    const volumeTextureNode = texture3D(createEmptyVolumeTexture())
    const distanceWordsNode = storage(emptyDistanceWords, 'uint').toReadOnly()
    const distanceWordsPointerNode = addressOf(distanceWordsNode, 'ptr<storage, array<u32>, read>')
    const distanceWordsPerVoxelNode = uniform(1, 'uint')

    const mipFragment = wgslFn(fragmentWGSL)
    material.fragmentNode = mipFragment({
        volume_map: volumeTextureNode,
        volume_sampler: sampler(volumeTextureNode),
        distance_words: distanceWordsPointerNode,

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
        ray_reverse: reference('reverse', 'int', ray),

        box_min_position: reference('min_position', 'vec3', box),
        box_max_position: reference('max_position', 'vec3', box),
        box_min_distance: reference('min_distance', 'float', box),
        box_max_distance: reference('max_distance', 'float', box),
        box_span_distance: reference('span_distance', 'float', box),

        shading_colormap: reference('colormap', 'int', shading),
        debug_option: reference('option', 'int', debug),
        debug_enabled: reference('DEBUG_ENABLED', 'int', defines),

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
        volumeTextureNode.value = textures.volume_map ?? volumeTextureNode.value

        const distanceTexture = textures.distance_map
        const distanceWords = toDistanceWords(distanceTexture)
        distanceWordsNode.value = distanceWords
        distanceWordsPerVoxelNode.value = getDistanceWordsPerVoxel(distanceTexture)

        material.needsUpdate = true
    }

    return material
}
