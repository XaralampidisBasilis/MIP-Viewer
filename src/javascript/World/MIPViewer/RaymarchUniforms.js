import * as THREE from 'three'

const _worldDirection = new THREE.Vector3()
const _localDirection = new THREE.Vector3()
const _indexDirection = new THREE.Vector3()
const _inverseModelMatrix = new THREE.Matrix4()
const _inverseProjectionMatrix = new THREE.Matrix4()
const _inverseViewMatrix = new THREE.Matrix4()
const _cameraPositionLocal = new THREE.Vector3()
const _cameraPositionIndex = new THREE.Vector3()
const _drawingBufferSize = new THREE.Vector2()
const _boxCenter = new THREE.Vector3()
const _boxHalfExtent = new THREE.Vector3()
const _boxCenterToPlaneOrigin = new THREE.Vector3()

function argmax3(x, y, z) 
{
    return x >= y ? (x >= z ? 0 : 2) : (y >= z ? 1 : 2)
}

function rayQuadrantIndex(signs, axis)
{
    const bx = Number(signs.x > 0)
    const by = Number(signs.y > 0)
    const bz = Number(signs.z > 0)

    const xy = bx ^ by
    const xz = bx ^ bz
    const yz = by ^ bz

    if (axis === 0) return (xz << 1) | xy
    if (axis === 1) return (yz << 1) | xy
    if (axis === 2) return (xz << 1) | yz
    return 0
}

function getSafeInverse(x)
{
    return Math.abs(x) > 1e-8 ? 1 / x : 1e8
}

function getBoxPlaneMinMaxDistance(boxMin, boxMax, planeOrigin, planeNormal)
{
    _boxCenter.addVectors(boxMin, boxMax).multiplyScalar(0.5)
    _boxHalfExtent.subVectors(boxMax, boxMin).multiplyScalar(0.5)
    _boxCenterToPlaneOrigin.subVectors(_boxCenter, planeOrigin)

    const centerDist = planeNormal.dot(_boxCenterToPlaneOrigin)
    const radius =
        Math.abs(planeNormal.x) * _boxHalfExtent.x +
        Math.abs(planeNormal.y) * _boxHalfExtent.y +
        Math.abs(planeNormal.z) * _boxHalfExtent.z

    return {
        min: centerDist - radius,
        max: centerDist + radius,
    }
}

function updateRayUniforms(uniforms)
{
    const ray = uniforms.u_ray.value

    ray.direction.copy(_indexDirection)

    ray.inv_direction.set(
        getSafeInverse(_indexDirection.x),
        getSafeInverse(_indexDirection.y),
        getSafeInverse(_indexDirection.z),
    )

    ray.sign_direction.set(
        _indexDirection.x >= 0 ? 1 : -1,
        _indexDirection.y >= 0 ? 1 : -1,
        _indexDirection.z >= 0 ? 1 : -1,
    )

    const absX = Math.abs(_indexDirection.x)
    const absY = Math.abs(_indexDirection.y)
    const absZ = Math.abs(_indexDirection.z)

    ray.step_distances.set(
        getSafeInverse(absX),
        getSafeInverse(absY),
        getSafeInverse(absZ),
    )

    ray.step_distance = 1 / Math.max(absX + absY + absZ, 1e-8)

    const dominantAxis = argmax3(absX, absY, absZ)
    const dominantSign = ray.sign_direction.getComponent(dominantAxis)

    ray.dominant_axis = dominantAxis
    ray.quadrant_index = rayQuadrantIndex(ray.sign_direction, dominantAxis)
    ray.group_index = ray.quadrant_index + 4 * dominantAxis
    ray.reverse = dominantSign < 0
}

function updateTransformUniforms(uniforms, camera, renderer)
{
    const transform = uniforms.u_transform.value

    renderer.getDrawingBufferSize(_drawingBufferSize)
    transform.resolution.copy(_drawingBufferSize)

    _inverseProjectionMatrix.copy(camera.projectionMatrixInverse)
    _inverseViewMatrix.copy(camera.matrixWorld)

    transform.inv_projection.copy(_inverseProjectionMatrix)
    transform.inv_view.copy(_inverseViewMatrix)
    transform.inv_model.copy(_inverseModelMatrix)
}

function updateBoxUniforms(uniforms, dimensions)
{
    const ray = uniforms.u_ray.value
    const box = uniforms.u_box.value

    _cameraPositionIndex.set(
        (_cameraPositionLocal.x + 0.5) * dimensions.x,
        (_cameraPositionLocal.y + 0.5) * dimensions.y,
        (_cameraPositionLocal.z + 0.5) * dimensions.z,
    )

    box.min_position.set(0, 0, 0)
    box.max_position.copy(dimensions)

    const boxMinMaxDistance = getBoxPlaneMinMaxDistance(
        box.min_position,
        box.max_position,
        _cameraPositionIndex,
        ray.direction
    )

    box.min_distance = boxMinMaxDistance.min
    box.max_distance = boxMinMaxDistance.max
    box.span_distance = box.max_distance - box.min_distance
}

export function updateRaymarchUniforms(viewer)
{
    const uniforms = viewer.material.uniforms
    const camera = viewer.camera.instance
    const mesh = viewer.mesh
    const dimensions = viewer.computes.volumeMap.dimensions
    const renderer = viewer.renderer.instance

    camera.updateWorldMatrix(true, false)
    mesh.updateWorldMatrix(true, false)

    camera.getWorldDirection(_worldDirection)

    _inverseModelMatrix.copy(mesh.matrixWorld).invert()
    _localDirection.copy(_worldDirection).transformDirection(_inverseModelMatrix)

    _indexDirection.set(
        _localDirection.x * dimensions.x,
        _localDirection.y * dimensions.y,
        _localDirection.z * dimensions.z,
    ).normalize()

    _cameraPositionLocal.copy(camera.position).applyMatrix4(_inverseModelMatrix)

    updateRayUniforms(uniforms)
    updateTransformUniforms(uniforms, camera, renderer)
    updateBoxUniforms(uniforms, dimensions)
}