import * as THREE from 'three'

const _worldDirection = new THREE.Vector3()
const _localDirection = new THREE.Vector3()
const _indexDirection = new THREE.Vector3()
const _inverseModelMatrix = new THREE.Matrix4()

function argmax3(x, y, z)
{
    let i = x < y ? 1 : 0
    const maxXY = i === 0 ? x : y

    if (maxXY < z)
    {
        i = 2
    }

    return i
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
    return (xz << 1) | yz
}

export function updateRayUniforms(uniforms, camera, mesh, dimensions)
{
    if (!uniforms?.u_ray || !camera || !mesh || !dimensions)
    {
        return
    }

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

    const absX = Math.abs(_indexDirection.x)
    const absY = Math.abs(_indexDirection.y)
    const absZ = Math.abs(_indexDirection.z)
    const axis = argmax3(absX, absY, absZ)

    const ray = uniforms.u_ray.value
    
    ray.direction.copy(_indexDirection)
    ray.inv_direction.set(
        1 / _indexDirection.x,
        1 / _indexDirection.y,
        1 / _indexDirection.z,
    )
    ray.spacing = 1 / (absX + absY + absZ)
    ray.signs.set(
        _indexDirection.x >= 0 ? 1 : -1,
        _indexDirection.y >= 0 ? 1 : -1,
        _indexDirection.z >= 0 ? 1 : -1,
    )
    ray.axis = axis
    ray.idx = rayQuadrantIndex(ray.signs, axis)
    ray.map = ray.idx + 4 * axis
    ray.reverse = Number(ray.signs.getComponent(axis) < 0)
}
