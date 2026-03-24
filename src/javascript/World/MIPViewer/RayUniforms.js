import * as THREE from 'three'

const _worldDirection = new THREE.Vector3()
const _localDirection = new THREE.Vector3()
const _indexDirection = new THREE.Vector3()
const _inverseModelMatrix = new THREE.Matrix4()

function argmax3(x, y, z) 
{
    return x >= y ? (x >= z ? 0 : 2) : (y >= z ? 1 : 2);
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
    return 
}

export function updateRayUniforms(uniforms, camera, mesh, dimensions)
{
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

    const ray = uniforms.u_ray.value

    ray.direction.copy(_indexDirection)

    ray.inv_direction.set(
        1 / _indexDirection.x,
        1 / _indexDirection.y,
        1 / _indexDirection.z,
    )

    ray.sign_direction.set(
        _indexDirection.x >= 0 ? 1 : -1,
        _indexDirection.y >= 0 ? 1 : -1,
        _indexDirection.z >= 0 ? 1 : -1,
    )

    ray.step_distances.set(
        Math.abs(1 / _indexDirection.x),
        Math.abs(1 / _indexDirection.y),
        Math.abs(1 / _indexDirection.z),
    )

    const absX = Math.abs(_indexDirection.x)
    const absY = Math.abs(_indexDirection.y)
    const absZ = Math.abs(_indexDirection.z)

    const dominantAxis = argmax3(absX, absY, absZ)
    const dominantSign = ray.sign_direction.getComponent(dominantAxis)
    
    ray.step_distance = 1 / (absX + absY + absZ)
    ray.axis = dominantAxis     
    ray.idx = rayQuadrantIndex(ray.sign_direction, dominantAxis)
    ray.map = ray.idx + 4 * dominantAxis
    ray.reverse = dominantSign < 0
    
    if (ray.reverse)
    {
        ray.sign_direction.negate()
        ray.direction.negate()
        ray.inv_direction.negate()
    }
}
