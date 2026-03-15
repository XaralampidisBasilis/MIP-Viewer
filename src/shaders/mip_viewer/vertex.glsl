
out vec3 v_ray_origin;

flat out vec3  v_ray_direction;
flat out vec3  v_ray_inv_direction;
flat out float v_ray_spacing;
flat out ivec3 v_ray_signs;
flat out uint  v_ray_axis;
flat out uint  v_ray_idx;
flat out uint  v_ray_map;
flat out uint  v_ray_reverse;

#include "./chunks/uniforms/uniforms_volume"
#include "./chunks/utils/math/sum"
#include "./chunks/utils/math/ssign"
#include "./chunks/utils/math/argmax"

vec3 positionLocalToIndexSpace(vec3 positionLocal)
{
    return (positionLocal + 0.5) * vec3(u_volume.dimensions);
}

vec3 directionLocalToIndexSpace(vec3 directionLocal)
{
    return directionLocal * vec3(u_volume.dimensions);
}

vec3 distanceToPosition(vec3 rayOrigin, vec3 rayDirection, float t)
{
    return rayOrigin + rayDirection * t;
}

vec3 getCameraPositionLocalSpace()
{
    return (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
}

vec3 getCameraPositionIndexSpace()
{
    return positionLocalToIndexSpace(getCameraPositionLocalSpace());
}

vec3 getCameraDirectionLocalSpace()
{
    // In view space, camera looks down -Z.
    return (inverse(modelViewMatrix) * vec4(0.0, 0.0, -1.0, 0.0)).xyz;
}

vec3 getRayDirectionIndexSpace()
{
    return normalize(directionLocalToIndexSpace(getCameraDirectionLocalSpace()));
}

vec3 getRayOriginIndexSpace(vec3 vertexPosition, vec3 cameraPosition, vec3 cameraDirection) 
{
    // compute the intersection point with the camera orthographic plane 
    float rayDistance = dot(vertexPosition - cameraPosition, cameraDirection);
    return vertexPosition - cameraDirection * rayDistance;
}

vec2 getRayStartEndDistanceIndexSpace(vec3 rayOrigin, vec3 rayDirection) 
{
    vec3 boxMax = vec3(u_volume.dimensions);
    vec3 boxMin = vec3(0.0);

    vec3 t0 = (boxMin - rayOrigin) / rayDirection;
    vec3 t1 = (boxMax - rayOrigin) / rayDirection;

    vec3 tMin = min(t0, t1); 
    vec3 tMax = max(t0, t1);

    float tEntry = max(max(tMin.x, tMin.y), tMin.z);
    float tExit  = min(min(tMax.x, tMax.y), tMax.z);

    return vec2(tEntry, tExit);
}

// Returns the 2-bit quadrant index (0..3) after projecting the ray sign
// configuration onto the plane orthogonal to the dominant axis.
uint rayQuadrantIndex(ivec3 signs, uint axis)
{
    uvec3 b = uvec3(greaterThan(signs, ivec3(0)));

    uint xy = b.x ^ b.y;
    uint xz = b.x ^ b.z;
    uint yz = b.y ^ b.z;

    if (axis == 0u) return (xz << 1) | xy;
    if (axis == 1u) return (yz << 1) | xy;
    if (axis == 2u) return (xz << 1) | yz;

    return 0u;
}

void classifyRay(vec3 rayDirection)
{
    vec3  absDirection = abs(rayDirection);
    vec3  invDirection = 1.0 / rayDirection;
    float spacing = 1.0 / sum(absDirection);
    ivec3 signs = ivec3(ssign(rayDirection));

    uint axis = uint(argmax(absDirection));
    uint idx = rayQuadrantIndex(signs, axis);
    uint reverse = uint(signs[axis] < 0);
    uint map = idx + 4u * axis;

    v_ray_inv_direction = invDirection;
    v_ray_signs = signs;
    v_ray_spacing = spacing;
    v_ray_axis = axis;
    v_ray_idx = idx;
    v_ray_map = map;
    v_ray_reverse = reverse;
}

void main()
{
    vec3 vertexPosition = positionLocalToIndexSpace(position);
    vec3 cameraPosition = getCameraPositionIndexSpace();
    vec3 rayDirection = getRayDirectionIndexSpace();
    vec3 rayOrigin = getRayOriginIndexSpace(vertexPosition, cameraPosition, rayDirection);
    classifyRay(rayDirection);

    v_ray_origin = rayOrigin;
    v_ray_direction = rayDirection;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}