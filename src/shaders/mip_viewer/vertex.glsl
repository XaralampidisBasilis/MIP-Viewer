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
    vec3 cameraDirectionView = vec3(0.0, 0.0, -1.0);
    return (inverse(modelViewMatrix) * vec4(cameraDirectionView, 0.0)).xyz;
}

vec3 getRayDirectionIndexSpace()
{
    return normalize(directionLocalToIndexSpace(getCameraDirectionLocalSpace()));
}

vec3 getRayOriginIndexSpace(vec3 vertexPosition, vec3 cameraPosition, vec3 rayDirection) 
{
    float t = dot(vertexPosition - cameraPosition, rayDirection);
    return vertexPosition - rayDirection * t;
}

vec2 intersectBox(
    vec3 boxMin,
    vec3 boxMax,
    vec3 rayOrigin,
    vec3 rayInvDirection
) {
    vec3 t0 = (boxMin - rayOrigin) * rayInvDirection;
    vec3 t1 = (boxMax - rayOrigin) * rayInvDirection;

    vec3 tMin3 = min(t0, t1); 
    vec3 tMax3 = max(t0, t1);

    float tEntry = max(max(tMin3.x, tMin3.y), tMin3.z);
    float tExit  = min(min(tMax3.x, tMax3.y), tMax3.z);

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
    vec3  abs_dir = abs(rayDirection);
    vec3  inv_direction = 1.0 / rayDirection;
    ivec3 signs = ivec3(ssign(rayDirection));
    float spacing = 1.0 / sum(abs_dir);

    uint axis = uint(argmax(abs_dir));
    uint idx = rayQuadrantIndex(signs, axis);
    uint map = idx + 4u * axis;
    uint reverse = uint(signs[axis] < 0);

    v_ray_direction     = direction;
    v_ray_inv_direction = inv_direction;
    v_ray_signs         = signs;
    v_ray_spacing       = spacing;
    v_ray_axis          = axis;
    v_ray_idx           = idx;
    v_ray_map           = map;
    v_ray_reverse       = reverse;
}

void main()
{
    vec3 positionIndex = positionLocalToIndexSpace(position);
    vec3 cameraIndex   = getCameraPositionIndexSpace();
    vec3 direction     = getRayDirectionIndexSpace();

    v_ray_origin = getRayOriginIndexSpace(positionIndex, cameraIndex, direction);

    classifyRay(direction);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}