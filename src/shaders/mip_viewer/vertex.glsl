
out vec3 v_ray_origin;

flat out float v_box_min_distance;
flat out float v_box_max_distance;

#include "./chunks/uniforms/uniforms_volume"
#include "./chunks/uniforms/uniforms_ray"

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
    return (inverse(modelViewMatrix) * vec4(0.0, 0.0, -1.0, 0.0)).xyz;
}

vec3 geCameraDirectionIndexSpace()
{
    return normalize(directionLocalToIndexSpace(getCameraDirectionLocalSpace()));
}

vec3 getRayOriginIndexSpace(vec3 vertexPosition, vec3 cameraPosition, vec3 cameraDirection) 
{
    // compute the intersection point with the camera orthographic plane 
    float rayDistance = dot(vertexPosition - cameraPosition, cameraDirection);
    return vertexPosition - cameraDirection * rayDistance;
}

// Returns the minimum and maximum signed distances from an AABB to a plane.
//
// The plane is defined by:
// - planeOrigin: any point lying on the plane
// - planeNormal: the plane normal
//
// result.x = minimum signed distance over the box
// result.y = maximum signed distance over the box
//
// If result.x > 0.0, the whole box is in front of the plane.
// If result.y < 0.0, the whole box is behind the plane.
// If result.x <= 0.0 && result.y >= 0.0, the plane intersects the box.
//
// Note: distances are only true Euclidean distances if planeNormal is normalized.
vec2 getBoxPlaneMinMaxDistance(vec3 boxMin, vec3 boxMax, vec3 planeOrigin, vec3 planeNormal) 
{
    // AABB center and half-size
    vec3 boxCenter = 0.5 * (boxMin + boxMax);
    vec3 boxHalfExtent = 0.5 * (boxMax - boxMin);

    // Signed distance from the box center to the plane
    float centerDist = dot(planeNormal, boxCenter - planeOrigin);

    // Maximum variation of that distance over the box
    float radius = dot(abs(planeNormal), boxHalfExtent);

    return vec2(centerDist - radius, centerDist + radius);
}

void main()
{
    vec3 boxMin = vec3(0.0);
    vec3 boxMax = vec3(u_volume.dimensions);

    vec3 vertexPosition = positionLocalToIndexSpace(position);
    vec3 cameraPosition = getCameraPositionIndexSpace();
    vec3 rayOrigin = getRayOriginIndexSpace(vertexPosition, cameraPosition, u_ray.direction);

    v_ray_origin = rayOrigin;

    vec2 boxMinMaxDistance = getBoxPlaneMinMaxDistance(boxMin, boxMax, cameraPosition, u_ray.direction);

    v_box_min_distance = boxMinMaxDistance.x;
    v_box_max_distance = boxMinMaxDistance.y;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
