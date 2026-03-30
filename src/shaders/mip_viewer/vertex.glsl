
out vec3 v_ray_origin;
out vec3 v_vertex_position;

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

void main()
{

    vec3 vertexPosition = positionLocalToIndexSpace(position);
    vec3 cameraPosition = getCameraPositionIndexSpace();
    vec3 rayOrigin = getRayOriginIndexSpace(vertexPosition, cameraPosition, u_ray.direction);

    v_ray_origin = rayOrigin;
    v_vertex_position = vertexPosition;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
