
out vec3 v_ray_origin;

#include "./chunks/uniforms/uniforms_volume"
#include "./chunks/uniforms/uniforms_ray"

vec3 positionLocalToIndexSpace(vec3 positionLocal)
{
    return (positionLocal + 0.5) * vec3(u_volume.dimensions);
}

vec3 getCameraPositionLocalSpace()
{
    return (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
}

vec3 getCameraPositionIndexSpace()
{
    return positionLocalToIndexSpace(getCameraPositionLocalSpace());
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
    
    v_ray_origin = getRayOriginIndexSpace(vertexPosition, cameraPosition, u_ray.direction);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
