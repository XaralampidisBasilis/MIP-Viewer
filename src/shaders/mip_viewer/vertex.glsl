
out vec3 v_position;
out vec3 v_camera_position;
out vec3 v_camera_direction;
flat out vec3 v_ray_direction;

uniform mat4 uCustomModelMatrix;

void main() 
{	
    const vec3 cameraDirection = vec3(0.0, 0.0, -1.0);
  
    // vertex position varying
    v_position = vec3(uCustomModelMatrix * vec4(position, 1.0));

    // camera direction
    v_camera_direction = vec3(uCustomModelMatrix * inverse(modelViewMatrix) * vec4(cameraDirection, 0.0));
    vec3 cameraPlanePosition = vec3(uCustomModelMatrix * inverse(modelMatrix) * vec4(cameraPosition, 1.0));

    // Orthographic rays start at the intersection between each ray line and the camera plane.
    v_ray_direction = normalize(v_camera_direction);
    float rayPlaneDistance = dot(v_position - cameraPlanePosition, v_ray_direction);
    v_camera_position = v_position - v_ray_direction * rayPlaneDistance;

    // Vertex position in physical space
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
