#ifndef GET_RAY_ORIGIN
#define GET_RAY_ORIGIN

vec3 getRayOrigin()
{
    // Convert the current fragment position from pixel coordinates
    // to normalized screen coordinates in the range [0, 1].
    // The +0.5 shifts from pixel corner to pixel center.
    vec2 uv = (gl_FragCoord.xy + vec2(0.5)) / u_transform.resolution;

    // Convert from [0, 1] screen coordinates to normalized device coordinates
    // in the range [-1, 1].
    vec2 ndc = uv * 2.0 - 1.0;

    // Build a clip-space position on the near plane (z = -1 in OpenGL NDC).
    // This gives the near-plane point corresponding to the current fragment.
    vec4 clipPosition = vec4(ndc, -1.0, 1.0);

    // Unproject from clip space back into view space.
    // For perspective projection, this is usually followed by division by w.
    vec4 viewPosition = u_transform.inv_projection * clipPosition;

    // Transform from view space into world space.
    vec4 worldPosition = u_transform.inv_view * vec4(viewPosition.xyz, 1.0);

    // Transform from world space into the volume's local/object space.
    vec4 localPosition = u_transform.inv_model * worldPosition;

    // Convert from local volume coordinates [-0.5, 0.5] to index coordinates
    // [0, dimensions], so the result is expressed in voxel index space.
    vec3 indexPosition = (localPosition.xyz + 0.5) * vec3(u_volume.dimensions);

    return indexPosition;
}

#endif // GET_RAY_ORIGIN