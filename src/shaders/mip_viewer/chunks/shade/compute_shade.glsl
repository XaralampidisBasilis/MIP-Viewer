
// Compose colors
frag.color = colormap(u_shading.colormap, mip.value);

// float depth = map(u_box.min_distance, u_box.max_distance, mip.distance);
// if (u_ray.reverse) depth = 1.0 - depth;
// frag.color = mix(frag.color, vec3(1.0 - depth), u_debug.variable1);

fragColor = vec4(frag.color, 1.0);