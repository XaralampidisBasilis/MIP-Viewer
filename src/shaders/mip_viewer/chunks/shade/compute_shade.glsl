
// Compose colors
frag.color = colormap(u_shading.colormap, mip.value);

// float depth = map(u_box.min_distance, u_box.max_distance, mip.distance);
// frag.color = mix(frag.color, 2.0 * vec3(1.0 - depth), u_debug.variable2);

fragColor = vec4(frag.color, 1.0);