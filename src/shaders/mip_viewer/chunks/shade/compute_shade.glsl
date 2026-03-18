
// Compose colors

frag.color = colormap(u_shading.colormap, mip.value);
fragColor = vec4(frag.color, 1.0);