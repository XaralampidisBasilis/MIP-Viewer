
// Compose colors
frag.color = colormap(mip.value, u_shading.colormap);
fragColor = vec4(frag.color, 1.0);