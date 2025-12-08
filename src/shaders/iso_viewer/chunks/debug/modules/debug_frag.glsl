// COMPUTE DEBUG 


// mapped color
vec4 debug_frag_color_material = to_color(frag.color_material);

// ambient color
vec4 debug_frag_color_ambient = to_color(frag.color_ambient);

// diffuse color
vec4 debug_frag_color_diffuse = to_color(frag.color_diffuse);

// specular color
vec4 debug_frag_color_specular = to_color(frag.color_specular);

// direct color
vec4 debug_frag_color_directional = to_color(frag.color_directional);

// shaded color
vec4 debug_frag_color = to_color(frag.color);

// shaded luminance
vec4 debug_frag_luminance = to_color(dot(frag.color, vec3(0.2126, 0.7152, 0.0722)));

// PRINT DEBUG
switch (u_debug.option - 500)
{
   
    case 11: fragColor = debug_frag_color_material;     break; 
    case 12: fragColor = debug_frag_color_ambient;      break; 
    case 13: fragColor = debug_frag_color_diffuse;      break; 
    case 14: fragColor = debug_frag_color_specular;     break; 
    case 15: fragColor = debug_frag_color_directional;  break; 
    case 16: fragColor = debug_frag_color;              break; 
    case 17: fragColor = debug_frag_luminance;          break; 
}   