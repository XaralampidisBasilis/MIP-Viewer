// COMPUTE DEBUG 


// mapped color
vec4 debug_frag_color = to_color(frag.color);

// PRINT DEBUG
switch (u_debug.option - 500)
{
    case 16: fragColor = debug_frag_color;           break; 
}   