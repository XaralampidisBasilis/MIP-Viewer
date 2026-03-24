
// COMPUTE DEBUG

// maximize
vec4 debug_cubic_maximize = to_color(cubic.maximize);

// max_value
vec4 debug_cubic_max_value = to_color(cubic.max_value);

// argmax_point
vec4 debug_cubic_argmax_point = to_color(cubic.argmax_point);

// coeffs
vec4 cubic_coeffs = abs(cubic.coeffs); 
vec4 debug_cubic_coeffs = to_color(
    cubic_coeffs[0] * hsv2rgb(vec3(0.0/4.0, 1.0, 1.0)) + // #FF0000
    cubic_coeffs[1] * hsv2rgb(vec3(1.0/4.0, 1.0, 1.0)) + // #80FF00
    cubic_coeffs[2] * hsv2rgb(vec3(2.0/4.0, 1.0, 1.0)) + // #00FFFF 
    cubic_coeffs[3] * hsv2rgb(vec3(3.0/4.0, 1.0, 1.0))   // #8000FF 
); 

// bernstein weights
vec4 cubic_bernstein_coeffs = abs(cubic.bernstein_coeffs); 
vec4 debug_cubic_bernstein_coeffs = to_color(
    cubic_bernstein_coeffs[0] * hsv2rgb(vec3(0.0/4.0, 1.0, 1.0)) + // #FF0000
    cubic_bernstein_coeffs[1] * hsv2rgb(vec3(1.0/4.0, 1.0, 1.0)) + // #80FF00 
    cubic_bernstein_coeffs[2] * hsv2rgb(vec3(2.0/4.0, 1.0, 1.0)) + // #00FFFF  
    cubic_bernstein_coeffs[3] * hsv2rgb(vec3(3.0/4.0, 1.0, 1.0))   // #8000FF 
); 

// PRINT DEBUG
switch (u_debug.option - 800)
{ 
    case 1: fragColor = debug_cubic_maximize;         break;
    case 2: fragColor = debug_cubic_coeffs;           break;
    case 3: fragColor = debug_cubic_bernstein_coeffs; break;
    case 4: fragColor = debug_cubic_argmax_point;         break;
    case 5: fragColor = debug_cubic_max_value;        break;
}
