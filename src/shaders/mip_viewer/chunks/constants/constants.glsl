
// Sampling points to compute the interpolation polynomials
const vec4 sampling_points = vec4(0, 1, 2, 3) / 3.0;

// Samples-to-Bernstein transformation matrix for a cubic polynomial sampled at points [0, 1/3, 2/3, 1]
// Transforms 4 samples of a quadratic function at those positions into polynomial coefficients
const mat4 cubic_inv_vander = mat4(
    2,   0,   0,  0,   
    -11,  18,  -9,  2,   
    18, -45,  36, -9, 
    -9,  27, -27,  9
) / 2.0;

// Samples-to-Bernstein transformation matrix for a cubic polynomial sampled at points [0, 1/3, 2/3, 1]
// Transforms 4 samples of a quadratic function at those positions into Bernstein coefficients
const mat4 cubic_bernstein = mat4(
    6, 0, 0, 0,  
    -5, 18, -9, 2, 
    2, -9, 18, -5,
    0, 0, 0, 6   
) / 6.0;

