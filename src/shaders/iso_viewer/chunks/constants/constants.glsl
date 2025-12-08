
// Sampling points to compute the interpolation polynomials
const vec4 sampling_points = vec4(0, 1, 2, 3) / 3.0;

// Inverse Vandermonde matrix for a quadratic polynomial sampled at points [1/3, 2/3, 1]
// Transforms 3 samples of a quadratic function at those positions into polynomial coefficients
const mat3 quad_inv_vander = mat3(
    6, -15, 9,    
   -6, 24, -18,   
    2, -9, 9      
) / 2.0;

// Samples-to-Bernstein transformation matrix for a quadratic polynomial sampled at points [1/3, 2/3, 1]
// Transforms 3 samples of a quadratic function at those positions into Bernstein coefficients
const mat3 quad_bernstein = mat3(
    12, -3, 0,    
   -12, 12, 0,    
    4, -5, 4      
) / 4.0;

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

// Precomputed coefficient correction matrix for mixed Bernstein basis multiplication
// This matrix transforms the product of cubic (degree 3) and quadratic (degree 2) Bernstein basis functions
// into the equivalent quintic (degree 5) Bernstein basis
//
// Specifically, it encodes the scalar coefficients:
//      B_n^3(t) * B_m^2(t) = [ binomial(3,n) * binomial(2,m) / binomial(5, n+m) ] * B_{n+m}^5(t)
//
// This 4x3 matrix corresponds to the coefficients of B_{n+m}^5 for all combinations of n in [0,3], m in [0,2]
const mat4x3 quintic_bernstein_weights = mat4x3(
    10, 4, 1,  
     6, 6, 3,  
     3, 6, 6,  
     1, 4,10   
) / 10.0;
