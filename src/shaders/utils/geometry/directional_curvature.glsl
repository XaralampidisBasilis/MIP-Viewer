
#ifndef DIRECTIONAL_CURVATURE
#define DIRECTIONAL_CURVATURE

float directional_curvature(in vec3 direction, in vec3 gradient, in mat3 hessian)
{
    vec3 normal = normalize(gradient);
    vec3 tangent = direction - normal * dot(normal, direction);
    if (length(tangent) < 0.001) return 0.0; 

    mat3 P = mat3(1.0) - outerProduct(normal, normal);
    mat3 S = (P * hessian * P) / length(gradient);
    float curvature = dot(tangent, S * tangent) / dot(tangent, tangent);

    return curvature;
}

#endif