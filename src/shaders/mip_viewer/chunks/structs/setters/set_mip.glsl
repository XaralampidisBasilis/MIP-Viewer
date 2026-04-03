#ifdef STRUCT_MIP

mip.update      = false;
mip.intersected = false;
mip.terminated  = false;
mip.position    = vec3(0.0);
mip.distance    = 0.0;
mip.value       = 0.0;
mip.gradient    = vec3(0.0);
mip.hessian     = mat3(0.0);
mip.normal      = vec3(0.0);
mip.curvatures  = vec2(0.0);

#endif // STRUCT_MIP
