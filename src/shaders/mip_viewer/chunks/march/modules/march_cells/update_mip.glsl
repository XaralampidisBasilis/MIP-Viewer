


vec4 c = cubic.coeffs;
vec2 p = quadratic_roots(vec3(c.y, 2.0 * c.z, 3.0 * c.w));
p = clamp(p, 0.0, 1.0);

vec2 v_ext = eval_poly(c, p);

vec4 v = vec4(cubic.values[0], v_ext.x, v_ext.y, cubic.values[3]);
vec4 t = vec4(0.0, p.x, p.y, 1.0);

int i_max = argmax(v);
float v_max = v[i_max];
float t_max = p[i_max];

if (v_max > mip.value) 
{
    mip.value = v_max;
    mip.distance = mix(cell.entry_distance, cell.exit_distance, t_max);
}