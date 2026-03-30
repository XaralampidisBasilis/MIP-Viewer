// COMPUTE DEBUG 

// terminated
vec4 debug_mip_terminated = to_color(mip.terminated);

// update
vec4 debug_mip_update = to_color(mip.update);

// distance
vec4 debug_mip_distance = to_color(map(u_box.min_distance, u_box.max_distance, mip.distance));

// position
vec4 debug_mip_position = to_color(map(u_box.min_position, u_box.max_position, mip.position));

// value
vec4 debug_mip_value = to_color(mip.value);

// normal
vec4 debug_mip_normal = to_color((map(-1.0, 1.0, mip.normal)));

// gradient
vec4 debug_mip_gradient = to_color(map(-1.0, 1.0, normalize(mip.gradient)) * length(mip.gradient));

// steepness
vec4 debug_mip_steepness = to_color(map(0.0, 1.0, length(mip.gradient)));

// curvatures
vec4 debug_mip_curvatures = to_color(mmix2(                
    COLOR.DARK_CYAN, COLOR.DARK_BLUE, COLOR.MAGENTA, // | < 0     | concave ellipsoid   | concave cylinder | hyperboloid Surface |                  
    COLOR.DARK_BLUE, COLOR.DARK_GRAY, COLOR.ORANGE,  // | = 0     | concave cylinder    | flap plane       | convex cylinder     |
    COLOR.MAGENTA,   COLOR.ORANGE,    COLOR.GOLD,    // | > 0     | hyperboloid Surface | convex cylinder  | convex ellipsoid    |
    map(-2.0, 2.0, mip.curvatures)                   // | k2 \ k1 | < 0                 | ~ 0              | > 0                 |
));                 


// PRINT DEBUG
switch (u_debug.option - 450)
{ 
    case 1: fragColor = debug_mip_terminated;      break;
    case 2: fragColor = debug_mip_update;          break;
    case 3: fragColor = debug_mip_distance;        break;
    case 4: fragColor = debug_mip_position;        break;
    case 5: fragColor = debug_mip_value;           break;
    case 6: fragColor = debug_mip_normal;          break;
    case 7: fragColor = debug_mip_gradient;        break;
    case 8: fragColor = debug_mip_steepness;       break;
    case 9: fragColor = debug_mip_curvatures;      break;
}