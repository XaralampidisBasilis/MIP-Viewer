// Compute normalized direction 
ray.direction = normalize(v_ray_direction);
ray.inv_direction = 1.0 / ray.direction;

// Compute directional mean cell spacing 
// For a specific ray direction, this result is the 
// mean span distance that a ray passes from a cell. 
ray.spacing = 1.0 / sum(abs(ray.direction));


// | ray.octant | bits (z y x) | direction sign (x,y,z) |
// | ---------: | :----------: | :--------------------: |
// |          0 |      000     |        (-, -, -)       |
// |          1 |      001     |        (+, -, -)       |
// |          2 |      010     |        (-, +, -)       |
// |          3 |      011     |        (+, +, -)       |
// |          4 |      100     |        (-, -, +)       |
// |          5 |      101     |        (+, -, +)       |
// |          6 |      110     |        (-, +, +)       |
// |          7 |      111     |        (+, +, +)       |

// Compute the octant sign of the direction 
ray.signs = ivec3(ssign(ray.direction));
ray.axis = argmax(abs(ray.direction));

ivec3 bits = ivec3(vec3(ray.signs) * 0.5 + 0.5); 
ray.octant = (bits.z << 2) | (bits.y << 1) | (bits.x << 0);

// |  ray.map   |   ray.axis   |        ray.octants        |
// | ---------: | :----------: | :-----------------------: |
// |          0 |      x       |        ---, +++           |
// |          1 |      x       |        -+-, +-+           |
// |          2 |      x       |        ++-, --+           |
// |          3 |      x       |        +--, -++           |
// |          4 |      y       |        ---, +++           |
// |          5 |      y       |        +--, -++           |
// |          6 |      y       |        ++-, --+           |
// |          7 |      y       |        -+-, +-+           |
// |          8 |      z       |        ---, +++           |
// |          9 |      z       |        -+-, +-+           |
// |         10 |      z       |        +--, -++           |
// |         11 |      z       |        ++-, --+           |

if (ray.axis == 0)
{
    if (ray.octant == 0) ray.map = 0 + 0; // 'x', '---' 
    if (ray.octant == 1) ray.map = 0 + 3; // 'x', '+--' 
    if (ray.octant == 2) ray.map = 0 + 1; // 'x', '-+-' 
    if (ray.octant == 3) ray.map = 0 + 2; // 'x', '++-' 
    if (ray.octant == 4) ray.map = 0 + 2; // 'x', '--+' 
    if (ray.octant == 5) ray.map = 0 + 1; // 'x', '+-+' 
    if (ray.octant == 6) ray.map = 0 + 3; // 'x', '-++' 
    if (ray.octant == 7) ray.map = 0 + 0; // 'x', '+++' 
}
if (ray.axis == 1)
{
    if (ray.octant == 0) ray.map = 4 + 0; // 'y', '---'
    if (ray.octant == 1) ray.map = 4 + 1; // 'y', '+--'
    if (ray.octant == 2) ray.map = 4 + 3; // 'y', '-+-'
    if (ray.octant == 3) ray.map = 4 + 2; // 'y', '++-'
    if (ray.octant == 4) ray.map = 4 + 2; // 'y', '--+'
    if (ray.octant == 5) ray.map = 4 + 3; // 'y', '+-+'
    if (ray.octant == 6) ray.map = 4 + 1; // 'y', '-++'
    if (ray.octant == 7) ray.map = 4 + 0; // 'y', '+++'
}
if (ray.axis == 2)
{
    if (ray.octant == 0) ray.map = 8 + 0; // 'z', '---'
    if (ray.octant == 1) ray.map = 8 + 2; // 'z', '+--'
    if (ray.octant == 2) ray.map = 8 + 1; // 'z', '-+-'
    if (ray.octant == 3) ray.map = 8 + 3; // 'z', '++-'
    if (ray.octant == 4) ray.map = 8 + 3; // 'z', '--+'
    if (ray.octant == 5) ray.map = 8 + 1; // 'z', '+-+'
    if (ray.octant == 6) ray.map = 8 + 2; // 'z', '-++'
    if (ray.octant == 7) ray.map = 8 + 0; // 'z', '+++'
}




