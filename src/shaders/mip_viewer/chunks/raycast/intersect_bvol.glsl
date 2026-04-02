

// INTERSECT_BVOL_START
#INCLUDE "./intersect_bvol_start"

// INTERSECT_BVOL_END
#include "./intersect_bvol_end"

// RAY DISCARD CONDITION
if (ray.start_distance > ray.end_distance)
{
    ray.discarded = true;
}




