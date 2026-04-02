
// UPDATE_MIP_IN_RAY
float ray_value = sampleVolume(ray.start_position);
mip.update = shouldUpdateMip(mip.value, ray_value);

if (mip.update) 
{
    mip.distance = ray.start_distance;
    mip.value = ray_value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif
}
