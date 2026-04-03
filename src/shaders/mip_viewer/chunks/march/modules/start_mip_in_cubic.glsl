
// START_MIP_IN_CUBIC
mip.update = shouldUpdateMip(mip.value, cubic.max_value);

if (mip.update)
{
    mip.distance = ray.start_distance;
    mip.value = cubic.max_value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif

}
