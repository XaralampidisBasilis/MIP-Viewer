
// START_MIP_IN_CUBIC
mip.update = cubic.values.w > mip.value;

if (mip.update)
{
    mip.distance = ray.start_distance;
    mip.value = cubic.values.w;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif

}
