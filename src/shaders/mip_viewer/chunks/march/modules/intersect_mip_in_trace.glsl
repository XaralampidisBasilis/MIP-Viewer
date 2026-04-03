
// INTERSECT_MIP_IN_TRACE
mip.intersected = shouldIntersectMip(mip.value, trace.value);

if (mip.intersected)
{
    mip.distance = trace.distance;
    mip.value = trace.value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif

}

