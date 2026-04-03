

// INTERSECT_MIP_IN_CUBIC
mip.intersected = shouldIntersectMip(mip.value, cubic.max_value);

if (mip.intersected) 
{
    mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic.argmax_point);
    mip.value = cubic.max_value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif
}
