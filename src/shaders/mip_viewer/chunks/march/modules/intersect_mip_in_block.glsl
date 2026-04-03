
// INTERSECT_MIP_IN_BLOCK
float block_value = sampleVolume(block.entry_position);
mip.intersected = shouldIntersectMip(mip.value, block_value);

if (mip.intersected) 
{
    mip.distance = block.entry_distance;
    mip.value = block_value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif
}
