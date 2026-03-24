

// UPDATE_MIP_IN_CUBIC
if (mip.update) 
{
    mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic.argmax_t);
    mip.value = cubic.max_value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif
}