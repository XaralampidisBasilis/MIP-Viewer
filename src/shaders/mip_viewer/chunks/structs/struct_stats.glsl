#ifndef STRUCT_STATS
#define STRUCT_STATS

struct Stats
{
    int num_groups;
    int num_cells;
    int num_blocks;
    int num_traces;
    int num_maxima;
    int num_mips;
    int num_fetches;     
    int num_volume_fetches;
    int num_distance_fetches;
};

Stats stats; // Global mutable struct

#endif // STRUCT_STATS
