#ifndef STRUCT_STATS
#define STRUCT_STATS

struct Stats
{
    int num_groups;
    int num_cells;
    int num_blocks;
    int num_traces;
    int num_cubics;
    int num_mips;
    int num_fetches;     
    int num_volume_fetches;
    int num_distance_fetches;
};

Stats stats; // Global mutable struct

void set_stats()
{
    stats.num_groups           = 0;
    stats.num_blocks           = 0;
    stats.num_cells            = 0;
    stats.num_traces           = 0;
    stats.num_cubics           = 0;
    stats.num_mips             = 0;
    stats.num_fetches          = 0;
    stats.num_volume_fetches   = 0;
    stats.num_distance_fetches = 0;
}

#endif // STRUCT_STATS