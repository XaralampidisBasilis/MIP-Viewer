#ifndef STRUCT_STATS
#define STRUCT_STATS

struct Stats
{
    int num_fetches; // texture fetch
    int num_groups;
    int num_cells;
    int num_blocks;
    int num_traces;
    int num_mips;
    int volume_samples;
    int occlusion_samples;
};

Stats stats; // Global mutable struct

void set_stats()
{
    stats.num_groups        = 0;
    stats.num_blocks        = 0;
    stats.num_cells         = 0;
    stats.num_traces        = 0;
    stats.num_mips          = 0;
    stats.num_fetches       = 0;
    stats.volume_samples    = 0;
    stats.occlusion_samples = 0;
}

#endif // STRUCT_STATS