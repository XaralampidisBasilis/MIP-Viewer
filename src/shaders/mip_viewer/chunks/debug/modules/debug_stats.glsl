
// COMPUTE DEBUG

// num cells
vec4 debug_stats_num_cells = to_color(turbo(float(stats.num_cells) / float(MAX_CELLS)));

// num traces
vec4 debug_stats_num_traces = to_color(turbo(float(stats.num_traces) / float(MAX_TRACES)));

// num mips
vec4 debug_stats_num_mips = to_color(turbo(float(stats.num_mips) / float(MAX_TRACES)));

// num blocks
vec4 debug_stats_num_blocks = to_color(turbo(float(stats.num_blocks) / float(MAX_BLOCKS)));

// num groups
vec4 debug_stats_num_groups = to_color(turbo(float(stats.num_groups) / float(MAX_GROUPS)));

// num fetches
vec4 debug_stats_num_fetches = to_color(turbo(float(stats.num_fetches) / float(MAX_CELLS + MAX_BLOCKS)));

// num_volume_fetches
vec4 debug_stats_num_volume_fetches = to_color(turbo(float(stats.num_volume_fetches) / float(MAX_TRACES)));

// num_distance_fetches
vec4 debug_stats_num_distance_fetches = to_color(turbo(float(stats.num_distance_fetches) / float(MAX_BLOCKS)));


// PRINT DEBUG
switch (u_debug.option - 900)
{
    case 1: fragColor = debug_stats_num_cells;         break;
    case 2: fragColor = debug_stats_num_traces;        break;
    case 3: fragColor = debug_stats_num_mips;          break;
    case 4: fragColor = debug_stats_num_blocks;        break;
    case 5: fragColor = debug_stats_num_groups;        break;
    case 6: fragColor = debug_stats_num_fetches;       break;
    case 7: fragColor = debug_stats_num_volume_fetches;   break;
    case 8: fragColor = debug_stats_num_distance_fetches;  break;
}