
// COMPUTE DEBUG

// num cells
vec4 debug_stats_num_cells = to_color(float(stats.num_cells) / float(MAX_CELLS));

// num traces
vec4 debug_stats_num_traces = to_color(float(stats.num_traces) / float(MAX_TRACES));

// num blocks
vec4 debug_stats_num_blocks = to_color(float(stats.num_blocks) / float(MAX_BLOCKS));

// num groups
vec4 debug_stats_num_groups = to_color(float(stats.num_groups) / float(MAX_GROUPS));

// num fetches
vec4 debug_stats_num_fetches = to_color(float(stats.num_fetches) / float(MAX_CELLS));


// PRINT DEBUG
switch (u_debug.option - 900)
{
    case 1: fragColor = debug_stats_num_cells;      break;
    case 2: fragColor = debug_stats_num_traces;     break;
    case 3: fragColor = debug_stats_num_blocks;     break;
    case 4: fragColor = debug_stats_num_groups;     break;
    case 5: fragColor = debug_stats_num_fetches;    break;
}