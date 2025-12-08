
#if MARCHING_METHOD == 0
#include "./modules/march_cells/march_cells"

#elif MARCHING_METHOD == 1
#include "./modules/march_traces/march_traces"
#endif
