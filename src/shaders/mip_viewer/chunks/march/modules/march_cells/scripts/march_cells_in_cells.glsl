

#if BERNSTEIN_ENABLED == 1
#include "./march_cells_in_cells@bernstein_enabled_true"
#else 
#include "./march_cells_in_cells@bernstein_enabled_false"
#endif