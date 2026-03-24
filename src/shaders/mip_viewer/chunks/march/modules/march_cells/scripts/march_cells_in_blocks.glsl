

#if BLOCK_SIZE == 1
#include "./march_cells_in_blocks@block_size_1"
#else
#include "./march_cells_in_blocks@block_size_K"
#endif