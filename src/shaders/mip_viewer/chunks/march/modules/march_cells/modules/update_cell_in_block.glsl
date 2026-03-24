

#if BLOCK_SIZE == 1
#include "./update_cell_in_block/block_size_1"
#else
#include "./update_cell_in_block/block_size_k"
#endif

