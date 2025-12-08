#if SKIPPING_ENABLED == 1
#if SKIPPING_STRATEGY == 0
#include "./march_traces_in_blocks"

#elif SKIPPING_STRATEGY == 1
#include "./march_traces_in_groups"
#endif

#else
#include "./march_traces_in_traces"
#endif
