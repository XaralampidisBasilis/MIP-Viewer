#if SKIPPING_ENABLED == 1

    #if VARIATION_ENABLED == 1
    #include "./march_traces_in_cells_in_blocks"
    #else
    #include "./march_traces_in_blocks"
    #endif

#else

    #include "./march_traces_in_traces"
    
#endif

