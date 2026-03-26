#if SKIPPING_ENABLED == 1

    #if PRODUCTION_ENABLED == 1
    #include "./production/march_traces_in_blocks"
    #else
    #include "./march_traces_in_blocks"
    #endif

#else

    #if PRODUCTION_ENABLED == 1
    #include "./production/march_traces_in_traces"
    #else
    #include "./march_traces_in_traces"
    #endif
    
#endif

