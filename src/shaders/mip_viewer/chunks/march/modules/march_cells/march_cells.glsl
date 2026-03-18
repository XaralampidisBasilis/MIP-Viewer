#if SKIPPING_ENABLED == 1

    #if SKIPPING_METHOD == 0
    
        #include "./march_cells_in_blocks3"

    #elif SKIPPING_METHOD == 1

        #include "./march_cells_in_blocks2"

    #endif    

#else

    #include "./march_cells_in_cells2"

#endif

