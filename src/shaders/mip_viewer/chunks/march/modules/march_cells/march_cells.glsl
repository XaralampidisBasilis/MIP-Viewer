#if SKIPPING_ENABLED == 1

    #if SKIPPING_METHOD == 0
    
        #include "./march_cells_in_cells1"

    #elif SKIPPING_METHOD == 1

        #include "./march_cells_in_cells2"

    #endif    

#else

    #include "./march_cells_in_cells"

#endif

