#if SKIPPING_ENABLED == 1

    #if SKIPPING_METHOD == 0
    
        #if VARIATION_ENABLED == 1
            #include "./march_cells_in_blocks1_var"
        #else
            #include "./march_cells_in_cells1_var"
        #endif

    #elif SKIPPING_METHOD == 1

        #if VARIATION_ENABLED == 1
            #include "./march_blocks_in_blocks2"
        #else
            #include "./march_cells_in_cells2_var"
        #endif

        // #include "./march_cells_in_blocks2_var"

    #endif    

#else

    // #if VARIATION_ENABLED == 1
    //     #include "./march_cells_in_cells0_var"
    // #else
    //     #include "./march_cells_in_cells0"
    // #endif

    #include "./march_cells_in_cells0_var"

#endif

