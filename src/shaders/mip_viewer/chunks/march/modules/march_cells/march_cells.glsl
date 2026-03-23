#if SKIPPING_ENABLED == 1

    #if VARIATION_ENABLED == 1
        #include "./march_cells_in_blocks1_var"
    #else
        #include "./march_cells_in_cells1_var"
    #endif 

#else

    #include "./march_cells_in_cells0_var"

#endif

