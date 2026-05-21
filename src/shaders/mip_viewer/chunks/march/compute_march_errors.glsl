
#if MARCHING_METHOD == 0

    #if SKIPPING_ENABLED == 1

        #include "./march_cells_in_cells"

        Mip ref = mip;  // reference mip for error calculation

        #include "./march_cells_in_blocks"
        
    #endif

#elif MARCHING_METHOD == 1

    #if SKIPPING_ENABLED == 1

        #if VARIATION_ENABLED == 1

            #include "./march_traces_in_traces"

            Mip ref = mip;  // reference mip for error calculation

            #include "./march_traces_in_blocks"

        #else

            #include "./march_traces_in_cells"

            Mip ref = mip;  // reference mip for error calculation

            #include "./march_traces_in_cells_in_blocks"
            
        #endif

    #endif

#endif

