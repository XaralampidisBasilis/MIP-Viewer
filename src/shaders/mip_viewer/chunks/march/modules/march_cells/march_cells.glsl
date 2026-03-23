#if SKIPPING_ENABLED == 1

    #if SKIPPING_METHOD == 0
    
        #if VARIATION_ENABLED == 1
            #include "./march_cells_in_blocks1@bernstein@production"
        #else
            #include "./march_cells_in_blocks1@bernstein"
        #endif

    #elif SKIPPING_METHOD == 1

        #if VARIATION_ENABLED == 1
            #include "./march_cells_in_blocks2@bernstein@production"
        #else
            #include "./march_cells_in_blocks2@bernstein"
        #endif

    #endif    

#else

    #include "./blockSize=1/march_cells_in_cells0@bernstein"

#endif


