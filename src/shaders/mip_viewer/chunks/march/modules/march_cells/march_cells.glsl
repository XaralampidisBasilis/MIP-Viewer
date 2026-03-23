#if SKIPPING_ENABLED == 1

    #if SKIPPING_METHOD == 0
    
        #if PRODUCTION_ENABLED == 1
            #include "./production/march_cells_in_blocks1@production=true"
        #else
            #include "./misc/march_cells_in_blocks1@bernstein=false"
        #endif

    #elif SKIPPING_METHOD == 1

        #if PRODUCTION_ENABLED == 1
            #include "./production/march_cells_in_blocks2@production=true"
        #else
            #include "./misc/march_cells_in_blocks2@bernstein=false"
        #endif

    #endif    

#else

    #include "./misc/march_cells_in_cells@bernstein=false"

#endif


