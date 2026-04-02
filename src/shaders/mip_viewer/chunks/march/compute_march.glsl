
#if MARCHING_METHOD == 0

    #include "./march_cells"

#elif MARCHING_METHOD == 1

    #include "./march_traces"

#elif MARCHING_METHOD == 2

    #include "./march_blocks"

#endif

