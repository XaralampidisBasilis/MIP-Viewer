
#if MARCHING_METHOD == 0

    #include "./first_march_cells"

#elif MARCHING_METHOD == 1

    #include "./first_march_traces"

#elif MARCHING_METHOD == 2

    #include "./first_march_blocks"

#endif

