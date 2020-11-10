# Compilation flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# Compile library 
add_library(bigfile bigfile.c bigfile-record.c)
set_target_properties(bigfile PROPERTIES PUBLIC_HEADER bigfile.h)

install(TARGETS bigfile
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include)

if(${MPI_C_FOUND})
    include_directories(${MPI_C_INCLUDE_PATH})

    add_library(bigfile-mpi bigfile-mpi.c mp-mpiu.c)
    set_target_properties(bigfile-mpi PROPERTIES PUBLIC_HEADER bigfile-mpi.h)

    install(TARGETS bigfile-mpi
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include)
endif()
