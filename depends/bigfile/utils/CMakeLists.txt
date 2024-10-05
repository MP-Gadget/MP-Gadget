# Compilation flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")

# Include library headers
include_directories(${CMAKE_SOURCE_DIR}/src)

# bigfile-get-attr
add_executable(bigfile-get-attr bigfile-get-attr.c)
target_link_libraries(bigfile-get-attr bigfile)

# bigfile-set-attr
add_executable(bigfile-set-attr bigfile-set-attr.c)
target_link_libraries(bigfile-set-attr bigfile)

# bigfile-copy
add_executable(bigfile-copy bigfile-copy.c)
target_link_libraries(bigfile-copy bigfile)

# bigfile-cat
add_executable(bigfile-cat bigfile-cat.c)
target_link_libraries(bigfile-cat bigfile)

# bigfile-create
add_executable(bigfile-create bigfile-create.c)
target_link_libraries(bigfile-create bigfile)

# bigfile-ls
add_executable(bigfile-ls bigfile-ls.c)
target_link_libraries(bigfile-ls bigfile)

# bigfile-checksum
#add_executable(bigfile-checksum bigfile-checksum.c)
#target_link_libraries(bigfile-checksum bigfile)

# bigfile-join
# add_executable(bigfile-join bigfile-join.c)
# target_link_libraries(bigfile-join bigfile)

# Install tagets
install(TARGETS bigfile-get-attr bigfile-set-attr bigfile-copy # bigfile-checksum
                bigfile-cat bigfile-create bigfile-ls # bigfile-join
        RUNTIME DESTINATION bin)

# MPI specific executables
if(${MPI_C_FOUND})
    include_directories(${MPI_C_INCLUDE_PATH})
    
    # bigfile-copy-mpi
    add_executable(bigfile-copy-mpi bigfile-copy-mpi.c)
    target_link_libraries(bigfile-copy-mpi bigfile-mpi bigfile ${MPI_C_LIBRARIES})

    # bigfile-iosim
    add_executable(bigfile-iosim bigfile-iosim.c)
    target_link_libraries(bigfile-iosim bigfile-mpi bigfile ${MPI_C_LIBRARIES})
    
    install(TARGETS bigfile-copy-mpi bigfile-iosim
            RUNTIME DESTINATION bin)
endif()

# Install bash scripts
install(PROGRAMS bigfile-check bigfile-rename bigfile-repartition bigfile-rm
        DESTINATION bin)
