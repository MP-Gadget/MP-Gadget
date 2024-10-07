cmake_minimum_required(VERSION 2.9)

project(bigfile)

# Finding optional dependencies
find_package(MPI)

# Add library subdirectoy
add_subdirectory(src)

# Add executables subdirectory
add_subdirectory(utils)
