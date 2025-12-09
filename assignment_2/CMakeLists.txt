cmake_minimum_required(VERSION 3.16)
project(PoissonSolver LANGUAGES C CXX)

# Find MPI
find_package(MPI REQUIRED)

# Set C and C++ standards
# set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)

# Optimization flags
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Sequential Poisson solver (plain C, no MPI needed for sequential)
# add_executable(SEQ_Poisson ${CMAKE_CURRENT_SOURCE_DIR}/SEQ_Poisson.c)
# target_link_libraries(SEQ_Poisson PRIVATE m)


# Parallel Poisson solver (MPI)
add_executable(PAR_Poisson ${CMAKE_CURRENT_SOURCE_DIR}/PAR_Poisson.cpp)
target_link_libraries(PAR_Poisson PRIVATE MPI::MPI_CXX m)
