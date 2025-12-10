cmake_minimum_required(VERSION 3.16)
project(PoissonSolver 
    VERSION 2.0.0
    LANGUAGES C CXX
    DESCRIPTION "Parallel 2D Poisson Solver with MPI"
)

# ============== Compiler Settings ==============
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# ============== Options ==============
option(POISSON_ENABLE_HDF5 "Enable HDF5+XDMF output" OFF)

# ============== Dependencies ==============
find_package(MPI REQUIRED COMPONENTS C CXX)

if(POISSON_ENABLE_HDF5)
    # HDF5 with MPI support needs MPI to be found first with C component
    find_package(HDF5 REQUIRED COMPONENTS C)
    add_compile_definitions(POISSON_HAS_HDF5)
endif()

# ============== Library: poisson_core ==============
add_library(poisson_core STATIC
    src/poisson/Config.cpp
    src/poisson/Grid.cpp
    src/poisson/Solver.cpp
    src/poisson/Timer.cpp
)

target_include_directories(poisson_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(poisson_core PUBLIC cxx_std_17)

# Compiler warnings
target_compile_options(poisson_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

# ============== Library: poisson_io ==============
add_library(poisson_io STATIC
    src/io/FileManager.cpp
    src/io/AsciiWriter.cpp
    src/io/HDF5Writer.cpp
    src/io/IterationRecorder.cpp
)

target_link_libraries(poisson_io PUBLIC poisson_core)

if(POISSON_ENABLE_HDF5)
    # Use HDF5 C library (C++ wrapper not needed with C API)
    target_link_libraries(poisson_io PUBLIC ${HDF5_C_LIBRARIES})
    target_include_directories(poisson_io PUBLIC ${HDF5_INCLUDE_DIRS})
endif()

# ============== Library: poisson_mpi ==============
add_library(poisson_mpi STATIC
    src/mpi/MPIContext.cpp
)

target_link_libraries(poisson_mpi PUBLIC 
    poisson_core 
    MPI::MPI_CXX
)

# ============== Main Executable ==============
add_executable(PAR_Poisson 
    ${CMAKE_CURRENT_SOURCE_DIR}/PAR_Poisson.cpp
)

target_link_libraries(PAR_Poisson PRIVATE
    poisson_core
    poisson_io
    poisson_mpi
    m
)

# ============== Summary ==============
message(STATUS "")
message(STATUS "Poisson Solver Configuration:")
message(STATUS "  HDF5 support: ${POISSON_ENABLE_HDF5}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "")
