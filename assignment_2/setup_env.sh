#!/bin/bash
# Setup script for HPC Development Environment
# Source this script: source setup_env.sh
#
# Options:
#   --hdf5    Also load HDF5 module for HDF5+XDMF output support

echo "Loading HPC development modules (2024r1)..."
# module load 2024r1 cmake openmpi ninja

# Check for --hdf5 flag
if [[ "$1" == "--hdf5" ]] || [[ "$POISSON_HDF5" == "1" ]]; then
    echo "Loading HDF5 module..."
    module load hdf5/1.14.3
    export POISSON_HDF5=1
fi

echo "Loaded modules:"
module list

echo ""
echo "Compilers available:"
echo "  mpicxx: $(which mpicxx)"
echo "  mpicc:  $(which mpicc)"
echo "  cmake:  $(which cmake)"
echo "  ninja:  $(which ninja)"
echo ""
echo "Environment ready! You can now run:"
echo "  cmake --preset debug-mpicxx-ninja"
echo "  cmake --build --preset debug-mpicxx-ninja"
echo ""
if [[ "$POISSON_HDF5" == "1" ]]; then
    echo "HDF5 enabled. Build with: cmake --preset debug-mpicxx-ninja -DPOISSON_ENABLE_HDF5=ON"
else
    echo "For HDF5 support: source setup_env.sh --hdf5"
fi
