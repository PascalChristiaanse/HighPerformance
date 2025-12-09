#!/bin/bash
# Setup script for HPC Development Environment
# Source this script: source setup_env.sh

echo "Loading HPC development modules (2024r1)..."
module load 2024r1 cmake openmpi ninja

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
