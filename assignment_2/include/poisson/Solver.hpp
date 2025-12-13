#pragma once

#include "poisson/Config.hpp"
#include "poisson/Grid.hpp"
#include <memory>
#include <functional>
#include <optional>
#include <string>

namespace poisson
{

    // Forward declarations
    class MPIContext;

    /// Result of solving the Poisson equation
    struct SolveResult
    {
        int iterations;       ///< Number of iterations performed
        double finalResidual; ///< Final maximum residual
        double elapsedTime;   ///< Time taken in seconds
        bool converged;       ///< Whether the solver converged
    };

    /// Poisson equation solver using red-black Gauss-Seidel method
    class Solver
    {
    public:
        /// Construct a serial solver
        /// @param config Solver configuration
        explicit Solver(const Config &config);

        /// Construct a parallel solver with MPI context
        /// @param config Solver configuration
        /// @param mpi Shared MPI context
        Solver(const Config &config, std::shared_ptr<MPIContext> mpi);

        /// Destructor
        ~Solver();

        // Move semantics
        Solver(Solver &&) noexcept;
        Solver &operator=(Solver &&) noexcept;

        // No copying
        Solver(const Solver &) = delete;
        Solver &operator=(const Solver &) = delete;

        /// Solve the Poisson equation
        /// @return SolveResult with iteration count and convergence info
        [[nodiscard]] SolveResult solve();

        /// Get the solution grid (const access)
        [[nodiscard]] const Grid &grid() const noexcept;

        /// Get the solution grid (mutable access)
        [[nodiscard]] Grid &grid() noexcept;

        /// Get the configuration
        [[nodiscard]] const Config &config() const noexcept;

        /// Progress callback type: (iteration, residual, grid)
        /// Grid is const reference - use for recording/monitoring only
        using ProgressCallback = std::function<void(int, double, const Grid &)>;

        /// Set callback for progress monitoring and iteration recording
        void setProgressCallback(ProgressCallback callback);

    private:
        /// Calculate subdomain dimensions based on MPI Cartesian topology
        /// @return {nx, ny} subdomain interior dimensions for this rank
        [[nodiscard]] std::array<int, 2> calculateSubdomainDims() const;

        /// Calculate subdomain offset in global grid
        /// @return {offsetX, offsetY} starting position in global grid
        [[nodiscard]] std::array<int, 2> calculateSubdomainOffset() const;

        /// Perform one red-black Gauss-Seidel step
        /// @param parity 0 for red, 1 for black
        /// @return Maximum local residual
        [[nodiscard]] double doStep(int parity);

        /// Exchange boundary data with neighbors (MPI)
        void exchangeBoundaries();

        /// Compute global maximum residual (MPI reduction)
        [[nodiscard]] double globalMaxResidual(double localMax);

        /// Initialize the grid with source points
        void initializeGrid();

        Config config_;
        std::shared_ptr<MPIContext> mpi_;
        std::array<int, 2> subdomainDims_; ///< Local subdomain dimensions (interior only)
        std::array<int, 2> subdomainOffset_; ///< Starting position in global grid
        Grid grid_;  // Must be declared after subdomainDims_ for proper initialization order
        ProgressCallback progressCallback_;
    };

} // namespace poisson
