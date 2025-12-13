#pragma once

#include "poisson/Grid.hpp"
#include "poisson/Telemetry.hpp"
#include <memory>
#include <string>

namespace poisson
{
    // Forward declarations
    class Config;
    class MPIContext;

    /// Abstract base class for solver strategies
    /// Different solving methods (Gauss-Seidel, SOR, CG) implement this interface
    class SolverStrategy
    {
    public:
        virtual ~SolverStrategy() = default;

        /// Get the name of this solver method
        [[nodiscard]] virtual std::string name() const = 0;

        /// Initialize the solver (called once before solve loop starts)
        virtual void initialize(Grid &grid, const Config &config,
                                std::shared_ptr<MPIContext> mpi,
                                const std::array<int, 2> &subdomainOffset) = 0;

        /// Perform one iteration step
        /// @param grid The grid to update
        /// @param config Solver configuration
        /// @param mpi MPI context (may be null for serial)
        /// @param subdomainOffset Global offset of this subdomain
        /// @return Local maximum residual from this step
        [[nodiscard]] virtual double doStep(Grid &grid, const Config &config,
                                            std::shared_ptr<MPIContext> mpi,
                                            const std::array<int, 2> &subdomainOffset) = 0;

        /// Get the current global residual (for CG this is different from max change)
        [[nodiscard]] virtual double getResidual() const = 0;

        /// Check if this method uses a different convergence criterion than max delta
        [[nodiscard]] virtual bool usesCustomConvergence() const { return false; }

        /// Check if this method does a single step per iteration (CG) vs red/black (GS/SOR)
        [[nodiscard]] virtual bool isSingleStepPerIteration() const { return false; }

        /// Create a copy of this strategy
        [[nodiscard]] virtual std::unique_ptr<SolverStrategy> clone() const = 0;
    };

    /// Factory for creating solver strategies
    class SolverStrategyFactory
    {
    public:
        /// Solver method enumeration
        enum class Method
        {
            GaussSeidel, ///< Red-black Gauss-Seidel (default)
            SOR,         ///< Successive Over-Relaxation
            CG           ///< Conjugate Gradient
        };

        /// Create a solver strategy by method type
        /// @param method The solver method to use
        /// @param omega Relaxation parameter for SOR (ignored for other methods)
        /// @param optimizedLoop Whether to use optimized loop without parity check
        [[nodiscard]] static std::unique_ptr<SolverStrategy> create(
            Method method, double omega = 1.0, bool optimizedLoop = false);

        /// Parse method from string
        /// @param str Method name: "gs", "gauss-seidel", "sor", "cg", "conjugate-gradient"
        /// @return Method enum value
        /// @throws std::invalid_argument if string is not recognized
        [[nodiscard]] static Method parseMethod(const std::string &str);

        /// Convert method to string
        [[nodiscard]] static std::string methodToString(Method method);
    };

} // namespace poisson
