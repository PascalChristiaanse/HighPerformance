#pragma once

#include "poisson/SolverStrategy.hpp"

namespace poisson
{

    /// Red-Black Gauss-Seidel solver strategy
    class GaussSeidelStrategy : public SolverStrategy
    {
    public:
        /// Constructor
        /// @param optimizedLoop If true, use stride-2 loop without parity check
        explicit GaussSeidelStrategy(bool optimizedLoop = false);

        [[nodiscard]] std::string name() const override { return "Gauss-Seidel"; }

        void initialize(Grid &grid, const Config &config,
                        std::shared_ptr<MPIContext> mpi,
                        const std::array<int, 2> &subdomainOffset) override;

        [[nodiscard]] double doStep(Grid &grid, const Config &config,
                                    std::shared_ptr<MPIContext> mpi,
                                    const std::array<int, 2> &subdomainOffset) override;

        [[nodiscard]] double getResidual() const override { return lastResidual_; }

        [[nodiscard]] std::unique_ptr<SolverStrategy> clone() const override
        {
            return std::make_unique<GaussSeidelStrategy>(*this);
        }

    protected:
        /// Standard step with parity check in inner loop
        [[nodiscard]] double doStepStandard(Grid &grid, int parity,
                                            const std::array<int, 2> &subdomainOffset);

        /// Optimized step with stride-2 loop (no parity check)
        [[nodiscard]] double doStepOptimized(Grid &grid, int parity,
                                             const std::array<int, 2> &subdomainOffset);

        bool optimizedLoop_{false};
        double lastResidual_{0.0};
        int currentParity_{0}; ///< 0 for red, 1 for black
    };

    /// Successive Over-Relaxation (SOR) solver strategy
    class SORStrategy : public SolverStrategy
    {
    public:
        /// Constructor
        /// @param omega Relaxation parameter (typically 1.0 < omega < 2.0 for SOR)
        /// @param optimizedLoop If true, use stride-2 loop without parity check
        explicit SORStrategy(double omega = 1.95, bool optimizedLoop = false);

        [[nodiscard]] std::string name() const override { return "SOR"; }

        void initialize(Grid &grid, const Config &config,
                        std::shared_ptr<MPIContext> mpi,
                        const std::array<int, 2> &subdomainOffset) override;

        [[nodiscard]] double doStep(Grid &grid, const Config &config,
                                    std::shared_ptr<MPIContext> mpi,
                                    const std::array<int, 2> &subdomainOffset) override;

        [[nodiscard]] double getResidual() const override { return lastResidual_; }

        [[nodiscard]] std::unique_ptr<SolverStrategy> clone() const override
        {
            return std::make_unique<SORStrategy>(*this);
        }

        /// Get the omega value
        [[nodiscard]] double omega() const noexcept { return omega_; }

    protected:
        /// Standard SOR step with parity check
        [[nodiscard]] double doStepStandard(Grid &grid, int parity,
                                            const std::array<int, 2> &subdomainOffset);

        /// Optimized SOR step with stride-2 loop
        [[nodiscard]] double doStepOptimized(Grid &grid, int parity,
                                             const std::array<int, 2> &subdomainOffset);

        double omega_{1.95};
        bool optimizedLoop_{false};
        double lastResidual_{0.0};
        int currentParity_{0};
    };

    /// Conjugate Gradient solver strategy
    class ConjugateGradientStrategy : public SolverStrategy
    {
    public:
        ConjugateGradientStrategy() = default;

        [[nodiscard]] std::string name() const override { return "Conjugate-Gradient"; }

        void initialize(Grid &grid, const Config &config,
                        std::shared_ptr<MPIContext> mpi,
                        const std::array<int, 2> &subdomainOffset) override;

        [[nodiscard]] double doStep(Grid &grid, const Config &config,
                                    std::shared_ptr<MPIContext> mpi,
                                    const std::array<int, 2> &subdomainOffset) override;

        [[nodiscard]] double getResidual() const override { return globalResidue_; }

        /// CG uses residual norm as convergence criterion
        [[nodiscard]] bool usesCustomConvergence() const override { return true; }

        /// CG does one step per iteration, not red/black
        [[nodiscard]] bool isSingleStepPerIteration() const override { return true; }

        [[nodiscard]] std::unique_ptr<SolverStrategy> clone() const override
        {
            // CG is stateful, clone returns fresh instance (re-initialized when used)
            return std::make_unique<ConjugateGradientStrategy>();
        }

    private:
        /// Initialize CG vectors (pCG, rCG, vCG)
        void initCG(Grid &grid, const Config &config,
                    std::shared_ptr<MPIContext> mpi);

        /// Exchange pCG boundaries with neighbors
        void exchangePCG(std::shared_ptr<MPIContext> mpi);

        std::unique_ptr<Grid> pCGGrid_;  ///< Grid for pCG to use MPI exchange
        std::vector<double> rCG_;  ///< Residual
        std::vector<double> vCG_;  ///< Matrix-vector product result
        double globalResidue_{0.0};
        int dimX_{0}, dimY_{0};
        bool initialized_{false};
    };

} // namespace poisson
