#include "poisson/SolverStrategies.hpp"
#include "poisson/SolverStrategy.hpp"
#include "poisson/Config.hpp"
#include "mpi/MPIContext.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace poisson
{

    // ============== Factory Implementation ==============

    std::unique_ptr<SolverStrategy> SolverStrategyFactory::create(
        Method method, double omega, bool optimizedLoop)
    {
        switch (method)
        {
        case Method::GaussSeidel:
            return std::make_unique<GaussSeidelStrategy>(optimizedLoop);
        case Method::SOR:
            return std::make_unique<SORStrategy>(omega, optimizedLoop);
        case Method::CG:
            return std::make_unique<ConjugateGradientStrategy>();
        default:
            throw std::invalid_argument("Unknown solver method");
        }
    }

    SolverStrategyFactory::Method SolverStrategyFactory::parseMethod(const std::string &str)
    {
        if (str == "gs" || str == "gauss-seidel" || str == "GaussSeidel")
        {
            return Method::GaussSeidel;
        }
        else if (str == "sor" || str == "SOR")
        {
            return Method::SOR;
        }
        else if (str == "cg" || str == "conjugate-gradient" || str == "CG" || str == "ConjugateGradient")
        {
            return Method::CG;
        }
        throw std::invalid_argument("Unknown solver method: " + str +
                                    ". Valid options: gs, sor, cg");
    }

    std::string SolverStrategyFactory::methodToString(Method method)
    {
        switch (method)
        {
        case Method::GaussSeidel:
            return "Gauss-Seidel";
        case Method::SOR:
            return "SOR";
        case Method::CG:
            return "Conjugate-Gradient";
        default:
            return "Unknown";
        }
    }

    // ============== Gauss-Seidel Implementation ==============

    GaussSeidelStrategy::GaussSeidelStrategy(bool optimizedLoop)
        : optimizedLoop_(optimizedLoop)
    {
    }

    void GaussSeidelStrategy::initialize(Grid &, const Config &,
                                         std::shared_ptr<MPIContext>,
                                         const std::array<int, 2> &)
    {
        // No special initialization needed for Gauss-Seidel
        currentParity_ = 0;
        lastResidual_ = 0.0;
    }

    double GaussSeidelStrategy::doStep(Grid &grid, const Config &,
                                       std::shared_ptr<MPIContext>,
                                       const std::array<int, 2> &subdomainOffset)
    {
        double maxErr;
        if (optimizedLoop_)
        {
            maxErr = doStepOptimized(grid, currentParity_, subdomainOffset);
        }
        else
        {
            maxErr = doStepStandard(grid, currentParity_, subdomainOffset);
        }

        lastResidual_ = maxErr;
        currentParity_ = 1 - currentParity_; // Alternate between red (0) and black (1)
        return maxErr;
    }

    double GaussSeidelStrategy::doStepStandard(Grid &grid, int parity,
                                               const std::array<int, 2> &subdomainOffset)
    {
        double maxErr = 0.0;

        const int xStart = grid.ghostLayers();
        const int xEnd = grid.dimX() - grid.ghostLayers();
        const int yStart = grid.ghostLayers();
        const int yEnd = grid.dimY() - grid.ghostLayers();

        // Global offset for correct parity calculation
        const int globalOffsetX = subdomainOffset[0] - xStart;
        const int globalOffsetY = subdomainOffset[1] - yStart;

        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                int globalX = x + globalOffsetX;
                int globalY = y + globalOffsetY;

                if ((globalX + globalY) % 2 == parity && grid.source(x, y) != 1)
                {
                    double oldPhi = grid.phi(x, y);

                    // 5-point stencil Laplacian (standard Gauss-Seidel, omega = 1)
                    grid.phi(x, y) = 0.25 * (grid.phi(x + 1, y) + grid.phi(x - 1, y) +
                                             grid.phi(x, y + 1) + grid.phi(x, y - 1));

                    double err = std::fabs(oldPhi - grid.phi(x, y));
                    if (err > maxErr)
                    {
                        maxErr = err;
                    }
                }
            }
        }

        return maxErr;
    }

    double GaussSeidelStrategy::doStepOptimized(Grid &grid, int parity,
                                                const std::array<int, 2> &subdomainOffset)
    {
        double maxErr = 0.0;

        const int xStart = grid.ghostLayers();
        const int xEnd = grid.dimX() - grid.ghostLayers();
        const int yStart = grid.ghostLayers();
        const int yEnd = grid.dimY() - grid.ghostLayers();

        const int globalOffsetX = subdomainOffset[0] - xStart;
        const int globalOffsetY = subdomainOffset[1] - yStart;

        for (int x = xStart; x < xEnd; ++x)
        {
            // Calculate starting y based on parity - stride by 2
            int globalX = x + globalOffsetX;
            int yStartAdj = yStart;

            // Adjust starting y so that (globalX + globalY) % 2 == parity
            int globalYStart = yStart + globalOffsetY;
            if ((globalX + globalYStart) % 2 != parity)
            {
                yStartAdj = yStart + 1;
            }

            for (int y = yStartAdj; y < yEnd; y += 2)
            {
                if (grid.source(x, y) != 1)
                {
                    double oldPhi = grid.phi(x, y);

                    grid.phi(x, y) = 0.25 * (grid.phi(x + 1, y) + grid.phi(x - 1, y) +
                                             grid.phi(x, y + 1) + grid.phi(x, y - 1));

                    double err = std::fabs(oldPhi - grid.phi(x, y));
                    if (err > maxErr)
                    {
                        maxErr = err;
                    }
                }
            }
        }

        return maxErr;
    }

    // ============== SOR Implementation ==============

    SORStrategy::SORStrategy(double omega, bool optimizedLoop)
        : omega_(omega), optimizedLoop_(optimizedLoop)
    {
    }

    void SORStrategy::initialize(Grid &, const Config &,
                                 std::shared_ptr<MPIContext>,
                                 const std::array<int, 2> &)
    {
        currentParity_ = 0;
        lastResidual_ = 0.0;
    }

    double SORStrategy::doStep(Grid &grid, const Config &,
                               std::shared_ptr<MPIContext>,
                               const std::array<int, 2> &subdomainOffset)
    {
        double maxErr;
        if (optimizedLoop_)
        {
            maxErr = doStepOptimized(grid, currentParity_, subdomainOffset);
        }
        else
        {
            maxErr = doStepStandard(grid, currentParity_, subdomainOffset);
        }

        lastResidual_ = maxErr;
        currentParity_ = 1 - currentParity_;
        return maxErr;
    }

    double SORStrategy::doStepStandard(Grid &grid, int parity,
                                       const std::array<int, 2> &subdomainOffset)
    {
        double maxErr = 0.0;

        const int xStart = grid.ghostLayers();
        const int xEnd = grid.dimX() - grid.ghostLayers();
        const int yStart = grid.ghostLayers();
        const int yEnd = grid.dimY() - grid.ghostLayers();

        const int globalOffsetX = subdomainOffset[0] - xStart;
        const int globalOffsetY = subdomainOffset[1] - yStart;

        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                int globalX = x + globalOffsetX;
                int globalY = y + globalOffsetY;

                if ((globalX + globalY) % 2 == parity && grid.source(x, y) != 1)
                {
                    double oldPhi = grid.phi(x, y);

                    // SOR formula: phi_new = omega * gauss_seidel + (1 - omega) * phi_old
                    // Which simplifies to: phi_new = phi_old + omega * (gauss_seidel - phi_old)
                    // Or using the change: phi_new = phi_old + omega * c
                    // where c = 0.25*(neighbors) - phi_old

                    double gsValue = 0.25 * (grid.phi(x + 1, y) + grid.phi(x - 1, y) +
                                             grid.phi(x, y + 1) + grid.phi(x, y - 1));
                    double change = gsValue - oldPhi;
                    grid.phi(x, y) = oldPhi + omega_ * change;

                    double err = std::fabs(omega_ * change);
                    if (err > maxErr)
                    {
                        maxErr = err;
                    }
                }
            }
        }

        return maxErr;
    }

    double SORStrategy::doStepOptimized(Grid &grid, int parity,
                                        const std::array<int, 2> &subdomainOffset)
    {
        double maxErr = 0.0;

        const int xStart = grid.ghostLayers();
        const int xEnd = grid.dimX() - grid.ghostLayers();
        const int yStart = grid.ghostLayers();
        const int yEnd = grid.dimY() - grid.ghostLayers();

        const int globalOffsetX = subdomainOffset[0] - xStart;
        const int globalOffsetY = subdomainOffset[1] - yStart;

        for (int x = xStart; x < xEnd; ++x)
        {
            int globalX = x + globalOffsetX;
            int yStartAdj = yStart;

            int globalYStart = yStart + globalOffsetY;
            if ((globalX + globalYStart) % 2 != parity)
            {
                yStartAdj = yStart + 1;
            }

            for (int y = yStartAdj; y < yEnd; y += 2)
            {
                if (grid.source(x, y) != 1)
                {
                    double oldPhi = grid.phi(x, y);

                    double gsValue = 0.25 * (grid.phi(x + 1, y) + grid.phi(x - 1, y) +
                                             grid.phi(x, y + 1) + grid.phi(x, y - 1));
                    double change = gsValue - oldPhi;
                    grid.phi(x, y) = oldPhi + omega_ * change;

                    double err = std::fabs(omega_ * change);
                    if (err > maxErr)
                    {
                        maxErr = err;
                    }
                }
            }
        }

        return maxErr;
    }

    // ============== Conjugate Gradient Implementation ==============

    void ConjugateGradientStrategy::initialize(Grid &grid, const Config &config,
                                               std::shared_ptr<MPIContext> mpi,
                                               const std::array<int, 2> &)
    {
        initCG(grid, config, mpi);
    }

    void ConjugateGradientStrategy::initCG(Grid &grid, const Config &,
                                           std::shared_ptr<MPIContext> mpi)
    {
        dimX_ = grid.dimX();
        dimY_ = grid.dimY();

        const int totalCells = dimX_ * dimY_;

        // Create a separate grid for pCG to use MPI boundary exchange
        pCGGrid_ = std::make_unique<Grid>(grid.interiorDimX(), grid.interiorDimY(), grid.ghostLayers());
        
        // Allocate other CG vectors (these don't need ghost cells)
        rCG_.resize(totalCells, 0.0);
        vCG_.resize(totalCells, 0.0);

        const int xStart = grid.ghostLayers();
        const int xEnd = dimX_ - grid.ghostLayers();
        const int yStart = grid.ghostLayers();
        const int yEnd = dimY_ - grid.ghostLayers();

        // Initialize residual: r = b - A*x
        // For Laplace equation: b = 0, A*x = x - 0.25*(neighbors)
        // So r = -A*x = 0.25*(neighbors) - x
        // Initially phi = 0 except at sources

        double localRdotR = 0.0;
        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                int idx = x * dimY_ + y;

                if (grid.source(x, y) != 1)
                {
                    // r = b - A*phi = 0 - (phi - 0.25*neighbors) = 0.25*neighbors - phi
                    rCG_[idx] = 0.25 * (grid.phi(x + 1, y) + grid.phi(x - 1, y) +
                                        grid.phi(x, y + 1) + grid.phi(x, y - 1)) -
                                grid.phi(x, y);
                    // p = r initially
                    pCGGrid_->phi(x, y) = rCG_[idx];
                    localRdotR += rCG_[idx] * rCG_[idx];
                }
                else
                {
                    rCG_[idx] = 0.0;
                    pCGGrid_->phi(x, y) = 0.0;
                }
            }
        }

        // Global reduction for initial residue
        if (mpi)
        {
            globalResidue_ = mpi->allReduceSum(localRdotR);
        }
        else
        {
            globalResidue_ = localRdotR;
        }

        initialized_ = true;
    }

    void ConjugateGradientStrategy::exchangePCG(std::shared_ptr<MPIContext> mpi)
    {
        if (mpi && pCGGrid_)
        {
            mpi->exchangeBoundaries(*pCGGrid_);
        }
    }

    double ConjugateGradientStrategy::doStep(Grid &grid, const Config &,
                                             std::shared_ptr<MPIContext> mpi,
                                             const std::array<int, 2> &)
    {
        if (!initialized_ || !pCGGrid_)
        {
            throw std::runtime_error("ConjugateGradientStrategy not initialized");
        }

        const int xStart = grid.ghostLayers();
        const int xEnd = dimX_ - grid.ghostLayers();
        const int yStart = grid.ghostLayers();
        const int yEnd = dimY_ - grid.ghostLayers();

        // Exchange pCG boundaries BEFORE matrix-vector multiply
        exchangePCG(mpi);

        // Step 1: Calculate v = A * p (matrix-vector multiply)
        // A*p = p - 0.25*(neighbors) for interior points
        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                int idx = x * dimY_ + y;

                vCG_[idx] = pCGGrid_->phi(x, y);
                if (grid.source(x, y) != 1)
                {
                    vCG_[idx] -= 0.25 * (pCGGrid_->phi(x + 1, y) + pCGGrid_->phi(x - 1, y) +
                                         pCGGrid_->phi(x, y + 1) + pCGGrid_->phi(x, y - 1));
                }
            }
        }

        // Step 2: Calculate p dot v
        double localPdotV = 0.0;
        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                int idx = x * dimY_ + y;
                localPdotV += pCGGrid_->phi(x, y) * vCG_[idx];
            }
        }

        double globalPdotV;
        if (mpi)
        {
            globalPdotV = mpi->allReduceSum(localPdotV);
        }
        else
        {
            globalPdotV = localPdotV;
        }

        // Step 3: Calculate alpha = r·r / p·v
        double alpha = globalResidue_ / globalPdotV;

        // Step 4: Update phi: phi = phi + alpha * p
        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                grid.phi(x, y) += alpha * pCGGrid_->phi(x, y);
            }
        }

        // Step 5: Update residual: r = r - alpha * v
        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                int idx = x * dimY_ + y;
                rCG_[idx] -= alpha * vCG_[idx];
            }
        }

        // Step 6: Calculate new r·r
        double localNewRdotR = 0.0;
        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                int idx = x * dimY_ + y;
                localNewRdotR += rCG_[idx] * rCG_[idx];
            }
        }

        double globalNewRdotR;
        if (mpi)
        {
            globalNewRdotR = mpi->allReduceSum(localNewRdotR);
        }
        else
        {
            globalNewRdotR = localNewRdotR;
        }

        // Step 7: Calculate gamma = new_r·r / old_r·r
        double gamma = globalNewRdotR / globalResidue_;
        globalResidue_ = globalNewRdotR;

        // Step 8: Update search direction: p = r + gamma * p
        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                int idx = x * dimY_ + y;
                pCGGrid_->phi(x, y) = rCG_[idx] + gamma * pCGGrid_->phi(x, y);
            }
        }

        return globalResidue_;
    }

} // namespace poisson
