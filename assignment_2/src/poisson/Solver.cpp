#include "poisson/Solver.hpp"
#include "poisson/Timer.hpp"
#include "mpi/MPIContext.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace poisson
{

    Solver::Solver(const Config &config)
        : config_(config), grid_(config.nx(), config.ny()), mpi_(nullptr)
    {
        initializeGrid();
    }

    Solver::Solver(const Config &config, std::shared_ptr<MPIContext> mpi)
        : config_(config), grid_(config.nx(), config.ny()), mpi_(std::move(mpi))
    {
        initializeGrid();
    }

    Solver::~Solver() = default;

    Solver::Solver(Solver &&) noexcept = default;
    Solver &Solver::operator=(Solver &&) noexcept = default;

    void Solver::initializeGrid()
    {
        // Apply all source points from configuration
        for (const auto &src : config_.sources())
        {
            grid_.applySource(src.x, src.y, src.value);
        }
    }

    SolveResult Solver::solve()
    {
        Timer timer;
        timer.start();

        int count = 0;
        double delta = 2.0 * config_.precisionGoal();

        while (delta > config_.precisionGoal() && count < config_.maxIterations())
        {
            // Red-black Gauss-Seidel
            double delta1 = doStep(0); // Red cells
            double delta2 = doStep(1); // Black cells

            // Exchange boundaries if using MPI
            exchangeBoundaries();

            // Local maximum
            double localMax = std::max(delta1, delta2);

            // Global reduction for parallel case
            delta = globalMaxResidual(localMax);

            ++count;

            // Progress callback if set
            if (progressCallback_)
            {
                progressCallback_(count, delta, grid_);
            }
        }

        timer.stop();

        return SolveResult{
            count,
            delta,
            timer.elapsedSeconds(),
            delta <= config_.precisionGoal()};
    }

    double Solver::doStep(int parity)
    {
        double maxErr = 0.0;

        const int xStart = grid_.ghostLayers();
        const int xEnd = grid_.dimX() - grid_.ghostLayers();
        const int yStart = grid_.ghostLayers();
        const int yEnd = grid_.dimY() - grid_.ghostLayers();

        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                // Red-black ordering: only update cells matching parity
                if ((x + y) % 2 == parity && grid_.source(x, y) != 1)
                {
                    double oldPhi = grid_.phi(x, y);

                    // 5-point stencil Laplacian
                    grid_.phi(x, y) = 0.25 * (grid_.phi(x + 1, y) + grid_.phi(x - 1, y) +
                                              grid_.phi(x, y + 1) + grid_.phi(x, y - 1));

                    double err = std::fabs(oldPhi - grid_.phi(x, y));
                    if (err > maxErr)
                    {
                        maxErr = err;
                    }
                }
            }
        }

        return maxErr;
    }

    void Solver::exchangeBoundaries()
    {
        // No-op for serial solver
        // MPI implementation will override this behavior
        if (mpi_)
        {
            // TODO: Implement MPI boundary exchange
            // mpi_->exchangeBoundaries(grid_);
        }
    }

    double Solver::globalMaxResidual(double localMax)
    {
        if (mpi_)
        {
            return mpi_->allReduceMax(localMax);
        }
        return localMax;
    }

    const Grid &Solver::grid() const noexcept
    {
        return grid_;
    }

    Grid &Solver::grid() noexcept
    {
        return grid_;
    }

    const Config &Solver::config() const noexcept
    {
        return config_;
    }

    void Solver::setProgressCallback(ProgressCallback callback)
    {
        progressCallback_ = std::move(callback);
    }

} // namespace poisson
