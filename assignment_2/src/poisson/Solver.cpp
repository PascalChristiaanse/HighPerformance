#include "poisson/Solver.hpp"
#include "poisson/Timer.hpp"
#include "mpi/MPIContext.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace poisson
{

    std::array<int, 2> Solver::calculateSubdomainDims() const
    {
        if (!mpi_ || mpi_->size() == 1)
        {
            // Serial case: full grid
            return {config_.nx(), config_.ny()};
        }

        // Get MPI Cartesian topology info
        auto cartDims = mpi_->cartDims();
        auto coords = mpi_->coords();

        // Calculate subdomain size per process
        int subNx = config_.nx() / cartDims[0];
        int subNy = config_.ny() / cartDims[1];

        // Handle uneven divisions (give remainder to last process in each dimension)
        if (coords[0] == cartDims[0] - 1)
        {
            subNx += config_.nx() % cartDims[0];
        }
        if (coords[1] == cartDims[1] - 1)
        {
            subNy += config_.ny() % cartDims[1];
        }

        return {subNx, subNy};
    }

    std::array<int, 2> Solver::calculateSubdomainOffset() const
    {
        if (!mpi_ || mpi_->size() == 1)
        {
            // Serial case: starts at (0, 0)
            return {0, 0};
        }

        auto cartDims = mpi_->cartDims();
        auto coords = mpi_->coords();

        // Calculate starting offset based on process coordinates
        int baseNx = config_.nx() / cartDims[0];
        int baseNy = config_.ny() / cartDims[1];

        int offsetX = coords[0] * baseNx;
        int offsetY = coords[1] * baseNy;

        return {offsetX, offsetY};
    }

    Solver::Solver(const Config &config)
        : config_(config), grid_(config.nx(), config.ny()), mpi_(nullptr),
          subdomainDims_({config.nx(), config.ny()}), subdomainOffset_({0, 0})
    {
        initializeGrid();
    }

    Solver::Solver(const Config &config, std::shared_ptr<MPIContext> mpi)
        : config_(config), mpi_(std::move(mpi)),
          subdomainDims_(calculateSubdomainDims()), 
          subdomainOffset_(calculateSubdomainOffset()),
          grid_(subdomainDims_[0], subdomainDims_[1])
    {
        printf("Process %d/%d with coords (%d, %d): Subdomain size = (%d, %d), Offset = (%d, %d)\n",
               mpi_->rank(), mpi_->size(),
               mpi_->coords()[0], mpi_->coords()[1],
               subdomainDims_[0], subdomainDims_[1],
               subdomainOffset_[0], subdomainOffset_[1]);
        
        // Set subdomain offset for parallel I/O
        grid_.setSubdomainOffset(subdomainOffset_);
        
        // Create MPI datatypes for boundary exchange
        mpi_->createBorderTypes(grid_);
        
        initializeGrid();
    }

    Solver::~Solver() = default;

    Solver::Solver(Solver &&) noexcept = default;
    Solver &Solver::operator=(Solver &&) noexcept = default;

    void Solver::initializeGrid()
    {
        // Apply source points that belong to this subdomain
        for (const auto &src : config_.sources())
        {
            // Convert normalized [0,1] coordinates to global grid indices
            int globalX = static_cast<int>(src.x * config_.nx());
            int globalY = static_cast<int>(src.y * config_.ny());

            // Check if this source point belongs to this subdomain
            int endX = subdomainOffset_[0] + subdomainDims_[0];
            int endY = subdomainOffset_[1] + subdomainDims_[1];

            if (globalX >= subdomainOffset_[0] && globalX < endX &&
                globalY >= subdomainOffset_[1] && globalY < endY)
            {
                // Scale source coordinates to local grid [0, 1]
                double localNormX = static_cast<double>(globalX - subdomainOffset_[0]) / subdomainDims_[0];
                double localNormY = static_cast<double>(globalY - subdomainOffset_[1]) / subdomainDims_[1];

                grid_.applySource(localNormX, localNormY, src.value);
            }
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
            exchangeBoundaries();

            double delta2 = doStep(1); // Black cells
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

        // Get global offset for correct parity calculation in parallel
        // The ghost layer offset (xStart, yStart) maps to global position subdomainOffset_
        const int globalOffsetX = subdomainOffset_[0] - xStart;
        const int globalOffsetY = subdomainOffset_[1] - yStart;

        for (int x = xStart; x < xEnd; ++x)
        {
            for (int y = yStart; y < yEnd; ++y)
            {
                // Use GLOBAL coordinates for red-black parity check
                // This ensures the same cells are "red" or "black" regardless of domain decomposition
                int globalX = x + globalOffsetX;
                int globalY = y + globalOffsetY;

                if ((globalX + globalY) % 2 == parity && grid_.source(x, y) != 1)
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
        if (mpi_)
        {
            mpi_->exchangeBoundaries(grid_);
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
