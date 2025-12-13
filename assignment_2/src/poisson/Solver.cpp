#include "poisson/Solver.hpp"
#include "poisson/Timer.hpp"
#include "poisson/SolverStrategies.hpp"
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
        // Create solver strategy based on config
        strategy_ = SolverStrategyFactory::create(
            static_cast<SolverStrategyFactory::Method>(config_.solverMethod()),
            config_.omega(),
            config_.useOptimizedLoop());

        initializeGrid();
        strategy_->initialize(grid_, config_, mpi_, subdomainOffset_);
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

        // Create solver strategy based on config
        strategy_ = SolverStrategyFactory::create(
            static_cast<SolverStrategyFactory::Method>(config_.solverMethod()),
            config_.omega(),
            config_.useOptimizedLoop());
        
        initializeGrid();
        strategy_->initialize(grid_, config_, mpi_, subdomainOffset_);
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
        Timer totalTimer;
        Timer stepTimer;
        Timer exchangeTimer;
        Timer reductionTimer;

        // Create telemetry if verbose timing is enabled
        auto telemetry = std::make_shared<SolveTelemetry>();

        totalTimer.start();

        int count = 0;
        double delta = 2.0 * config_.precisionGoal();
        const int errorCheckInterval = config_.errorCheckInterval();
        const int sweepsPerExchange = config_.sweepsPerExchange();
        const bool usesCustomConvergence = strategy_->usesCustomConvergence();
        const bool singleStepPerIteration = strategy_->isSingleStepPerIteration();

        while (delta > config_.precisionGoal() && count < config_.maxIterations())
        {
            IterationTelemetry iterTelemetry;
            iterTelemetry.iteration = count + 1;

            // Perform sweeps
            double localMax = 0.0;
            
            stepTimer.start();
            if (singleStepPerIteration)
            {
                // CG: single step per iteration (handles its own boundary exchange)
                localMax = strategy_->doStep(grid_, config_, mpi_, subdomainOffset_);
            }
            else
            {
                // GS/SOR: red-black sweeps
                for (int sweep = 0; sweep < sweepsPerExchange; ++sweep)
                {
                    // Red step
                    double delta1 = strategy_->doStep(grid_, config_, mpi_, subdomainOffset_);
                    
                    // Black step  
                    double delta2 = strategy_->doStep(grid_, config_, mpi_, subdomainOffset_);
                    
                    localMax = std::max(localMax, std::max(delta1, delta2));
                }
            }
            stepTimer.stop();
            iterTelemetry.stepTime = stepTimer.elapsedSeconds();

            // Exchange boundaries (once per iteration, after all sweeps)
            // Skip for CG as it handles its own pCG exchange
            if (!singleStepPerIteration)
            {
                exchangeTimer.start();
                exchangeBoundaries();
                exchangeTimer.stop();
                iterTelemetry.exchangeTime = exchangeTimer.elapsedSeconds();
            }
            else
            {
                iterTelemetry.exchangeTime = 0.0;
            }

            // Check convergence (potentially less frequently)
            if ((count + 1) % errorCheckInterval == 0 || count == 0)
            {
                reductionTimer.start();
                if (usesCustomConvergence)
                {
                    // CG uses its own residual calculation
                    delta = strategy_->getResidual();
                }
                else
                {
                    // GS/SOR use max change
                    delta = globalMaxResidual(localMax);
                }
                reductionTimer.stop();
                iterTelemetry.reductionTime = reductionTimer.elapsedSeconds();
            }
            else
            {
                iterTelemetry.reductionTime = 0.0;
            }

            iterTelemetry.residual = delta;
            iterTelemetry.totalIterTime = iterTelemetry.stepTime + 
                                          iterTelemetry.exchangeTime + 
                                          iterTelemetry.reductionTime;

            telemetry->recordIteration(iterTelemetry);

            ++count;

            // Progress callback if set
            if (progressCallback_)
            {
                progressCallback_(count, delta, grid_);
            }
        }

        totalTimer.stop();

        // Finalize telemetry
        telemetry->setFinalStats(totalTimer.elapsedSeconds(), count,
                                 delta <= config_.precisionGoal(), delta);

        return SolveResult{
            count,
            delta,
            totalTimer.elapsedSeconds(),
            delta <= config_.precisionGoal(),
            telemetry};
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
