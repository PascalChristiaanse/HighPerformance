#include "io/IterationRecorder.hpp"

namespace io
{

    IterationRecorder::IterationRecorder(
        const poisson::Config &config,
        const std::filesystem::path &basePath,
        int mpiRank,
        int mpiSize)
        : config_(config), basePath_(basePath), mpiRank_(mpiRank), mpiSize_(mpiSize)
    {
    }

    IterationRecorder::~IterationRecorder()
    {
        if (enabled_ && !finalized_)
        {
            finalize();
        }
    }

    bool IterationRecorder::enable(int saveInterval)
    {
        if (!isAvailable())
        {
            lastError_ = "HDF5 support not available. Rebuild with -DPOISSON_ENABLE_HDF5=ON";
            return false;
        }

        if (saveInterval < 1)
        {
            lastError_ = "Save interval must be >= 1";
            return false;
        }

        // Only rank 0 actually writes in parallel mode
        // (or all ranks if we later add parallel HDF5 support)
        if (mpiRank_ != 0)
        {
            // Non-root ranks just mark as enabled but don't write
            enabled_ = true;
            saveInterval_ = saveInterval;
            return true;
        }

        writer_ = std::make_unique<HDF5Writer>(mpiRank_, mpiSize_);

        if (!writer_->openTimeSeries(basePath_, config_))
        {
            lastError_ = "Failed to open time series: " + writer_->lastError();
            writer_.reset();
            return false;
        }

        enabled_ = true;
        saveInterval_ = saveInterval;
        recordedCount_ = 0;
        finalized_ = false;

        return true;
    }

    void IterationRecorder::disable()
    {
        if (enabled_ && !finalized_ && writer_)
        {
            finalize();
        }
        enabled_ = false;
        writer_.reset();
    }

    bool IterationRecorder::record(int iteration, double residual, const poisson::Grid &grid)
    {
        if (!enabled_)
        {
            return true; // Silently succeed when disabled
        }

        // Check save interval
        if (iteration % saveInterval_ != 0)
        {
            return true; // Skip this iteration
        }

        // Non-root ranks just count
        if (mpiRank_ != 0)
        {
            ++recordedCount_;
            return true;
        }

        if (!writer_ || finalized_)
        {
            lastError_ = "Recorder not properly initialized or already finalized";
            return false;
        }

        if (!writer_->appendTimeStep(grid, iteration, residual))
        {
            lastError_ = "Failed to append time step: " + writer_->lastError();
            return false;
        }

        ++recordedCount_;
        return true;
    }

    bool IterationRecorder::finalize()
    {
        if (finalized_)
        {
            return true; // Already done
        }

        finalized_ = true;

        if (!enabled_)
        {
            return true;
        }

        // Non-root ranks have nothing to finalize
        if (mpiRank_ != 0)
        {
            return true;
        }

        if (!writer_)
        {
            return true; // No writer means nothing to close
        }

        if (!writer_->closeTimeSeries())
        {
            lastError_ = "Failed to close time series: " + writer_->lastError();
            return false;
        }

        writer_.reset();
        return true;
    }

    std::function<void(int, double, const poisson::Grid &)> IterationRecorder::createCallback()
    {
        return [this](int iteration, double residual, const poisson::Grid &grid)
        {
            this->record(iteration, residual, grid);
        };
    }

} // namespace io
