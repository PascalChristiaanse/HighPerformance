#pragma once

#include "io/HDF5Writer.hpp"
#include "poisson/Grid.hpp"
#include "poisson/Config.hpp"
#include <functional>
#include <filesystem>
#include <memory>
#include <string>

namespace io
{

    /**
     * @brief Records solver iterations to HDF5 for visualization in Paraview.
     *
     * This class provides optional iteration recording that can be enabled/disabled
     * for performance benchmarking vs visualization needs.
     *
     * Usage:
     *   IterationRecorder recorder(config, outputPath);
     *   recorder.enable(saveInterval);  // Save every N iterations
     *
     *   solver.setProgressCallback([&recorder](int iter, double residual, const Grid& grid) {
     *       recorder.record(iter, residual, grid);
     *   });
     *
     *   solver.solve(grid);
     *   recorder.finalize();  // Close HDF5 and write XDMF
     */
    class IterationRecorder
    {
    public:
        /// @brief Construct a recorder (disabled by default)
        /// @param config Solver configuration (for grid dimensions in output)
        /// @param basePath Base path for output files (will create .h5 and .xdmf)
        /// @param mpiRank MPI rank (optional, for parallel recording)
        /// @param mpiSize MPI size (optional)
        IterationRecorder(
            const poisson::Config &config,
            const std::filesystem::path &basePath,
            int mpiRank = 0,
            int mpiSize = 1);

        /// @brief Destructor - ensures files are closed
        ~IterationRecorder();

        // Non-copyable
        IterationRecorder(const IterationRecorder &) = delete;
        IterationRecorder &operator=(const IterationRecorder &) = delete;

        // Movable
        IterationRecorder(IterationRecorder &&) noexcept = default;
        IterationRecorder &operator=(IterationRecorder &&) noexcept = default;

        /// @brief Enable iteration recording
        /// @param saveInterval Save every N iterations (1 = every iteration, 10 = every 10th, etc.)
        /// @return true if successfully opened output file
        [[nodiscard]] bool enable(int saveInterval = 1);

        /// @brief Disable iteration recording (no-op after this)
        void disable();

        /// @brief Check if recording is enabled
        [[nodiscard]] bool isEnabled() const noexcept { return enabled_; }

        /// @brief Get the save interval
        [[nodiscard]] int saveInterval() const noexcept { return saveInterval_; }

        /// @brief Record an iteration (called from solver callback)
        /// @param iteration Current iteration number
        /// @param residual Current residual value
        /// @param grid Current grid state
        /// @return true if recorded successfully (or skipped due to interval)
        bool record(int iteration, double residual, const poisson::Grid &grid);

        /// @brief Finalize recording - close HDF5 file and write XDMF
        /// @return true if finalized successfully
        bool finalize();

        /// @brief Get the number of iterations recorded
        [[nodiscard]] std::size_t recordedCount() const noexcept { return recordedCount_; }

        /// @brief Get last error message
        [[nodiscard]] const std::string &lastError() const noexcept { return lastError_; }

        /// @brief Check if HDF5 support is available
        [[nodiscard]] static bool isAvailable() noexcept { return HDF5Writer::isAvailable(); }

        /// @brief Create a progress callback suitable for Solver::setProgressCallback
        /// @return A callback function that records iterations
        [[nodiscard]] std::function<void(int, double, const poisson::Grid &)> createCallback();

    private:
        poisson::Config config_;
        std::filesystem::path basePath_;
        int mpiRank_;
        int mpiSize_;

        bool enabled_ = false;
        int saveInterval_ = 1;
        std::size_t recordedCount_ = 0;
        std::string lastError_;

        std::unique_ptr<HDF5Writer> writer_;
        bool finalized_ = false;
    };

} // namespace io
