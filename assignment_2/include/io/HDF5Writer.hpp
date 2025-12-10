#pragma once

#include "io/FileManager.hpp"
#include <vector>
#include <optional>

#ifdef POISSON_HAS_HDF5
#include <hdf5.h>
#endif

namespace io
{

    /// HDF5 + XDMF3 file writer for Paraview visualization
    /// Supports both single snapshots and time series (iteration recording)
    class HDF5Writer final : public FileManager
    {
    public:
        /// Default constructor for serial use
        HDF5Writer();

        /// Constructor for MPI use
        /// @param mpiRank This process's rank
        /// @param mpiSize Total number of processes
        HDF5Writer(int mpiRank, int mpiSize);

        ~HDF5Writer() override;

        // Prevent copying (HDF5 handles)
        HDF5Writer(const HDF5Writer &) = delete;
        HDF5Writer &operator=(const HDF5Writer &) = delete;

        // Allow moving
        HDF5Writer(HDF5Writer &&other) noexcept;
        HDF5Writer &operator=(HDF5Writer &&other) noexcept;

        [[nodiscard]] bool write(
            const poisson::Grid &grid,
            const poisson::Config &config,
            const std::filesystem::path &basePath) override;

        [[nodiscard]] bool write(
            const poisson::Grid &grid,
            const poisson::Config &config,
            const std::filesystem::path &basePath,
            double time,
            int timeStep) override;

        [[nodiscard]] OutputFormat format() const noexcept override
        {
            return OutputFormat::HDF5_XDMF;
        }

        [[nodiscard]] bool supportsParallelIO() const noexcept override
        {
            return true;
        }

        [[nodiscard]] std::string extension() const noexcept override
        {
            return ".h5";
        }

        [[nodiscard]] std::string lastError() const noexcept override
        {
            return lastError_;
        }

        /// Check if HDF5 support is available at compile time
        [[nodiscard]] static bool isAvailable() noexcept;

        // ============== Time Series API ==============

        /// Open an HDF5 file for time series writing (iteration recording)
        /// @param basePath Base path for the HDF5 file
        /// @param config Grid configuration
        /// @return true on success
        [[nodiscard]] bool openTimeSeries(
            const std::filesystem::path &basePath,
            const poisson::Config &config);

        /// Append a time step to the open HDF5 file
        /// @param grid Grid data to write
        /// @param iteration Iteration number (used as time value)
        /// @param residual Current residual (stored as attribute)
        /// @return true on success
        [[nodiscard]] bool appendTimeStep(
            const poisson::Grid &grid,
            int iteration,
            double residual);

        /// Close the time series file and write XDMF
        /// @return true on success
        [[nodiscard]] bool closeTimeSeries();

        /// Check if a time series is currently open
        [[nodiscard]] bool isTimeSeriesOpen() const noexcept { return timeSeriesOpen_; }

        /// Get the number of time steps written
        [[nodiscard]] size_t timeStepCount() const noexcept { return timeSteps_.size(); }

        /// Write XDMF3 companion file for Paraview
        /// @param basePath Base path for files
        /// @param config Grid configuration for dimensions
        /// @param times Optional vector of time values for time series
        [[nodiscard]] bool writeXDMF(
            const std::filesystem::path &basePath,
            const poisson::Config &config,
            const std::vector<double> &times = {});

    private:
#ifdef POISSON_HAS_HDF5
        /// Write grid data to an HDF5 dataset
        [[nodiscard]] bool writeGridData(
            const poisson::Grid &grid,
            const std::string &datasetPath);

        /// Create the mesh group with coordinate arrays
        [[nodiscard]] bool createMeshGroup(const poisson::Config &config);

        hid_t fileId_{H5I_INVALID_HID};
#endif

        std::optional<int> mpiRank_;
        std::optional<int> mpiSize_;
        mutable std::string lastError_;

        // Time series state
        std::filesystem::path timeSeriesPath_;
        poisson::Config timeSeriesConfig_;
        std::vector<double> timeSteps_; // Iteration numbers as "time"
        std::vector<double> residuals_; // Residuals at each step
        bool timeSeriesOpen_{false};
    };

} // namespace io
