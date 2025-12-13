#pragma once

#include "poisson/Grid.hpp"
#include "poisson/Config.hpp"
#include <filesystem>
#include <string>
#include <memory>
#include <optional>
#include <mpi.h>

namespace io
{

    /// Output format enumeration
    enum class OutputFormat
    {
        Ascii,    ///< Plain text format (x y value)
        HDF5_XDMF ///< HDF5 data + XDMF3 metadata for Paraview
    };

    /// Abstract base class for file I/O operations
    class FileManager
    {
    public:
        virtual ~FileManager() = default;

        /// Write grid data to file
        /// @param grid Grid to write
        /// @param config Configuration for metadata
        /// @param basePath Base path for output files (without extension)
        /// @return true on success, false on failure
        [[nodiscard]] virtual bool write(
            const poisson::Grid &grid,
            const poisson::Config &config,
            const std::filesystem::path &basePath) = 0;

        /// Write grid data with time information (for time series)
        /// @param grid Grid to write
        /// @param config Configuration for metadata
        /// @param basePath Base path for output files
        /// @param time Current simulation time
        /// @param timeStep Time step index
        /// @return true on success, false on failure
        [[nodiscard]] virtual bool write(
            const poisson::Grid &grid,
            const poisson::Config &config,
            const std::filesystem::path &basePath,
            double time,
            int timeStep) = 0;

        /// Get the output format type
        [[nodiscard]] virtual OutputFormat format() const noexcept = 0;

        /// Check if this writer supports parallel I/O
        [[nodiscard]] virtual bool supportsParallelIO() const noexcept = 0;

        /// Get the file extension for this format
        [[nodiscard]] virtual std::string extension() const noexcept = 0;

        /// Get the last error message
        [[nodiscard]] virtual std::string lastError() const noexcept = 0;

        /// Factory method to create a FileManager (separate files per rank)
        /// @param format Desired output format
        /// @param mpiRank Optional MPI rank for parallel file naming
        /// @param mpiSize Optional MPI size for parallel I/O
        [[nodiscard]] static std::unique_ptr<FileManager> create(
            OutputFormat format,
            std::optional<int> mpiRank = std::nullopt,
            std::optional<int> mpiSize = std::nullopt);

        /// Factory method to create a FileManager with parallel I/O (single shared file)
        /// @param format Desired output format
        /// @param mpiRank MPI rank
        /// @param mpiSize MPI size
        /// @param comm MPI communicator for collective I/O
        [[nodiscard]] static std::unique_ptr<FileManager> createParallel(
            OutputFormat format,
            int mpiRank,
            int mpiSize,
            MPI_Comm comm);
    };

} // namespace io
