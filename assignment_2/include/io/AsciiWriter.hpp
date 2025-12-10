#pragma once

#include "io/FileManager.hpp"
#include <optional>

namespace io
{

    /// ASCII text file writer for grid data
    /// Outputs in format: "x y value" per line
    class AsciiWriter final : public FileManager
    {
    public:
        /// Default constructor for serial use
        AsciiWriter() = default;

        /// Constructor for MPI use (adds rank to filename)
        explicit AsciiWriter(int mpiRank);

        ~AsciiWriter() override = default;

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
            return OutputFormat::Ascii;
        }

        [[nodiscard]] bool supportsParallelIO() const noexcept override
        {
            return false;
        }

        [[nodiscard]] std::string extension() const noexcept override
        {
            return ".dat";
        }

        [[nodiscard]] std::string lastError() const noexcept override
        {
            return lastError_;
        }

    private:
        /// Build the output filename
        [[nodiscard]] std::filesystem::path buildFilename(
            const std::filesystem::path &basePath,
            std::optional<int> timeStep = std::nullopt) const;

        /// Write the grid data to file
        [[nodiscard]] bool writeToFile(
            const poisson::Grid &grid,
            const std::filesystem::path &filepath);

        std::optional<int> mpiRank_;
        mutable std::string lastError_;
    };

} // namespace io
