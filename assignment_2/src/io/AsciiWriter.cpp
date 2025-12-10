#include "io/AsciiWriter.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>

namespace io
{

    AsciiWriter::AsciiWriter(int mpiRank) : mpiRank_(mpiRank) {}

    bool AsciiWriter::write(
        const poisson::Grid &grid,
        const poisson::Config & /*config*/,
        const std::filesystem::path &basePath)
    {
        auto filepath = buildFilename(basePath);
        return writeToFile(grid, filepath);
    }

    bool AsciiWriter::write(
        const poisson::Grid &grid,
        const poisson::Config & /*config*/,
        const std::filesystem::path &basePath,
        double /*time*/,
        int timeStep)
    {
        auto filepath = buildFilename(basePath, timeStep);
        return writeToFile(grid, filepath);
    }

    std::filesystem::path AsciiWriter::buildFilename(
        const std::filesystem::path &basePath,
        std::optional<int> timeStep) const
    {
        std::ostringstream oss;
        oss << basePath.stem().string();

        if (timeStep.has_value())
        {
            oss << "_" << std::setfill('0') << std::setw(6) << *timeStep;
        }

        if (mpiRank_.has_value())
        {
            oss << "_rank" << std::setfill('0') << std::setw(4) << *mpiRank_;
        }

        oss << extension();

        auto result = basePath.parent_path() / oss.str();
        return result;
    }

    bool AsciiWriter::writeToFile(
        const poisson::Grid &grid,
        const std::filesystem::path &filepath)
    {
        std::ofstream file(filepath);
        if (!file.is_open())
        {
            lastError_ = "Failed to open file: " + filepath.string();
            return false;
        }

        const int ghost = grid.ghostLayers();
        const int xEnd = grid.dimX() - ghost;
        const int yEnd = grid.dimY() - ghost;

        // Write interior cells only (skip ghost layers)
        for (int x = ghost; x < xEnd; ++x)
        {
            for (int y = ghost; y < yEnd; ++y)
            {
                file << x << " " << y << " "
                     << std::setprecision(10) << grid.phi(x, y) << "\n";
            }
        }

        if (!file.good())
        {
            lastError_ = "Error writing to file: " + filepath.string();
            return false;
        }

        return true;
    }

} // namespace io
