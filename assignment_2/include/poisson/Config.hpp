#pragma once

#include <array>
#include <filesystem>
#include <vector>
#include <string>
#include <stdexcept>

namespace poisson
{

    /// Source point in the domain (normalized coordinates)
    struct SourcePoint
    {
        double x;     ///< x-coordinate [0, 1]
        double y;     ///< y-coordinate [0, 1]
        double value; ///< Source value
    };

    /// Configuration parameters for the Poisson solver
    class Config
    {
    public:
        /// Load configuration from input file
        /// @param path Path to input.dat file
        /// @throws std::runtime_error if file cannot be read
        static Config fromFile(const std::filesystem::path &path);

        /// Default constructor with sensible defaults
        Config() = default;

        // Grid dimensions
        [[nodiscard]] int nx() const noexcept { return gridSize_[0]; }
        [[nodiscard]] int ny() const noexcept { return gridSize_[1]; }
        [[nodiscard]] const std::array<int, 2> &gridSize() const noexcept { return gridSize_; }

        // Solver parameters
        [[nodiscard]] double precisionGoal() const noexcept { return precisionGoal_; }
        [[nodiscard]] int maxIterations() const noexcept { return maxIterations_; }

        // Source points
        [[nodiscard]] const std::vector<SourcePoint> &sources() const noexcept { return sources_; }

        // Builder-style setters
        Config &setGridSize(int nx, int ny);
        Config &setPrecisionGoal(double goal);
        Config &setMaxIterations(int maxIter);
        Config &addSource(double x, double y, double value);

    private:
        std::array<int, 2> gridSize_{100, 100};
        double precisionGoal_{1e-6};
        int maxIterations_{10000};
        std::vector<SourcePoint> sources_;
    };

} // namespace poisson
