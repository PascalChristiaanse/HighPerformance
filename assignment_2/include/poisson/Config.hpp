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

    /// Solver method enumeration
    enum class SolverMethod
    {
        GaussSeidel, ///< Red-black Gauss-Seidel (default)
        SOR,         ///< Successive Over-Relaxation
        CG           ///< Conjugate Gradient
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

        // === New solver options (exercises 1.2.x) ===

        /// Solver method (Gauss-Seidel, SOR, CG)
        [[nodiscard]] SolverMethod solverMethod() const noexcept { return solverMethod_; }

        /// SOR relaxation parameter omega (1.0 = standard GS, ~1.9-1.99 for SOR)
        [[nodiscard]] double omega() const noexcept { return omega_; }

        /// How often to check convergence (1 = every iteration, 10 = every 10th)
        [[nodiscard]] int errorCheckInterval() const noexcept { return errorCheckInterval_; }

        /// Number of red-black sweeps between boundary exchanges
        [[nodiscard]] int sweepsPerExchange() const noexcept { return sweepsPerExchange_; }

        /// Use optimized inner loop without parity check (exercise 1.2.9)
        [[nodiscard]] bool useOptimizedLoop() const noexcept { return useOptimizedLoop_; }

        /// Verbose timing output
        [[nodiscard]] bool verboseTiming() const noexcept { return verboseTiming_; }

        /// Timing CSV output path (empty = no CSV output)
        [[nodiscard]] const std::filesystem::path &timingOutputPath() const noexcept { return timingOutputPath_; }

        // Builder-style setters
        Config &setGridSize(int nx, int ny);
        Config &setPrecisionGoal(double goal);
        Config &setMaxIterations(int maxIter);
        Config &addSource(double x, double y, double value);

        // New setters for solver options
        Config &setSolverMethod(SolverMethod method);
        Config &setOmega(double omega);
        Config &setErrorCheckInterval(int interval);
        Config &setSweepsPerExchange(int sweeps);
        Config &setUseOptimizedLoop(bool useOptimized);
        Config &setVerboseTiming(bool verbose);
        Config &setTimingOutputPath(const std::filesystem::path &path);

        // Utility methods
        [[nodiscard]] static SolverMethod parseMethod(const std::string &str);
        [[nodiscard]] static std::string methodToString(SolverMethod method);

    private:
        std::array<int, 2> gridSize_{100, 100};
        double precisionGoal_{1e-6};
        int maxIterations_{10000};
        std::vector<SourcePoint> sources_;

        // New solver options
        SolverMethod solverMethod_{SolverMethod::GaussSeidel};
        double omega_{1.0};
        int errorCheckInterval_{1};
        int sweepsPerExchange_{1};
        bool useOptimizedLoop_{false};
        bool verboseTiming_{false};
        std::filesystem::path timingOutputPath_;
    };

} // namespace poisson
