#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <numeric>

namespace poisson
{

    /// Telemetry data for a single iteration
    struct IterationTelemetry
    {
        int iteration{0};
        double residual{0.0};
        double stepTime{0.0};      ///< Time for compute step (seconds)
        double exchangeTime{0.0};  ///< Time for border exchange (seconds)
        double reductionTime{0.0}; ///< Time for MPI_Allreduce (seconds)
        double totalIterTime{0.0}; ///< Total iteration time (seconds)
    };

    /// Aggregated telemetry data for the entire solve
    class SolveTelemetry
    {
    public:
        /// Add telemetry for one iteration
        void recordIteration(const IterationTelemetry &telemetry)
        {
            iterations_.push_back(telemetry);
        }

        /// Set final solve statistics
        void setFinalStats(double totalTime, int totalIterations, bool converged, double finalResidual)
        {
            totalTime_ = totalTime;
            totalIterations_ = totalIterations;
            converged_ = converged;
            finalResidual_ = finalResidual;
        }

        /// Get all iteration telemetry
        [[nodiscard]] const std::vector<IterationTelemetry> &iterations() const noexcept { return iterations_; }

        /// Get total solve time
        [[nodiscard]] double totalTime() const noexcept { return totalTime_; }

        /// Get total iterations
        [[nodiscard]] int totalIterations() const noexcept { return totalIterations_; }

        /// Whether solver converged
        [[nodiscard]] bool converged() const noexcept { return converged_; }

        /// Final residual
        [[nodiscard]] double finalResidual() const noexcept { return finalResidual_; }

        /// Calculate total compute time
        [[nodiscard]] double totalComputeTime() const noexcept
        {
            return std::accumulate(iterations_.begin(), iterations_.end(), 0.0,
                                   [](double sum, const IterationTelemetry &t)
                                   { return sum + t.stepTime; });
        }

        /// Calculate total exchange time
        [[nodiscard]] double totalExchangeTime() const noexcept
        {
            return std::accumulate(iterations_.begin(), iterations_.end(), 0.0,
                                   [](double sum, const IterationTelemetry &t)
                                   { return sum + t.exchangeTime; });
        }

        /// Calculate total reduction time
        [[nodiscard]] double totalReductionTime() const noexcept
        {
            return std::accumulate(iterations_.begin(), iterations_.end(), 0.0,
                                   [](double sum, const IterationTelemetry &t)
                                   { return sum + t.reductionTime; });
        }

        /// Calculate average time per iteration
        [[nodiscard]] double avgIterationTime() const noexcept
        {
            if (iterations_.empty())
                return 0.0;
            return totalTime_ / static_cast<double>(iterations_.size());
        }

        /// Calculate average compute time per iteration
        [[nodiscard]] double avgComputeTime() const noexcept
        {
            if (iterations_.empty())
                return 0.0;
            return totalComputeTime() / static_cast<double>(iterations_.size());
        }

        /// Calculate average exchange time per iteration
        [[nodiscard]] double avgExchangeTime() const noexcept
        {
            if (iterations_.empty())
                return 0.0;
            return totalExchangeTime() / static_cast<double>(iterations_.size());
        }

        /// Calculate average reduction time per iteration
        [[nodiscard]] double avgReductionTime() const noexcept
        {
            if (iterations_.empty())
                return 0.0;
            return totalReductionTime() / static_cast<double>(iterations_.size());
        }

        /// Alpha coefficient: startup/initialization overhead (intercept in t(n) = α + βn)
        [[nodiscard]] double alpha() const noexcept
        {
            // Simple estimate: total overhead time not scaling with iterations
            // For more accurate: fit linear regression
            return totalTime_ - beta() * totalIterations_;
        }

        /// Beta coefficient: time per iteration (slope in t(n) = α + βn)
        [[nodiscard]] double beta() const noexcept
        {
            return avgIterationTime();
        }

        /// Compute fraction of total time
        [[nodiscard]] double computeFraction() const noexcept
        {
            if (totalTime_ <= 0.0)
                return 0.0;
            return totalComputeTime() / totalTime_;
        }

        /// Communication fraction of total time
        [[nodiscard]] double commFraction() const noexcept
        {
            if (totalTime_ <= 0.0)
                return 0.0;
            return (totalExchangeTime() + totalReductionTime()) / totalTime_;
        }

        /// Write telemetry to CSV file
        bool writeCSV(const std::filesystem::path &path) const
        {
            std::ofstream file(path);
            if (!file.is_open())
            {
                return false;
            }

            // Header
            file << "iteration,residual,step_time,exchange_time,reduction_time,total_iter_time\n";

            // Data rows
            for (const auto &iter : iterations_)
            {
                file << iter.iteration << ","
                     << std::scientific << std::setprecision(12) << iter.residual << ","
                     << std::fixed << std::setprecision(9) << iter.stepTime << ","
                     << iter.exchangeTime << ","
                     << iter.reductionTime << ","
                     << iter.totalIterTime << "\n";
            }

            return true;
        }

        /// Print verbose timing summary to stream
        void printSummary(std::ostream &out, int gridNx, int gridNy, int numProcs,
                          int procsX, int procsY, const std::string &solverMethod,
                          double omega = 1.0) const
        {
            out << "\n╔══════════════════════════════════════════════════════════════════╗\n";
            out << "║                    TIMING & TELEMETRY SUMMARY                    ║\n";
            out << "╠══════════════════════════════════════════════════════════════════╣\n";

            // Problem configuration
            out << "║ Problem Configuration:                                           ║\n";
            out << "║   Grid size (g):        " << std::setw(6) << gridNx << " x " << std::setw(6) << gridNy
                << std::setw(24) << "║\n";
            out << "║   Processors (p):       " << std::setw(6) << numProcs
                << " (" << procsX << " x " << procsY << " topology)"
                << std::setw(17 - std::to_string(procsX).length() - std::to_string(procsY).length()) << "║\n";
            out << "║   Solver method:        " << std::setw(20) << std::left << solverMethod << std::right
                << std::setw(21) << "║\n";
            if (solverMethod == "SOR" || solverMethod == "sor")
            {
                out << "║   Omega (ω):            " << std::setw(20) << std::fixed << std::setprecision(4) << omega
                    << std::setw(21) << "║\n";
            }

            out << "╠══════════════════════════════════════════════════════════════════╣\n";

            // Convergence results
            out << "║ Convergence Results:                                             ║\n";
            out << "║   Iterations (n):       " << std::setw(20) << totalIterations_
                << std::setw(21) << "║\n";
            out << "║   Converged:            " << std::setw(20) << (converged_ ? "YES" : "NO")
                << std::setw(21) << "║\n";
            out << "║   Final residual:       " << std::setw(20) << std::scientific << std::setprecision(6) << finalResidual_
                << std::setw(21) << "║\n";

            out << "╠══════════════════════════════════════════════════════════════════╣\n";

            // Timing breakdown
            out << "║ Timing Breakdown:                                                ║\n";
            out << std::fixed << std::setprecision(6);
            out << "║   Total time (t):       " << std::setw(14) << totalTime_ << " s"
                << std::setw(25) << "║\n";
            out << "║   Compute time:         " << std::setw(14) << totalComputeTime() << " s ("
                << std::setw(5) << std::setprecision(1) << (computeFraction() * 100.0) << "%)"
                << std::setw(15) << "║\n";
            out << "║   Exchange time:        " << std::setw(14) << std::setprecision(6) << totalExchangeTime() << " s ("
                << std::setw(5) << std::setprecision(1) << (totalExchangeTime() / totalTime_ * 100.0) << "%)"
                << std::setw(15) << "║\n";
            out << "║   Reduction time:       " << std::setw(14) << std::setprecision(6) << totalReductionTime() << " s ("
                << std::setw(5) << std::setprecision(1) << (totalReductionTime() / totalTime_ * 100.0) << "%)"
                << std::setw(15) << "║\n";

            out << "╠══════════════════════════════════════════════════════════════════╣\n";

            // Per-iteration statistics
            out << "║ Per-Iteration Statistics:                                        ║\n";
            out << std::setprecision(9);
            out << "║   Avg iteration time:   " << std::setw(14) << avgIterationTime() << " s"
                << std::setw(25) << "║\n";
            out << "║   Avg compute time:     " << std::setw(14) << avgComputeTime() << " s"
                << std::setw(25) << "║\n";
            out << "║   Avg exchange time:    " << std::setw(14) << avgExchangeTime() << " s"
                << std::setw(25) << "║\n";
            out << "║   Avg reduction time:   " << std::setw(14) << avgReductionTime() << " s"
                << std::setw(25) << "║\n";

            out << "╠══════════════════════════════════════════════════════════════════╣\n";

            // Alpha-Beta analysis (t(n) = α + βn)
            out << "║ Linear Model: t(n) = α + β·n                                     ║\n";
            out << std::setprecision(6);
            out << "║   α (overhead):         " << std::setw(14) << alpha() << " s"
                << std::setw(25) << "║\n";
            out << "║   β (time/iteration):   " << std::setw(14) << beta() << " s"
                << std::setw(25) << "║\n";

            out << "╚══════════════════════════════════════════════════════════════════╝\n";
        }

        /// Clear all data
        void clear()
        {
            iterations_.clear();
            totalTime_ = 0.0;
            totalIterations_ = 0;
            converged_ = false;
            finalResidual_ = 0.0;
        }

    private:
        std::vector<IterationTelemetry> iterations_;
        double totalTime_{0.0};
        int totalIterations_{0};
        bool converged_{false};
        double finalResidual_{0.0};
    };

} // namespace poisson
