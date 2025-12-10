#pragma once

#include <chrono>
#include <string>

namespace poisson
{

    /// High-resolution timer with start/stop/resume functionality
    class Timer
    {
    public:
        using Clock = std::chrono::high_resolution_clock;
        using Duration = std::chrono::duration<double>;
        using TimePoint = Clock::time_point;

        Timer() = default;

        /// Start the timer (resets accumulated time)
        void start() noexcept;

        /// Stop the timer
        void stop() noexcept;

        /// Resume the timer without resetting
        void resume() noexcept;

        /// Reset the timer
        void reset() noexcept;

        /// Get elapsed time as duration
        [[nodiscard]] Duration elapsed() const noexcept;

        /// Get elapsed time in seconds
        [[nodiscard]] double elapsedSeconds() const noexcept;

        /// Check if timer is running
        [[nodiscard]] bool isRunning() const noexcept { return running_; }

        /// Get formatted string representation
        [[nodiscard]] std::string toString() const;

        /// RAII scope guard for automatic timing
        class ScopedTimer
        {
        public:
            explicit ScopedTimer(Timer &timer) : timer_(timer) { timer_.start(); }
            ~ScopedTimer() { timer_.stop(); }

            // Non-copyable, non-movable
            ScopedTimer(const ScopedTimer &) = delete;
            ScopedTimer &operator=(const ScopedTimer &) = delete;

        private:
            Timer &timer_;
        };

        /// Create a scoped timer that starts now and stops on destruction
        [[nodiscard]] ScopedTimer scoped() { return ScopedTimer(*this); }

    private:
        TimePoint startTime_{};
        Duration accumulated_{Duration::zero()};
        bool running_{false};
    };

} // namespace poisson
