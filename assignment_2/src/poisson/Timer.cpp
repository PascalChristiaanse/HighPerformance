#include "poisson/Timer.hpp"
#include <sstream>
#include <iomanip>

namespace poisson
{

    void Timer::start() noexcept
    {
        if (!running_)
        {
            accumulated_ = Duration::zero();
            startTime_ = Clock::now();
            running_ = true;
        }
    }

    void Timer::stop() noexcept
    {
        if (running_)
        {
            accumulated_ += Clock::now() - startTime_;
            running_ = false;
        }
    }

    void Timer::resume() noexcept
    {
        if (!running_)
        {
            startTime_ = Clock::now();
            running_ = true;
        }
    }

    void Timer::reset() noexcept
    {
        accumulated_ = Duration::zero();
        running_ = false;
    }

    Timer::Duration Timer::elapsed() const noexcept
    {
        if (running_)
        {
            return accumulated_ + (Clock::now() - startTime_);
        }
        return accumulated_;
    }

    double Timer::elapsedSeconds() const noexcept
    {
        return elapsed().count();
    }

    std::string Timer::toString() const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << elapsedSeconds() << " s";
        return oss.str();
    }

} // namespace poisson
