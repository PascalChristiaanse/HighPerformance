#include "poisson/Config.hpp"
#include <fstream>
#include <sstream>

namespace poisson
{

    Config Config::fromFile(const std::filesystem::path &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open config file: " + path.string());
        }

        Config config;
        std::string line;

        // Read grid dimensions
        if (std::getline(file, line))
        {
            int nx;
            if (std::sscanf(line.c_str(), "nx: %d", &nx) == 1)
            {
                config.gridSize_[0] = nx;
            }
        }

        if (std::getline(file, line))
        {
            int ny;
            if (std::sscanf(line.c_str(), "ny: %d", &ny) == 1)
            {
                config.gridSize_[1] = ny;
            }
        }

        // Read precision goal
        if (std::getline(file, line))
        {
            double precision;
            if (std::sscanf(line.c_str(), "precision goal: %lf", &precision) == 1)
            {
                config.precisionGoal_ = precision;
            }
        }

        // Read max iterations
        if (std::getline(file, line))
        {
            int maxIter;
            if (std::sscanf(line.c_str(), "max iterations: %d", &maxIter) == 1)
            {
                config.maxIterations_ = maxIter;
            }
        }

        // Read source points
        while (std::getline(file, line))
        {
            double x, y, value;
            if (std::sscanf(line.c_str(), "source: %lf %lf %lf", &x, &y, &value) == 3)
            {
                config.addSource(x, y, value);
            }
        }

        return config;
    }

    Config &Config::setGridSize(int nx, int ny)
    {
        gridSize_[0] = nx;
        gridSize_[1] = ny;
        return *this;
    }

    Config &Config::setPrecisionGoal(double goal)
    {
        precisionGoal_ = goal;
        return *this;
    }

    Config &Config::setMaxIterations(int maxIter)
    {
        maxIterations_ = maxIter;
        return *this;
    }

    Config &Config::addSource(double x, double y, double value)
    {
        sources_.push_back({x, y, value});
        return *this;
    }

} // namespace poisson
