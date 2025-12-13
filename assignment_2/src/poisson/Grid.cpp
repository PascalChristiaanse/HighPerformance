#include "poisson/Grid.hpp"
#include <algorithm>
#include <cstring>

namespace poisson
{

    Grid::Grid(int nx, int ny, int ghostLayers)
        : dims_{nx + 2 * ghostLayers, ny + 2 * ghostLayers}, ghostLayers_(ghostLayers), 
          phiData_(std::make_unique<double[]>(totalCells())), 
          sourceData_(std::make_unique<int[]>(totalCells()))
    {
        clear();
    }

    Grid::~Grid() = default;

    Grid::Grid(Grid &&other) noexcept
        : dims_(other.dims_), ghostLayers_(other.ghostLayers_), phiData_(std::move(other.phiData_)), sourceData_(std::move(other.sourceData_))
    {
        other.dims_ = {0, 0};
        other.ghostLayers_ = 0;
    }

    Grid &Grid::operator=(Grid &&other) noexcept
    {
        if (this != &other)
        {
            dims_ = other.dims_;
            ghostLayers_ = other.ghostLayers_;
            phiData_ = std::move(other.phiData_);
            sourceData_ = std::move(other.sourceData_);
            other.dims_ = {0, 0};
            other.ghostLayers_ = 0;
        }
        return *this;
    }

    double &Grid::phi(int x, int y) noexcept
    {
        return phiData_[index(x, y)];
    }

    const double &Grid::phi(int x, int y) const noexcept
    {
        return phiData_[index(x, y)];
    }

    int &Grid::source(int x, int y) noexcept
    {
        return sourceData_[index(x, y)];
    }

    const int &Grid::source(int x, int y) const noexcept
    {
        return sourceData_[index(x, y)];
    }

    int Grid::dim(Direction d) const noexcept
    {
        return dims_[static_cast<int>(d)];
    }

    int Grid::interiorDim(Direction d) const noexcept
    {
        return dims_[static_cast<int>(d)] - 2 * ghostLayers_;
    }

    std::size_t Grid::totalCells() const noexcept
    {
        return static_cast<std::size_t>(dims_[0]) * static_cast<std::size_t>(dims_[1]);
    }

    void Grid::clear()
    {
        std::memset(phiData_.get(), 0, totalCells() * sizeof(double));
        std::memset(sourceData_.get(), 0, totalCells() * sizeof(int));
    }

    void Grid::applySource(double normX, double normY, double value)
    {
        // Convert normalized [0,1] coordinates to grid indices
        int x = static_cast<int>(normX * interiorDimX()) + ghostLayers_;
        int y = static_cast<int>(normY * interiorDimY()) + ghostLayers_;

        // Clamp to valid range
        x = std::clamp(x, ghostLayers_, dims_[0] - ghostLayers_ - 1);
        y = std::clamp(y, ghostLayers_, dims_[1] - ghostLayers_ - 1);

        phi(x, y) = value;
        source(x, y) = 1;
    }

    std::size_t Grid::index(int x, int y) const noexcept
    {
        return static_cast<std::size_t>(x) * static_cast<std::size_t>(dims_[1]) + static_cast<std::size_t>(y);
    }

} // namespace poisson
