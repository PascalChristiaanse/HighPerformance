#pragma once

#include <array>
#include <memory>
#include <cstddef>

namespace poisson
{

    /// Direction enumeration for 2D grid operations
    enum class Direction
    {
        X = 0,
        Y = 1
    };

    /// 2D Grid data structure for the Poisson solver
    /// Manages phi (solution) and source (boundary condition mask) arrays
    class Grid
    {
    public:
        /// Construct a grid with given dimensions (including ghost layers)
        /// @param nx Number of interior cells in X direction
        /// @param ny Number of interior cells in Y direction
        /// @param ghostLayers Number of ghost layers (default 1)
        Grid(int nx, int ny, int ghostLayers = 1);

        /// Destructor
        ~Grid();

        // Move semantics
        Grid(Grid &&other) noexcept;
        Grid &operator=(Grid &&other) noexcept;

        // No copying (large data)
        Grid(const Grid &) = delete;
        Grid &operator=(const Grid &) = delete;

        /// Access phi value at (x, y) - mutable
        [[nodiscard]] double &phi(int x, int y) noexcept;

        /// Access phi value at (x, y) - const
        [[nodiscard]] const double &phi(int x, int y) const noexcept;

        /// Access source mask at (x, y) - mutable
        [[nodiscard]] int &source(int x, int y) noexcept;

        /// Access source mask at (x, y) - const
        [[nodiscard]] const int &source(int x, int y) const noexcept;

        /// Get raw pointer to phi data (for I/O operations)
        [[nodiscard]] double *phiData() noexcept { return phiData_.get(); }
        [[nodiscard]] const double *phiData() const noexcept { return phiData_.get(); }

        /// Get raw pointer to source data
        [[nodiscard]] int *sourceData() noexcept { return sourceData_.get(); }
        [[nodiscard]] const int *sourceData() const noexcept { return sourceData_.get(); }

        /// Total dimension in given direction (including ghost cells)
        [[nodiscard]] int dim(Direction d) const noexcept;
        [[nodiscard]] int dimX() const noexcept { return dims_[0]; }
        [[nodiscard]] int dimY() const noexcept { return dims_[1]; }

        /// Interior dimension (excluding ghost cells)
        [[nodiscard]] int interiorDim(Direction d) const noexcept;
        [[nodiscard]] int interiorDimX() const noexcept { return dims_[0] - 2 * ghostLayers_; }
        [[nodiscard]] int interiorDimY() const noexcept { return dims_[1] - 2 * ghostLayers_; }

        /// Total dimensions array
        [[nodiscard]] const std::array<int, 2> &dims() const noexcept { return dims_; }

        /// Number of ghost layers
        [[nodiscard]] int ghostLayers() const noexcept { return ghostLayers_; }

        /// Total number of cells
        [[nodiscard]] std::size_t totalCells() const noexcept;

        /// Clear all values to zero
        void clear();

        /// Apply a source point (normalized coordinates)
        void applySource(double normX, double normY, double value);

        /// Set subdomain offset for parallel I/O
        void setSubdomainOffset(std::array<int, 2> offset) { subdomainOffset_ = offset; }

        /// Get subdomain offset for parallel I/O
        [[nodiscard]] const std::array<int, 2> &subdomainOffset() const noexcept { return subdomainOffset_; }

    private:
        /// Convert 2D index to 1D index
        [[nodiscard]] std::size_t index(int x, int y) const noexcept;

        std::array<int, 2> dims_;           ///< Total dimensions (with ghost cells)
        int ghostLayers_;                   ///< Number of ghost layers
        std::unique_ptr<double[]> phiData_; ///< Solution values
        std::unique_ptr<int[]> sourceData_; ///< Source mask (1 = fixed source)
        std::array<int, 2> subdomainOffset_{0, 0}; ///< Offset in global grid for parallel I/O
    };

} // namespace poisson
