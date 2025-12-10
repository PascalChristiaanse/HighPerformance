#include "io/HDF5Writer.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace io
{

    // ============== Static methods ==============

    bool HDF5Writer::isAvailable() noexcept
    {
#ifdef POISSON_HAS_HDF5
        return true;
#else
        return false;
#endif
    }

    // ============== Constructors/Destructor ==============

    HDF5Writer::HDF5Writer()
        : mpiRank_(std::nullopt), mpiSize_(std::nullopt)
    {
    }

    HDF5Writer::HDF5Writer(int mpiRank, int mpiSize)
        : mpiRank_(mpiRank), mpiSize_(mpiSize)
    {
    }

    HDF5Writer::~HDF5Writer()
    {
        if (timeSeriesOpen_)
        {
            (void)closeTimeSeries();
        }
    }

    HDF5Writer::HDF5Writer(HDF5Writer &&other) noexcept
        : mpiRank_(other.mpiRank_), mpiSize_(other.mpiSize_), lastError_(std::move(other.lastError_)), timeSeriesPath_(std::move(other.timeSeriesPath_)), timeSeriesConfig_(std::move(other.timeSeriesConfig_)), timeSteps_(std::move(other.timeSteps_)), residuals_(std::move(other.residuals_)), timeSeriesOpen_(other.timeSeriesOpen_)
#ifdef POISSON_HAS_HDF5
          ,
          fileId_(other.fileId_)
#endif
    {
        other.timeSeriesOpen_ = false;
#ifdef POISSON_HAS_HDF5
        other.fileId_ = H5I_INVALID_HID;
#endif
    }

    HDF5Writer &HDF5Writer::operator=(HDF5Writer &&other) noexcept
    {
        if (this != &other)
        {
            if (timeSeriesOpen_)
            {
                (void)closeTimeSeries();
            }

            mpiRank_ = other.mpiRank_;
            mpiSize_ = other.mpiSize_;
            lastError_ = std::move(other.lastError_);
            timeSeriesPath_ = std::move(other.timeSeriesPath_);
            timeSeriesConfig_ = std::move(other.timeSeriesConfig_);
            timeSteps_ = std::move(other.timeSteps_);
            residuals_ = std::move(other.residuals_);
            timeSeriesOpen_ = other.timeSeriesOpen_;

#ifdef POISSON_HAS_HDF5
            fileId_ = other.fileId_;
            other.fileId_ = H5I_INVALID_HID;
#endif
            other.timeSeriesOpen_ = false;
        }
        return *this;
    }

    // ============== Single snapshot write ==============

    bool HDF5Writer::write(
        const poisson::Grid &grid,
        const poisson::Config &config,
        const std::filesystem::path &basePath)
    {
#ifdef POISSON_HAS_HDF5
        auto h5Path = basePath;
        h5Path.replace_extension(".h5");

        // Create file
        hid_t fileId = H5Fcreate(h5Path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (fileId < 0)
        {
            lastError_ = "Failed to create HDF5 file: " + h5Path.string();
            return false;
        }

        // Create /Data group
        hid_t dataGroup = H5Gcreate2(fileId, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataGroup < 0)
        {
            lastError_ = "Failed to create /Data group";
            H5Fclose(fileId);
            return false;
        }

        // Write phi dataset
        const int nx = grid.interiorDimX();
        const int ny = grid.interiorDimY();
        hsize_t dims[2] = {static_cast<hsize_t>(nx), static_cast<hsize_t>(ny)};

        hid_t dataspace = H5Screate_simple(2, dims, nullptr);
        hid_t dataset = H5Dcreate2(dataGroup, "phi", H5T_NATIVE_DOUBLE, dataspace,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Copy interior data to contiguous buffer
        std::vector<double> buffer(nx * ny);
        const int ghost = grid.ghostLayers();
        for (int x = 0; x < nx; ++x)
        {
            for (int y = 0; y < ny; ++y)
            {
                buffer[x * ny + y] = grid.phi(x + ghost, y + ghost);
            }
        }

        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

        // Add attributes
        hsize_t attrDims[1] = {1};
        hid_t attrSpace = H5Screate_simple(1, attrDims, nullptr);

        int nxAttr = config.nx();
        hid_t nxAttrId = H5Acreate2(dataset, "nx", H5T_NATIVE_INT, attrSpace, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(nxAttrId, H5T_NATIVE_INT, &nxAttr);
        H5Aclose(nxAttrId);

        int nyAttr = config.ny();
        hid_t nyAttrId = H5Acreate2(dataset, "ny", H5T_NATIVE_INT, attrSpace, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(nyAttrId, H5T_NATIVE_INT, &nyAttr);
        H5Aclose(nyAttrId);

        H5Sclose(attrSpace);
        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Gclose(dataGroup);
        H5Fclose(fileId);

        // Write XDMF companion file
        return writeXDMF(basePath, config, {});

#else
        (void)grid;
        (void)config;
        (void)basePath;
        lastError_ = "HDF5 support not compiled. Rebuild with -DPOISSON_ENABLE_HDF5=ON";
        return false;
#endif
    }

    bool HDF5Writer::write(
        const poisson::Grid &grid,
        const poisson::Config &config,
        const std::filesystem::path &basePath,
        double time,
        int timeStep)
    {
#ifdef POISSON_HAS_HDF5
        // For single time step, open, write, close
        if (!openTimeSeries(basePath, config))
        {
            return false;
        }
        if (!appendTimeStep(grid, timeStep, 0.0))
        {
            closeTimeSeries();
            return false;
        }
        // Override the time value
        timeSteps_.back() = time;
        return closeTimeSeries();
#else
        (void)grid;
        (void)config;
        (void)basePath;
        (void)time;
        (void)timeStep;
        lastError_ = "HDF5 support not compiled. Rebuild with -DPOISSON_ENABLE_HDF5=ON";
        return false;
#endif
    }

    // ============== Time Series API ==============

    bool HDF5Writer::openTimeSeries(
        const std::filesystem::path &basePath,
        const poisson::Config &config)
    {
#ifdef POISSON_HAS_HDF5
        if (timeSeriesOpen_)
        {
            lastError_ = "Time series already open. Close it first.";
            return false;
        }

        timeSeriesPath_ = basePath;
        timeSeriesPath_.replace_extension(".h5");
        timeSeriesConfig_ = config;
        timeSteps_.clear();
        residuals_.clear();

        // Create file
        fileId_ = H5Fcreate(timeSeriesPath_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (fileId_ < 0)
        {
            lastError_ = "Failed to create HDF5 file: " + timeSeriesPath_.string();
            return false;
        }

        // Create /Time group for time series data
        hid_t timeGroup = H5Gcreate2(fileId_, "/Time", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (timeGroup < 0)
        {
            lastError_ = "Failed to create /Time group";
            H5Fclose(fileId_);
            fileId_ = H5I_INVALID_HID;
            return false;
        }
        H5Gclose(timeGroup);

        // Create /Mesh group with coordinates (optional, for reference)
        if (!createMeshGroup(config))
        {
            H5Fclose(fileId_);
            fileId_ = H5I_INVALID_HID;
            return false;
        }

        timeSeriesOpen_ = true;
        return true;

#else
        (void)basePath;
        (void)config;
        lastError_ = "HDF5 support not compiled. Rebuild with -DPOISSON_ENABLE_HDF5=ON";
        return false;
#endif
    }

    bool HDF5Writer::appendTimeStep(
        const poisson::Grid &grid,
        int iteration,
        double residual)
    {
#ifdef POISSON_HAS_HDF5
        if (!timeSeriesOpen_)
        {
            lastError_ = "No time series open. Call openTimeSeries first.";
            return false;
        }

        // Create step group name
        std::ostringstream groupName;
        groupName << "/Time/step_" << std::setfill('0') << std::setw(6) << timeSteps_.size();

        // Create step group
        hid_t stepGroup = H5Gcreate2(fileId_, groupName.str().c_str(),
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (stepGroup < 0)
        {
            lastError_ = "Failed to create step group: " + groupName.str();
            return false;
        }

        // Write phi dataset
        const int nx = grid.interiorDimX();
        const int ny = grid.interiorDimY();
        hsize_t dims[2] = {static_cast<hsize_t>(nx), static_cast<hsize_t>(ny)};

        hid_t dataspace = H5Screate_simple(2, dims, nullptr);
        hid_t dataset = H5Dcreate2(stepGroup, "phi", H5T_NATIVE_DOUBLE, dataspace,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Copy interior data to contiguous buffer
        std::vector<double> buffer(nx * ny);
        const int ghost = grid.ghostLayers();
        for (int x = 0; x < nx; ++x)
        {
            for (int y = 0; y < ny; ++y)
            {
                buffer[x * ny + y] = grid.phi(x + ghost, y + ghost);
            }
        }

        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

        // Add iteration and residual as attributes
        hsize_t attrDims[1] = {1};
        hid_t attrSpace = H5Screate_simple(1, attrDims, nullptr);

        hid_t iterAttr = H5Acreate2(stepGroup, "iteration", H5T_NATIVE_INT,
                                    attrSpace, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(iterAttr, H5T_NATIVE_INT, &iteration);
        H5Aclose(iterAttr);

        hid_t residAttr = H5Acreate2(stepGroup, "residual", H5T_NATIVE_DOUBLE,
                                     attrSpace, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(residAttr, H5T_NATIVE_DOUBLE, &residual);
        H5Aclose(residAttr);

        H5Sclose(attrSpace);
        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Gclose(stepGroup);

        // Record time step
        timeSteps_.push_back(static_cast<double>(iteration));
        residuals_.push_back(residual);

        return true;

#else
        (void)grid;
        (void)iteration;
        (void)residual;
        lastError_ = "HDF5 support not compiled. Rebuild with -DPOISSON_ENABLE_HDF5=ON";
        return false;
#endif
    }

    bool HDF5Writer::closeTimeSeries()
    {
#ifdef POISSON_HAS_HDF5
        if (!timeSeriesOpen_)
        {
            return true; // Already closed
        }

        // Close HDF5 file
        if (fileId_ >= 0)
        {
            H5Fclose(fileId_);
            fileId_ = H5I_INVALID_HID;
        }

        timeSeriesOpen_ = false;

        // Write XDMF companion file
        auto basePath = timeSeriesPath_;
        basePath.replace_extension("");
        return writeXDMF(basePath, timeSeriesConfig_, timeSteps_);

#else
        lastError_ = "HDF5 support not compiled.";
        return false;
#endif
    }

#ifdef POISSON_HAS_HDF5
    bool HDF5Writer::createMeshGroup(const poisson::Config &config)
    {
        hid_t meshGroup = H5Gcreate2(fileId_, "/Mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (meshGroup < 0)
        {
            lastError_ = "Failed to create /Mesh group";
            return false;
        }

        const int nx = config.nx();
        const int ny = config.ny();
        const double dx = 1.0 / nx;
        const double dy = 1.0 / ny;

        // Create X coordinate array (node positions)
        {
            hsize_t dims[1] = {static_cast<hsize_t>(nx + 1)};
            hid_t dataspace = H5Screate_simple(1, dims, nullptr);
            hid_t dataset = H5Dcreate2(meshGroup, "X", H5T_NATIVE_DOUBLE, dataspace,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            std::vector<double> xCoords(nx + 1);
            for (int i = 0; i <= nx; ++i)
            {
                xCoords[i] = i * dx;
            }
            H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, xCoords.data());

            H5Dclose(dataset);
            H5Sclose(dataspace);
        }

        // Create Y coordinate array
        {
            hsize_t dims[1] = {static_cast<hsize_t>(ny + 1)};
            hid_t dataspace = H5Screate_simple(1, dims, nullptr);
            hid_t dataset = H5Dcreate2(meshGroup, "Y", H5T_NATIVE_DOUBLE, dataspace,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            std::vector<double> yCoords(ny + 1);
            for (int j = 0; j <= ny; ++j)
            {
                yCoords[j] = j * dy;
            }
            H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, yCoords.data());

            H5Dclose(dataset);
            H5Sclose(dataspace);
        }

        H5Gclose(meshGroup);
        return true;
    }
#endif

    // ============== XDMF Writing ==============

    bool HDF5Writer::writeXDMF(
        const std::filesystem::path &basePath,
        const poisson::Config &config,
        const std::vector<double> &times)
    {
        auto xdmfPath = basePath;
        xdmfPath.replace_extension(".xdmf");

        std::ofstream file(xdmfPath);
        if (!file.is_open())
        {
            lastError_ = "Failed to open XDMF file: " + xdmfPath.string();
            return false;
        }

        const int nx = config.nx();
        const int ny = config.ny();
        const double dx = 1.0 / nx;
        const double dy = 1.0 / ny;
        const std::string h5Filename = basePath.stem().string() + ".h5";

        file << R"(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
)";

        if (times.empty())
        {
            // Single snapshot
            file << R"(    <Grid Name="PoissonSolution" GridType="Uniform">
      <Topology TopologyType="2DCoRectMesh" Dimensions=")"
                 << (ny + 1) << " " << (nx + 1) << R"("/>
      <Geometry GeometryType="ORIGIN_DXDY">
        <DataItem Name="Origin" Dimensions="2" NumberType="Float" Precision="8" Format="XML">
          0.0 0.0
        </DataItem>
        <DataItem Name="Spacing" Dimensions="2" NumberType="Float" Precision="8" Format="XML">
          )" << dy
                 << " " << dx << R"(
        </DataItem>
      </Geometry>
      <Attribute Name="phi" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions=")"
                 << nx << " " << ny
                 << R"(" NumberType="Float" Precision="8" Format="HDF">
          )" << h5Filename
                 << R"(:/Data/phi
        </DataItem>
      </Attribute>
    </Grid>
)";
        }
        else
        {
            // Time series (iteration recording)
            file << R"(    <Grid Name="IterationSeries" GridType="Collection" CollectionType="Temporal">
)";
            for (size_t i = 0; i < times.size(); ++i)
            {
                file << R"(      <Grid Name="Iteration_)" << static_cast<int>(times[i]) << R"(" GridType="Uniform">
        <Time Value=")"
                     << times[i] << R"("/>
        <Topology TopologyType="2DCoRectMesh" Dimensions=")"
                     << (ny + 1) << " " << (nx + 1) << R"("/>
        <Geometry GeometryType="ORIGIN_DXDY">
          <DataItem Dimensions="2" Format="XML">0.0 0.0</DataItem>
          <DataItem Dimensions="2" Format="XML">)"
                     << dy << " " << dx << R"(</DataItem>
        </Geometry>
        <Attribute Name="phi" AttributeType="Scalar" Center="Cell">
          <DataItem Dimensions=")"
                     << nx << " " << ny << R"(" NumberType="Float" Precision="8" Format="HDF">
            )" << h5Filename
                     << ":/Time/step_"
                     << std::setfill('0') << std::setw(6) << i << R"(/phi
          </DataItem>
        </Attribute>
      </Grid>
)";
            }
            file << R"(    </Grid>
)";
        }

        file << R"(  </Domain>
</Xdmf>
)";

        if (!file.good())
        {
            lastError_ = "Error writing XDMF file";
            return false;
        }

        return true;
    }

} // namespace io
