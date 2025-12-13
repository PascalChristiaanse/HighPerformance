#include "mpi/MPIContext.hpp"

namespace poisson
{

    MPIContext::MPIContext(int &argc, char **&argv)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);

        initialized_ = true;
    }

    MPIContext::~MPIContext()
    {
        if (initialized_)
        {
            freeBorderTypes();
            if (cartComm_ != MPI_COMM_NULL)
            {
                MPI_Comm_free(&cartComm_);
            }
            MPI_Finalize();
        }
    }

    void MPIContext::createCartesian(std::array<int, 2> dims,
                                     std::array<bool, 2> periodic)
    {
        cartDims_ = dims;

        // If dims are 0, let MPI determine the decomposition
        if (cartDims_[0] == 0 || cartDims_[1] == 0)
        {
            MPI_Dims_create(size_, 2, cartDims_.data());
        }

        // Convert bool to int for MPI
        int periodicInt[2] = {periodic[0] ? 1 : 0, periodic[1] ? 1 : 0};

        // Create Cartesian communicator
        if (MPI_Cart_create(comm_, 2, cartDims_.data(), periodicInt, 1, &cartComm_) != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Cart_create failed");
        }

        // Get this process's coordinates
        if (MPI_Cart_coords(cartComm_, rank_, 2, coords_.data()) != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Cart_coords failed");
        }

        // Get neighbor ranks
        // neighbors_[0] = left (X-), neighbors_[1] = right (X+)
        // neighbors_[2] = down (Y-), neighbors_[3] = up (Y+)
        MPI_Cart_shift(cartComm_, 0, 1, &neighbors_[0], &neighbors_[1]); // X direction
        MPI_Cart_shift(cartComm_, 1, 1, &neighbors_[2], &neighbors_[3]); // Y direction
    }

    int MPIContext::neighborRank(int direction, bool positive) const
    {
        int index = direction * 2 + (positive ? 1 : 0);
        return neighbors_[index];
    }

    void MPIContext::barrier() const
    {
        MPI_Barrier(cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
    }

    double MPIContext::allReduceMax(double localValue) const
    {
        double globalValue;
        MPI_Allreduce(&localValue, &globalValue, 1, MPI_DOUBLE, MPI_MAX,
                      cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        return globalValue;
    }

    double MPIContext::allReduceSum(double localValue) const
    {
        double globalValue;
        MPI_Allreduce(&localValue, &globalValue, 1, MPI_DOUBLE, MPI_SUM,
                      cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        return globalValue;
    }

    int MPIContext::allReduceSum(int localValue) const
    {
        int globalValue;
        MPI_Allreduce(&localValue, &globalValue, 1, MPI_INT, MPI_SUM,
                      cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        return globalValue;
    }

    void MPIContext::sendRecv(const double *sendBuf, int sendCount, int destRank,
                              double *recvBuf, int recvCount, int srcRank,
                              int tag) const
    {
        MPI_Status status;
        MPI_Sendrecv(sendBuf, sendCount, MPI_DOUBLE, destRank, tag,
                     recvBuf, recvCount, MPI_DOUBLE, srcRank, tag,
                     cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_, &status);
    }

    void MPIContext::gather(const double *sendBuf, int sendCount,
                            double *recvBuf, int recvCount) const
    {
        MPI_Gather(sendBuf, sendCount, MPI_DOUBLE,
                   recvBuf, recvCount, MPI_DOUBLE, 0,
                   cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
    }

    void MPIContext::gatherv(const double *sendBuf, int sendCount,
                             double *recvBuf, const int *recvCounts,
                             const int *displs) const
    {
        MPI_Gatherv(sendBuf, sendCount, MPI_DOUBLE,
                    recvBuf, recvCounts, displs, MPI_DOUBLE, 0,
                    cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
    }

    void MPIContext::broadcast(double *buffer, int count) const
    {
        MPI_Bcast(buffer, count, MPI_DOUBLE,
                  0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
    }

    void MPIContext::broadcast(int *buffer, int count) const
    {
        MPI_Bcast(buffer, count, MPI_INT,
                  0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
    }

    void MPIContext::broadcast(char *buffer, int count) const
    {
        MPI_Bcast(buffer, count, MPI_CHAR,
                  0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
    }

    void MPIContext::broadcast(std::string &str) const
    {
        // Broadcast the size first
        int size = static_cast<int>(str.size());
        MPI_Bcast(&size, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);

        // Resize on non-root processes
        if (!isRoot())
        {
            str.resize(size);
        }

        // Broadcast the string data
        if (size > 0)
        {
            MPI_Bcast(const_cast<char *>(str.data()), size, MPI_CHAR,
                      0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        }
    }

    void MPIContext::broadcast(std::vector<double> &vec) const
    {
        // Broadcast the size first
        int size = static_cast<int>(vec.size());
        MPI_Bcast(&size, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);

        // Resize on non-root processes
        if (!isRoot())
        {
            vec.resize(size);
        }

        // Broadcast the vector data
        if (size > 0)
        {
            MPI_Bcast(vec.data(), size, MPI_DOUBLE,
                      0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        }
    }

    void MPIContext::broadcast(std::vector<int> &vec) const
    {
        // Broadcast the size first
        int size = static_cast<int>(vec.size());
        MPI_Bcast(&size, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);

        // Resize on non-root processes
        if (!isRoot())
        {
            vec.resize(size);
        }

        // Broadcast the vector data
        if (size > 0)
        {
            MPI_Bcast(vec.data(), size, MPI_INT,
                      0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        }
    }

    void MPIContext::broadcast(std::vector<char> &vec) const
    {
        // Broadcast the size first
        int size = static_cast<int>(vec.size());
        MPI_Bcast(&size, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);

        // Resize on non-root processes
        if (!isRoot())
        {
            vec.resize(size);
        }

        // Broadcast the vector data
        if (size > 0)
        {
            MPI_Bcast(vec.data(), size, MPI_CHAR,
                      0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        }
    }

    void MPIContext::broadcast(std::vector<std::string> &vec) const
    {
        // Broadcast the number of strings
        int numStrings = static_cast<int>(vec.size());
        MPI_Bcast(&numStrings, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);

        // Resize on non-root processes
        if (!isRoot())
        {
            vec.resize(numStrings);
        }

        // Broadcast each string
        for (int i = 0; i < numStrings; ++i)
        {
            broadcast(vec[i]);
        }
    }

    void MPIContext::broadcast(poisson::Config &config) const
    {
        // Broadcast grid dimensions
        int nx = config.nx();
        int ny = config.ny();
        MPI_Bcast(&nx, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        MPI_Bcast(&ny, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);

        // Broadcast solver parameters
        double precisionGoal = config.precisionGoal();
        int maxIterations = config.maxIterations();
        MPI_Bcast(&precisionGoal, 1, MPI_DOUBLE, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        MPI_Bcast(&maxIterations, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);

        // Broadcast number of sources
        int numSources = static_cast<int>(config.sources().size());
        MPI_Bcast(&numSources, 1, MPI_INT, 0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);

        // Broadcast source data
        std::vector<double> sourceData;
        if (isRoot())
        {
            // Pack source data: [x1, y1, value1, x2, y2, value2, ...]
            sourceData.reserve(numSources * 3);
            for (const auto &src : config.sources())
            {
                sourceData.push_back(src.x);
                sourceData.push_back(src.y);
                sourceData.push_back(src.value);
            }
        }
        else
        {
            sourceData.resize(numSources * 3);
        }

        // Broadcast the packed source data
        if (numSources > 0)
        {
            MPI_Bcast(sourceData.data(), numSources * 3, MPI_DOUBLE,
                      0, cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_);
        }

        // Reconstruct config on non-root processes
        if (!isRoot())
        {
            config.setGridSize(nx, ny);
            config.setPrecisionGoal(precisionGoal);
            config.setMaxIterations(maxIterations);

            // Unpack and add sources
            for (int i = 0; i < numSources; ++i)
            {
                double x = sourceData[i * 3];
                double y = sourceData[i * 3 + 1];
                double value = sourceData[i * 3 + 2];
                config.addSource(x, y, value);
            }
        }
    }

    void MPIContext::createBorderTypes(const Grid &grid)
    {
        if (borderTypesCreated_)
        {
            freeBorderTypes();
        }

        const int dimX = grid.dimX();
        const int dimY = grid.dimY();
        const int ghost = grid.ghostLayers();

        // Interior dimensions (excluding ghost cells)
        const int interiorX = dimX - 2 * ghost;
        const int interiorY = dimY - 2 * ghost;

        // X-direction border type (horizontal row - for top/bottom exchange)
        // This is a row of interior Y elements, which ARE contiguous in memory
        // Memory layout: phi[x][y] = phiData[x * dimY + y]
        // A row at fixed x: phi[x][ghost], phi[x][ghost+1], ..., phi[x][ghost+interiorY-1]
        // These are contiguous: indices are x*dimY+ghost, x*dimY+ghost+1, etc.
        MPI_Type_contiguous(interiorY, MPI_DOUBLE, &borderTypeX_);
        MPI_Type_commit(&borderTypeX_);

        // Y-direction border type (vertical column - for left/right exchange)  
        // This is a column of interior X elements, which are NOT contiguous
        // A column at fixed y: phi[ghost][y], phi[ghost+1][y], ..., phi[ghost+interiorX-1][y]
        // Indices: ghost*dimY+y, (ghost+1)*dimY+y, etc. - stride is dimY
        MPI_Type_vector(interiorX,    // count: number of blocks (rows)
                        1,            // blocklength: 1 element per block
                        dimY,         // stride: distance between elements (in doubles)
                        MPI_DOUBLE,
                        &borderTypeY_);
        MPI_Type_commit(&borderTypeY_);

        borderTypesCreated_ = true;
    }

    void MPIContext::freeBorderTypes()
    {
        if (borderTypesCreated_)
        {
            if (borderTypeX_ != MPI_DATATYPE_NULL)
            {
                MPI_Type_free(&borderTypeX_);
                borderTypeX_ = MPI_DATATYPE_NULL;
            }
            if (borderTypeY_ != MPI_DATATYPE_NULL)
            {
                MPI_Type_free(&borderTypeY_);
                borderTypeY_ = MPI_DATATYPE_NULL;
            }
            borderTypesCreated_ = false;
        }
    }

    void MPIContext::exchangeBoundaries(Grid &grid) const
    {
        if (!borderTypesCreated_)
        {
            throw std::runtime_error("Border types not created. Call createBorderTypes first.");
        }

        MPI_Comm comm = cartComm_ != MPI_COMM_NULL ? cartComm_ : comm_;
        MPI_Status status;

        const int dimY = grid.dimY();
        const int ghost = grid.ghostLayers();
        double *phi = grid.phiData();

        // Neighbor indices: [0]=X-, [1]=X+, [2]=Y-, [3]=Y+

        // ============== X-direction exchange (top/bottom neighbors) ==============
        // Send top interior row to X+ neighbor, receive into bottom ghost row from X- neighbor
        // Top interior row: x = dimX - 2*ghost (i.e., dimX - ghost - 1 for ghost=1)
        // Bottom ghost row: x = 0
        const int topInteriorRow = grid.dimX() - 2 * ghost;
        const int bottomGhostRow = 0;
        const int bottomInteriorRow = ghost;
        const int topGhostRow = grid.dimX() - ghost;

        // Send to X+ (bottom neighbor in our coord system), receive from X-
        MPI_Sendrecv(&phi[topInteriorRow * dimY + ghost], 1, borderTypeX_, neighbors_[1], 0,
                     &phi[bottomGhostRow * dimY + ghost], 1, borderTypeX_, neighbors_[0], 0,
                     comm, &status);

        // Send to X- (top neighbor), receive from X+
        MPI_Sendrecv(&phi[bottomInteriorRow * dimY + ghost], 1, borderTypeX_, neighbors_[0], 1,
                     &phi[topGhostRow * dimY + ghost], 1, borderTypeX_, neighbors_[1], 1,
                     comm, &status);

        // ============== Y-direction exchange (left/right neighbors) ==============
        // Send right interior column to Y+ neighbor, receive into left ghost column from Y- neighbor
        // Right interior column: y = dimY - 2*ghost
        // Left ghost column: y = 0
        const int rightInteriorCol = dimY - 2 * ghost;
        const int leftGhostCol = 0;
        const int leftInteriorCol = ghost;
        const int rightGhostCol = dimY - ghost;

        // Send to Y+ (right neighbor), receive from Y-
        MPI_Sendrecv(&phi[ghost * dimY + rightInteriorCol], 1, borderTypeY_, neighbors_[3], 2,
                     &phi[ghost * dimY + leftGhostCol], 1, borderTypeY_, neighbors_[2], 2,
                     comm, &status);

        // Send to Y- (left neighbor), receive from Y+
        MPI_Sendrecv(&phi[ghost * dimY + leftInteriorCol], 1, borderTypeY_, neighbors_[2], 3,
                     &phi[ghost * dimY + rightGhostCol], 1, borderTypeY_, neighbors_[3], 3,
                     comm, &status);
    }

    double MPIContext::wtime() const
    {
        barrier();
        return MPI_Wtime();
    }

} // namespace poisson
