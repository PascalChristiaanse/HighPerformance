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
        MPI_Cart_create(comm_, 2, cartDims_.data(), periodicInt, 1, &cartComm_);

        // Get this process's coordinates
        MPI_Cart_coords(cartComm_, rank_, 2, coords_.data());

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

    double MPIContext::wtime() const
    {
        barrier();
        return MPI_Wtime();
    }

} // namespace poisson
