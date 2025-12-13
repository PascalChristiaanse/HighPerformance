#pragma once

#include <mpi.h>
#include <array>
#include <vector>
#include <stdexcept>

#include "poisson/Config.hpp"
#include "poisson/Grid.hpp"

namespace poisson
{

    /// RAII wrapper for MPI initialization and common operations
    class MPIContext
    {
    public:
        /// Initialize MPI
        /// @param argc Reference to main's argc
        /// @param argv Reference to main's argv
        MPIContext(int &argc, char **&argv);

        /// Finalize MPI
        ~MPIContext();

        // Non-copyable, non-movable (singleton-like per process)
        MPIContext(const MPIContext &) = delete;
        MPIContext &operator=(const MPIContext &) = delete;
        MPIContext(MPIContext &&) = delete;
        MPIContext &operator=(MPIContext &&) = delete;

        /// Get this process's rank
        [[nodiscard]] int rank() const noexcept { return rank_; }

        /// Get total number of processes
        [[nodiscard]] int size() const noexcept { return size_; }

        /// Check if this is the root process (rank 0)
        [[nodiscard]] bool isRoot() const noexcept { return rank_ == 0; }

        /// Get the MPI communicator
        [[nodiscard]] MPI_Comm comm() const noexcept { return comm_; }

        // ============== Cartesian Topology ==============

        /// Create a 2D Cartesian topology for domain decomposition
        /// @param dims Number of processes in each direction [X, Y]
        /// @param periodic Whether each direction is periodic
        void createCartesian(std::array<int, 2> dims,
                             std::array<bool, 2> periodic = {false, false});

        /// Get this process's coordinates in the Cartesian grid
        [[nodiscard]] const std::array<int, 2> &coords() const noexcept { return coords_; }

        /// Get the Cartesian dimensions
        [[nodiscard]] const std::array<int, 2> &cartDims() const noexcept { return cartDims_; }

        /// Get the Cartesian communicator
        [[nodiscard]] MPI_Comm cartComm() const noexcept { return cartComm_; }

        /// Get neighbor rank in given direction
        /// @param direction 0 for X, 1 for Y
        /// @param positive true for +direction, false for -direction
        /// @return Neighbor rank or MPI_PROC_NULL if no neighbor
        [[nodiscard]] int neighborRank(int direction, bool positive) const;

        // ============== Communication ==============

        /// Barrier synchronization
        void barrier() const;

        /// All-reduce maximum of double values
        [[nodiscard]] double allReduceMax(double localValue) const;

        /// All-reduce sum of double values
        [[nodiscard]] double allReduceSum(double localValue) const;

        /// All-reduce sum of int values
        [[nodiscard]] int allReduceSum(int localValue) const;

        /// Send and receive boundary data
        /// @param sendBuf Data to send
        /// @param sendCount Number of elements to send
        /// @param destRank Destination rank
        /// @param recvBuf Buffer to receive into
        /// @param recvCount Number of elements to receive
        /// @param srcRank Source rank
        /// @param tag Message tag
        void sendRecv(const double *sendBuf, int sendCount, int destRank,
                      double *recvBuf, int recvCount, int srcRank,
                      int tag = 0) const;

        /// Gather data from all processes to root
        /// @param sendBuf Local data to send
        /// @param sendCount Number of elements to send
        /// @param recvBuf Receive buffer (only used on root)
        /// @param recvCount Number of elements to receive from each process
        void gather(const double *sendBuf, int sendCount,
                    double *recvBuf, int recvCount) const;

        /// Gather variable-sized data to root
        void gatherv(const double *sendBuf, int sendCount,
                     double *recvBuf, const int *recvCounts,
                     const int *displs) const;


        // Broadcast data from root to all processes
        /// @param buffer Data buffer
        /// @param count Number of elements
        void broadcast(double *buffer, int count) const;
        void broadcast(int *buffer, int count) const;
        void broadcast(char *buffer, int count) const;
        void broadcast(std::string &str) const;
        void broadcast(std::vector<double> &vec) const;
        void broadcast(std::vector<int> &vec) const;
        void broadcast(std::vector<char> &vec) const;
        void broadcast(std::vector<std::string> &vec) const;

        // Broadcast Config
        /// @param config Configuration object
        void broadcast(poisson::Config &config) const;

        // ============== Boundary Exchange ==============

        /// Create MPI datatypes for boundary exchange on a grid
        /// @param grid The grid to create datatypes for
        void createBorderTypes(const Grid &grid);

        /// Free MPI datatypes for boundary exchange
        void freeBorderTypes();

        /// Exchange boundary data with neighbors
        /// @param grid The grid to exchange boundaries for
        void exchangeBoundaries(Grid &grid) const;

        // ============== Timing ==============

        /// Get wall-clock time (with barrier for synchronization)
        [[nodiscard]] double wtime() const;

    private:
        MPI_Comm comm_{MPI_COMM_WORLD};
        MPI_Comm cartComm_{MPI_COMM_NULL};
        int rank_{0};
        int size_{1};
        std::array<int, 2> coords_{0, 0};
        std::array<int, 2> cartDims_{1, 1};
        std::array<int, 4> neighbors_{{MPI_PROC_NULL, MPI_PROC_NULL,
                                       MPI_PROC_NULL, MPI_PROC_NULL}};
        bool initialized_{false};

        // MPI datatypes for boundary exchange
        MPI_Datatype borderTypeX_{MPI_DATATYPE_NULL};  // For X-direction (rows - contiguous)
        MPI_Datatype borderTypeY_{MPI_DATATYPE_NULL};  // For Y-direction (columns - strided)
        bool borderTypesCreated_{false};
    };

} // namespace poisson
