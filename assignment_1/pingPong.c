#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Maximum array size 2^20= 1048576 elements
#define MAX_ARRAY_SIZE (1 << 20)

int main(int argc, char **argv)
{
    printf("Running ping pong on two nodes ...\n");
    // Variables for the process rank and number of processes
    int myRank, numProcs, i;
    MPI_Status status;

    // Initialize MPI, find out MPI communicator size and process rank
    printf("Initializing MPI...\n");
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int *myArray = (int *)malloc(sizeof(int) * MAX_ARRAY_SIZE);
    if (myArray == NULL)
    {
        printf("Not enough memory\n");
        exit(1);
    }
    // Initialize myArray
    for (i = 0; i < MAX_ARRAY_SIZE; i++)
        myArray[i] = 1;

    int numberOfElementsToSend;
    int numberOfElementsReceived;

    FILE *csv = NULL;
    if (myRank == 0)
    {
        csv = fopen("/home/pchristiaanse/HighPerformance/assignment_1/pingpong_results_two_nodes.csv", "w");
        if (!csv)
        {
            printf("Could not open CSV file for writing!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(csv, "elements,sample_idx,time\n");
    }

    // PART C
    if (numProcs < 2)
    {
        printf("Error: Run the program with at least 2 MPI tasks!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Loop over message sizes: 1, 2, 4, ..., MAX_ARRAY_SIZE
    for (int exp = 0; exp < 21; exp++)
    {
        numberOfElementsToSend = 1 << exp;
        numberOfElementsReceived = numberOfElementsToSend;
        if (myRank == 0)
        {
            myArray[0] = myArray[1] + 1;
            for (i = 0; i < 5; i++)
            {
                double startTime = MPI_Wtime();
                MPI_Send(myArray, numberOfElementsToSend, MPI_INT, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(myArray, numberOfElementsReceived, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                double endTime = MPI_Wtime();
                double sampleTime = endTime - startTime;
                fprintf(csv, "%d,%d,%f\n", numberOfElementsToSend, i, sampleTime);
                printf("Elements: %d, Sample %d, Time: %f\n", numberOfElementsToSend, i, sampleTime);
            }
        }
        else if (myRank == 1)
        {
            for (i = 0; i < 5; i++)
            {
                MPI_Recv(myArray, numberOfElementsReceived, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(myArray, numberOfElementsToSend, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (myRank == 0 && csv)
        fclose(csv);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
