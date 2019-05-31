#include <unistd.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, len;
    char procname[MPI_MAX_PROCESSOR_NAME];
    
    MPI_Init(&argc, &argv);
    MPI_Get_processor_name(procname, &len);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    printf("Rank %d on %s\n", rank, procname);
    
    MPI_Finalize();
    return 0;
}