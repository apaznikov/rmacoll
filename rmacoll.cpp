//
// rmacoll.cpp: Implementation of MPI RMA (one-sided) collectives 
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <iostream>

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"
#include "broadcast_linear.h"
#include "broadcast_binomial.h"

// int myrank = 0;
// int nproc = 0;

// Avoid global pointer to waiter?
extern std::weak_ptr<waiter_c> waiter_wp;

// extern waiter_c waiter;

enum bcast_types_t {
    linear   = 1,
    binomial = 2
} bcast_type = binomial;

// Prototype for RMA broadcast function
std::function<int(const void *, int, MPI_Datatype, MPI_Aint, 
                  int, MPI_Datatype, MPI_Win, MPI_Comm)> RMA_Bcast;

// test_rma_bcast: Test RMA collectives
void test_rmacoll(decltype(RMA_Bcast) bcast_func, MPI_Comm comm)
{
    int myrank, nproc;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nproc);

    // Number of variables in window
    const auto count = 1;           

    RMA_Win_guard<int> win(count, comm);
    {
        auto ptr = win.get_ptr();
        *ptr = (myrank + 1) * 10;

        std::cout << myrank << "\t" << "BEFORE\t" << *ptr << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        if (myrank == 0) {
            // RMA_Lock_guard lock_all(win.get_win());
            
            bcast_func(ptr, count, MPI_INT, 0, 
                       count, MPI_INT, win.get_win(), MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << myrank << "\t" << "AFTER\t" << *ptr << std::endl;
    }
}

// // RMA_test: for test
// // TODO remove
// void RMA_test()
// {
//     // Number of variables in window
//     const auto count = 1;           
//     const auto target_rank = 1;
//     int readval = 0;
// 
//     RMA_Win_guard<int> win(count);
//     {
//         auto *ptr = win.get_ptr();
// 
//         RMA_Lock_guard lock(target_rank, win.get_win());
// 
//         *ptr = myrank * 10;
//         std::cout << myrank << ": " << *ptr << std::endl;
//         MPI_Barrier(MPI_COMM_WORLD);
// 
//         MPI_Get(&readval, count, MPI_INT, target_rank, 
//                 0, count, MPI_INT, win.get_win());
//     }
//     
//     std::cout << myrank << ": read from " << target_rank << " " 
//               << readval << std::endl;
// }


int main(int argc, char *argv[]) 
{
    try {
        // MPI thread level support provided
        auto mpi_thr_provided = MPI_THREAD_SINGLE;
        int myrank, nproc;

        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thr_provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);

        if (bcast_type == binomial) {
            auto waiter_sp = std::make_shared<waiter_c>(mpi_thr_provided, 
                                                        MPI_COMM_WORLD);
            waiter_wp = waiter_sp;
            // waiter.start(mpi_thr_provided);

            test_rmacoll(&RMA_Bcast_binomial, MPI_COMM_WORLD);

            usleep(100000);
            waiter_sp->term();
            // waiter.term();

        } else if (bcast_type == linear) {
            test_rmacoll(RMA_Bcast_linear, MPI_COMM_WORLD);
        }

        MPI_Finalize();

        return 0;
    } 

    catch (std::exception const &ex) {
        std::cout << "Exception: " << ex.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, RET_CODE_ERROR); 
    }
}
