//
// rmacoll.cpp: Implementation of MPI RMA (one-sided) collectives 
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <iostream>
#include <memory>

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"
#include "broadcast_linear.h"
#include "broadcast_binomial.h"

// int myrank = 0;
// int nproc = 0;

// Avoid global pointer to waiter?
extern std::weak_ptr<waiter_c> waiter_weak_ptr;

// extern waiter_c waiter;

enum bcast_types_t {
    linear   = 1,
    binomial = 2
} bcast_type = binomial;

// Prototype for RMA broadcast function
std::function<int(const void*, int, MPI_Datatype, MPI_Aint, 
                  int, MPI_Datatype, MPI_Win, win_id_t, MPI_Comm)> RMA_Bcast;

auto bcast_val = 100;

// test_rma_bcast: Test RMA collectives
void test_rmacoll(decltype(RMA_Bcast) bcast_func, 
                  RMA_Win_guard<int> &scoped_win, MPI_Comm comm)
{
    auto myrank = 0;
    MPI_Comm_rank(comm, &myrank);
    
    auto sptr = scoped_win.get_sptr();
    auto ptr = sptr.get();

    if (myrank != 0) 
        std::cout << myrank << "\t" << "BEFORE\t" << ptr[0] << std::endl;

    // std::cerr << myrank << "R scoped win " << (void *) &scoped_win.get_win() 
    //           << std::endl;

    if (myrank == 0) {
        bcast_func(&bcast_val, scoped_win.get_count(), MPI_INT, 0, 
                   scoped_win.get_count(), MPI_INT, 
                   scoped_win.get_win(), scoped_win.get_id(), 
                   MPI_COMM_WORLD);
    }

    if (myrank != 0) {
        while (ptr[0] == 0);
        std::cout << myrank << "R AFTER\t" << ptr[0] << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

int myrank, nproc;

int main(int argc, char *argv[]) 
{
    try {
        // MPI thread level support provided
        auto mpi_thr_provided = MPI_THREAD_SINGLE;

        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thr_provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        
        // Number of variables in window
        const auto count = 1;           

        // int *ptr = nullptr;
        // MPI_Alloc_mem(count * sizeof(int), MPI_INFO_NULL, &ptr);
        // ptr[0] = 0;

        {
            int *raw_ptr = nullptr;
            MPI_Alloc_mem(count * sizeof(int), MPI_INFO_NULL, &raw_ptr);
            std::shared_ptr<int> ptr(raw_ptr, [](auto ptr){
                        MPI_Free_mem(ptr);
                    });
            ptr.get()[0] = 0;

            // auto ptr = new int[2];
            // *ptr = 0;

            RMA_Win_guard<int> scoped_win(ptr, count, MPI_COMM_WORLD);

            if (bcast_type == binomial) {
                auto waiter_sh_ptr = std::make_shared<waiter_c>(mpi_thr_provided, 
                                                                MPI_COMM_WORLD);
                waiter_weak_ptr = waiter_sh_ptr;
               
                test_rmacoll(&RMA_Bcast_binomial, scoped_win, MPI_COMM_WORLD);
            } else if (bcast_type == linear) {
                test_rmacoll(RMA_Bcast_linear, scoped_win, MPI_COMM_WORLD);
            }
        }

        MPI_Finalize();

        return 0;
    } 

    catch (std::exception const &ex) {
        std::cout << "Exception: " << ex.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, RET_CODE_ERROR); 
    }
}
