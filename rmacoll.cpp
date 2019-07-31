//
// rmacoll.cpp: Implementation of MPI RMA (one-sided) collectives 
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <iostream>
#include <memory>
#include <vector>

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"
#include "broadcast_linear.h"
#include "broadcast_binomial.h"

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

using bcast_buf_t = int;
const auto bcast_root = 0;
const auto bcast_buf_size = 1;
const auto bcast_val = 100;

// test_rmacoll_1root: Test RMA collectives (one root)
void test_rmacoll_1root(decltype(RMA_Bcast) bcast_func, 
                        int root, MPI_Comm comm)
{
    // Allocate and init memory for bcast ("to buf" -- on all procs)
    bcast_buf_t *raw_ptr = nullptr;
    MPI_Alloc_mem(bcast_buf_size * sizeof(bcast_buf_t), 
                  MPI_INFO_NULL, &raw_ptr);
    std::shared_ptr<bcast_buf_t[]> sptr(raw_ptr, 
            [](auto p){ MPI_Free_mem(p); });

    std::fill_n(raw_ptr, bcast_buf_size, 0);

    // Create RMA window (all proc - to recv)
    RMA_Win_guard<bcast_buf_t> scoped_win(sptr, bcast_buf_size, 
                                          MPI_COMM_WORLD);

    auto myrank = 0;
    MPI_Comm_rank(comm, &myrank);
    
    if (myrank != root) {
        usleep((myrank + 1) * 50000);
        std::cout << myrank << "R BEFORE\t" << raw_ptr[0] << std::endl;
    }
    MPI_Barrier(comm);

    // std::cerr << myrank << "R scoped win " << (void *) &scoped_win.get_win() 
    //           << std::endl;

    if (myrank == root) {
        // Create and init buf for bcast ("from buf" - on root)
        std::array<bcast_buf_t, bcast_buf_size> bcast_buf;
        bcast_buf.fill(bcast_val);

        bcast_func(bcast_buf.data(), scoped_win.get_count(), MPI_INT, 0, 
                   scoped_win.get_count(), MPI_INT, 
                   scoped_win.get_win(), scoped_win.get_id(), 
                   MPI_COMM_WORLD);
    }

    // if (myrank != root) {
    //     // Wait until bcast will be finalized
    //     while (raw_ptr[0] == 0);
    // }

    // if (myrank == 0)
    //     RMA_Bcast_flush();

    MPI_Barrier(MPI_COMM_WORLD);

    if (myrank != root) {
        usleep((myrank + 1) * 50000);
        std::cout << myrank << "R AFTER\t" << raw_ptr[0] << std::endl;
    }
}

// test_rmacoll_nroot: Test RMA collectives (multiple root)
void test_rmacoll_nroot(decltype(RMA_Bcast) bcast_func, MPI_Comm comm)
{
    // Allocate and init memory for bcast bufs ("to buf" -- on all procs)
    std::vector<std::shared_ptr<bcast_buf_t[]>> vec_buf;

    // Array of RMA windows (all proc - to recv)
    // One window for each root
    std::vector<RMA_Win_guard<bcast_buf_t>> sc_win_vec(nproc);

    auto nproc = 0;
    MPI_Comm_size(comm, &nproc);

    for (auto rank = 0; rank < nproc; rank++) {
        bcast_buf_t *raw_ptr = nullptr;

        MPI_Alloc_mem(bcast_buf_size * sizeof(bcast_buf_t), 
                      MPI_INFO_NULL, &raw_ptr);

        std::shared_ptr<bcast_buf_t[]> sptr(raw_ptr, 
                [](auto p){ MPI_Free_mem(p); });

        std::fill_n(raw_ptr, bcast_buf_size, 0);

        sc_win_vec[rank].init(sptr, bcast_buf_size, MPI_COMM_WORLD);

        vec_buf.push_back(std::move(sptr));
    }

    MPI_Barrier(comm);

    auto myrank = 0;
    MPI_Comm_rank(comm, &myrank);
    
    // Print before bcast
    usleep(myrank * 10000);
    std::cout << myrank << "R BEFORE ";
    for (auto buf: vec_buf)
        std::cout << buf[0] << " ";
    std::cout << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // Create and init buf for bcast ("from buf" - on all procs)
    std::array<bcast_buf_t, bcast_buf_size> bcast_buf;
    bcast_buf.fill((myrank + 1) * 10);

    // std::cout << myrank << "R FIRST " << bcast_buf[0] << std::endl;

    bcast_func(bcast_buf.data(), sc_win_vec[myrank].get_count(), MPI_INT, 0,
               sc_win_vec[myrank].get_count(), MPI_INT, 
               sc_win_vec[myrank].get_win(), sc_win_vec[myrank].get_id(), 
               MPI_COMM_WORLD);

    bcast_buf.fill((myrank + 1) * 100);

    bcast_func(bcast_buf.data(), sc_win_vec[myrank].get_count(), MPI_INT, 0,
               sc_win_vec[myrank].get_count(), MPI_INT, 
               sc_win_vec[myrank].get_win(), sc_win_vec[myrank].get_id(), 
               MPI_COMM_WORLD);

    if (myrank == 0)
        RMA_Bcast_flush();

    // Print after bcast
    usleep((myrank + 1) * 50000);
    std::cout << myrank << "R AFTER ";
    for (auto buf: vec_buf)
        std::cout << buf[0] << " ";
    std::cout << std::endl;

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
        
        {
            if (bcast_type == binomial) {
                auto waiter_sh_ptr = std::make_shared<waiter_c>
                    (mpi_thr_provided, MPI_COMM_WORLD);

                waiter_weak_ptr = waiter_sh_ptr;

                test_rmacoll_1root(&RMA_Bcast_binomial, bcast_root, 
                                   MPI_COMM_WORLD);
                // test_rmacoll_nroot(&RMA_Bcast_binomial, MPI_COMM_WORLD);
                
                usleep(200000);
            } else if (bcast_type == linear) {
                test_rmacoll_1root(RMA_Bcast_linear, bcast_root, 
                                   MPI_COMM_WORLD);
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
