//
// rmacoll.cpp: Implementation of MPI RMA (one-sided) collectives 
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <iostream>
#include <memory>
#include <vector>
#include <sstream>
#include <fstream>

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

// For tests
// const auto bcast_buf_size_min = 100'000;
// const auto bcast_buf_size_max = 2'000'000;
// const auto bcast_buf_size_step = 100'000;

// For light test
// const auto bcast_buf_size_min = 5'000'000;
// const auto bcast_buf_size_max = 5'000'000;
// const auto bcast_buf_size_step = 100'000;
// const auto warmup_flag = true;
// const auto warmup_ntimes = 5;
// 
// const auto ntimes = 10;

// For debug
#define _DEBUG
const auto bcast_buf_size_min = 10;
const auto bcast_buf_size_max = 10;
const auto bcast_buf_size_step = 10;
const auto warmup_flag = false;
const auto warmup_ntimes = 0;

const auto ntimes = 1;

const auto bcast_val = 100;

// test_rmacoll_1root: Test RMA collectives (one root)
void test_rmacoll_1root(decltype(RMA_Bcast) bcast_func, 
                        int root, MPI_Comm comm)
{
    // Prepare file for output data
    std::ofstream datafile;
    if (myrank == root) {
        auto nproc = 0;
        MPI_Comm_size(comm, &nproc);
        std::stringstream ss;
        if (bcast_type == linear)
            ss << "results/linear-n" << nproc << ".dat";
        else
            ss << "results/binomial-n" << nproc << ".dat";

        datafile.open(ss.str());
        datafile << "size\ttime\n";
    }

    for (auto bcast_buf_size = bcast_buf_size_min;
         bcast_buf_size <= bcast_buf_size_max;
         bcast_buf_size += bcast_buf_size_step) { 

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
#ifdef _DEBUG
            std::cout << myrank << "R BEFORE\t" << raw_ptr[0] << std::endl;
#endif
        }

        // Create and init buf for bcast ("from buf" - on root)
        auto bcast_buf_alloc_size = bcast_buf_size;
        if (myrank != root)
            bcast_buf_alloc_size = 0;

        std::vector<bcast_buf_t> bcast_buf(bcast_buf_alloc_size, bcast_val);
        MPI_Barrier(comm);

        if (warmup_flag == true) {
            for (auto i = 0; i < warmup_ntimes; i++) {

                if (myrank == root) {
                    bcast_func(bcast_buf.data(), scoped_win.get_count(), 
                               MPI_INT, 0, scoped_win.get_count(), MPI_INT, 
                               scoped_win.get_win(), scoped_win.get_id(), 
                               MPI_COMM_WORLD);
                }

                if (bcast_type == binomial) {
                    if (myrank == root)  {
                        RMA_Bcast_flush();
                    }
                } 
            }
        }

        auto tbegin = MPI_Wtime();

        for (auto i = 0; i < ntimes; i++) {

            auto ti1 = MPI_Wtime();

            if (myrank == root) {
                bcast_func(bcast_buf.data(), scoped_win.get_count(), MPI_INT, 0,
                           scoped_win.get_count(), MPI_INT, 
                           scoped_win.get_win(), scoped_win.get_id(), 
                           MPI_COMM_WORLD);
            }

            // If binomial broadcast, call flush
            if (bcast_type == binomial) {
                if (myrank == root)  {
                    // auto t1 = MPI_Wtime();

                    RMA_Bcast_flush();

                    // auto t2 = MPI_Wtime();
                    // std::cout << "i = " << i << " FLUSH " << t2 - t1 
                    //           << std::endl;
                }
            } 

            if (myrank == root) {
                auto ti2 = MPI_Wtime();
                std::cout << "i = " << i << " " << ti2 - ti1 << std::endl
                          << std::endl;
            }
            
        }

        if (myrank == root) {
            auto tend = MPI_Wtime();
            auto tavg = (tend - tbegin) / ntimes;
            auto bcast_type_str = bcast_type == linear ? "linear": "binomial";
            std::cout << "Elapsed time (nproc = "
                      << nproc << ", bufsize = " << bcast_buf_size 
                      << " bcast_type = " << bcast_type_str << "): " 
                      << tavg << std::endl;

            datafile << bcast_buf_size << "\t" << tavg << std::endl;
        }

        MPI_Barrier(comm);

#ifdef _DEBUG
        if (myrank != root) {
            usleep((myrank + 1) * 50000);
            std::cout << myrank << "R AFTER\t";
            for (auto i = 0; i < bcast_buf_size; i++)
                std::cout << raw_ptr[i] << " ";
            std::cout << std::endl;
        }
#endif
    }

    datafile.close();
}

// test_rmacoll_nroot: Test RMA collectives (multiple root)
void test_rmacoll_nroot(decltype(RMA_Bcast) bcast_func, int bcast_buf_size, 
                        MPI_Comm comm)
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
    std::vector<bcast_buf_t> bcast_buf(bcast_buf_size, (myrank + 1) * 10);

    // std::cout << myrank << "R FIRST " << bcast_buf[0] << std::endl;

    bcast_func(bcast_buf.data(), sc_win_vec[myrank].get_count(), MPI_INT, 0,
               sc_win_vec[myrank].get_count(), MPI_INT, 
               sc_win_vec[myrank].get_win(), sc_win_vec[myrank].get_id(), 
               MPI_COMM_WORLD);

    std::fill(bcast_buf.begin(), bcast_buf.end(), (myrank + 1) * 100);

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

        // Get bcast type from argument
        if (argc == 2) {
            if (std::string(argv[1]) == "linear")
                bcast_type = linear;
            else if (std::string(argv[1]) == "binomial")
                bcast_type = binomial;
        }

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
