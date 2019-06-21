//
// broadcast_binomial.cpp: Binomial tree algorithm for RMA broadcast
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#pragma once

#include <bitset>
#include <iostream>
#include <thread>
#include <atomic>
#include <future>

#include <boost/thread/scoped_thread.hpp>

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"

const auto waiter_timeout = 1000;

extern int myrank;
extern int nproc;

// RMA_Bcast_binomial: Binomial tree broadcast
int RMA_Bcast_binomial(const void *origin_addr, int origin_count, 
                       MPI_Datatype origin_datatype, MPI_Aint target_disp,
                       int target_count, MPI_Datatype target_datatype,
                       MPI_Win win, win_id_t wid, MPI_Comm comm);

struct req_t {
    // Buffer to put/get
    int buf;        

    // Buf size for current message (in bytes)
    int bufsize;

    // Operation type
    enum op_t { noop = 0, bcast = 1 };
    op_t op;
    
    // Root (for collectives with root)
    int root;       

    // RMA window's id
    win_id_t wid;
};

const auto req_size = sizeof(req_t);

// Offsets for RMA operations
const MPI_Aint offset_buf = offsetof(req_t, buf);
const MPI_Aint offset_op = offsetof(req_t, op);
const MPI_Aint offset_root = offsetof(req_t, root);

// Class for waiter thread for binomial broadcast
class waiter_c
{
public:
    waiter_c() 
    {
    }

    waiter_c(int mpi_thr_provided, MPI_Comm comm)
    {
        start(mpi_thr_provided, comm);
    }

    // start: Start waiter thread for binomial broadcast
    void start(int mpi_thr_provided, MPI_Comm comm)
    {
        if (mpi_thr_provided != MPI_THREAD_MULTIPLE) {
            throw std::runtime_error(
                "Level of provided thread support must be MPI_THREAD_MULTIPLE");
        }

        req_win.init(req_len, MPI_COMM_WORLD);

        MPI_Comm_rank(comm, &myrank);
        MPI_Comm_size(comm, &nproc);

        waiter_thr = boost::scoped_thread<>(boost::thread
                (&waiter_c::waiter_loop, this));
    }

    // term: Terminate waiter thread
    void term()
    {
        waiter_term_flag = true;
    }

    // get_winguard: Get win guard
    RMA_Win_guard<req_t> &get_winguard()
    {
        return req_win;
    }

    auto &get_fut()
    {
        return ready_fut;
    }

    ~waiter_c() {
        // auto rank = 0;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // if (rank == 9)
        //     std::cout << rank << "R WaiterDest for " << req_win.get_id() 
        //               << std::endl;

        waiter_term_flag = true;
        waiter_thr.join();
    }

private:
    std::atomic<bool> waiter_term_flag{false};

    std::promise<void> ready_prom;
    std::future<void> ready_fut = ready_prom.get_future();

    // Request from root field
    const unsigned int req_len = 1;
    RMA_Win_guard<req_t> req_win;

    boost::scoped_thread<> waiter_thr;

    // waiter_loop: Waiter thread on each process
    void waiter_loop();

    int myrank, nproc;
};
