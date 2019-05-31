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

#include <boost/thread/scoped_thread.hpp>

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"

extern int myrank;
extern int nproc;

// RMA_Bcast_binomial: Binomial tree broadcast
int RMA_Bcast_binomial(const void *origin_addr, int origin_count, 
                       MPI_Datatype origin_datatype, MPI_Aint target_disp,
                       int target_count, MPI_Datatype target_datatype,
                       MPI_Win win, MPI_Comm comm);


// Class for waiter thread for binomial broadcast
class waiter_c
{
public:
    struct req_t {
        int buf;        // buffer to send/recv
        // operation (0 -- no operation, 1 -- bcast)
        int op;
        int root;       // root (for root exchange)
    };

    waiter_c() 
    {
    }

    waiter_c(int mpi_thr_provided)
    {
        start(mpi_thr_provided);
    }

    // start: Start waiter thread for binomial broadcast
    void start(int mpi_thr_provided)
    {
        if (mpi_thr_provided != MPI_THREAD_MULTIPLE) {
            throw std::runtime_error(
                "Level of provided thread support must be MPI_THREAD_MULTIPLE");
        }

        req_win.init(req_len);

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

private:
    std::atomic<bool> waiter_term_flag{false};

    boost::scoped_thread<> waiter_thr;

    // Request from root field
    const unsigned int req_len = 1;
    RMA_Win_guard<req_t> req_win;

    // waiter_loop: Waiter thread on each process
    void waiter_loop();
};
