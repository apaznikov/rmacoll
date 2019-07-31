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

// RMA_Bcast_test: Test if RMA bcast is done
int RMA_Bcast_test();

// RMA_Bcast_flush: Wait until RMA_Bcast is completed
int RMA_Bcast_flush();

// RMA_Bcast_test: Test if RMA bcast is done
int RMA_Bcast_test(bool &done);

// Default buffer size
const auto BUFCOUNT = 1000;

// Operation request 
struct req_t {
    enum op_t { noop = 0, bcast = 1, resize = 2 };
    op_t op;
    
    // Buf size for current message (in bytes)
    int count;
};
    
// Data for operation
struct data_t {
    // Buffer to put/get
    int buf[BUFCOUNT];

    // Root (for collectives with root)
    int root;

    // RMA window's id
    win_id_t wid;
};

const auto req_size = sizeof(req_t);
const auto data_size = sizeof(data_t);

// Offsets for RMA operations
const MPI_Aint offset_buf = offsetof(data_t, buf);

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

        MPI_Comm_rank(comm, &myrank);
        MPI_Comm_size(comm, &nproc);
        
        // Init RMA window with array of requests
        req_win_g.init(nproc, comm);

        // Init RMA window with array of data
        data_win_g.init(nproc, comm);

        // Init RMA window for complete flags
        doneflag_win_g.init(nproc, comm);

        set_doneflags(true);

        set_ops();

        waiter_thr = boost::scoped_thread<>(boost::thread
                (&waiter_c::waiter_loop, this));
    }

    void set_doneflags(bool val)
    {
        auto doneflag_arr = doneflag_win_g.get_ptr();
        for (auto rank = 0; rank < nproc; rank++)
            doneflag_arr[rank] = val;
        doneflag_arr[myrank] = true;
    }


    // term: Terminate waiter thread
    void term()
    {
        waiter_term_flag = true;
    }

    // get_datawin: get window for request
    MPI_Win &get_datawin()
    {
        return data_win_g.get_win();
    }

    // get_opwin: get window for operation request
    MPI_Win &get_reqwin()
    {
        return req_win_g.get_win();
    }
    
    // get_winguard: Get win guard for done flag
    MPI_Win &get_doneflagwin()
    {
        return doneflag_win_g.get_win();
    }

    auto &get_fut()
    {
        return ready_fut;
    }

    int get_nproc()
    {
        return nproc;
    }

    int get_myrank()
    {
        return myrank;
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
    
    // Window for operation request
    RMA_Win_guard<req_t> req_win_g;

    // Window for data (buffer + info)
    RMA_Win_guard<data_t> data_win_g;

    // Array of complete flags
    RMA_Win_guard<bool> doneflag_win_g;

    boost::scoped_thread<> waiter_thr;

    // waiter_loop: Waiter thread on each process
    void waiter_loop();

    int myrank, nproc;

    void set_ops()
    {
        auto req_arr = req_win_g.get_ptr();
        for (auto rank = 0; rank < nproc; rank++)
            req_arr[rank].op = req_t::op_t::noop;
    }
};
