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
#include <functional>

#include <boost/thread/scoped_thread.hpp>

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"

// #define _DEBUG

const auto waiter_timeout = 1000;
const auto flush_timeout  = 1000;

extern int myrank;
extern int nproc;

// Default shmem buffer size
const auto SHMEM_BUF_DEFSIZE = 1000;

// RMA_Bcast_binomial: Binomial tree broadcast
int RMA_Bcast_binomial(const void *origin_addr, int origin_count, 
                       MPI_Datatype origin_datatype, MPI_Aint target_disp,
                       int target_count, MPI_Datatype target_datatype,
                       win_id_t wid, MPI_Comm comm);

// RMA_Bcast_binomial: Binomial tree broadcast
int RMA_Bcast_binomial_shmem(const void *origin_addr, int origin_count, 
                             MPI_Datatype origin_datatype, MPI_Aint target_disp,
                             int target_count, MPI_Datatype target_datatype,
                             win_id_t wid, MPI_Comm comm);

// RMA_Bcast_test: Test if RMA bcast is done
int RMA_Bcast_test();

// RMA_Bcast_flush: Wait until RMA_Bcast is completed
int RMA_Bcast_flush();

// RMA_Bcast_test: Test if RMA bcast is done
int RMA_Bcast_test(bool &done);

// comp_srank: Compute rank relative to root
int comp_srank(int myrank, int root, int nproc);

// comp_rank: Compute rank from srank
int comp_rank(int srank, int root, int nproc);

// Operation request 
enum req_t { noop = 0, bcast = 1, resize = 2 };

using buf_dtype = int;
    
// Description of the operations
struct descr_t {
    // Root (for collectives with root)
    int root = -1;

    // Buf size for current message (in bytes)
    int bufsize;

    // RMA window's id
    win_id_t wid = MPI_WIN_NO_ID;
};

const auto req_size = sizeof(req_t);
const auto descr_t_size = sizeof(descr_t);

// Offsets for RMA operations
// const MPI_Aint offset_buf = offsetof(data_t, buf);
// const MPI_Aint offset_root_wid = offsetof(data_t, root);

const auto root_wid_size = sizeof(descr_t::root) + sizeof(descr_t::wid);

struct leaders_t {
    // True if this process is the leader in shmem communicator
    bool isleader;

    // Rank in the original communicator
    int origrank;
};

// Class for waiter thread for binomial broadcast
class waiter_c
{
public:
    enum type_t { bin, bin_shmem };

    waiter_c() 
    {
    }

    waiter_c(int mpi_thr_provided, MPI_Comm comm, type_t _type)
    {
        type = _type;
        start(mpi_thr_provided, comm);
    }

    // start: Start waiter thread for binomial broadcast
    void start(int mpi_thr_provided, MPI_Comm orig_comm)
    {
        if (mpi_thr_provided != MPI_THREAD_MULTIPLE) {
            throw std::runtime_error(
                "Level of provided thread support must be MPI_THREAD_MULTIPLE");
        }

        // std::cerr << myrank << "R b 101\n";

        // MPI_Comm_rank(comm, &myrank);
        // MPI_Comm_size(comm, &nproc);
        // 
        // // Init RMA window with array of requests
        // req_win_g.init(nproc, comm);

        // std::cerr << myrank << "R b 102\n";

        // // Init RMA window with array of descriptions
        // descr_win_g.init(nproc, comm);

        // std::cerr << myrank << "R b 103\n";

        // // Init window for complete counter
        // donecntr_win_g.init(1, comm);

        // std::cerr << myrank << "R b 104\n";

        // set_donecntr(nproc - 1);

        // set_ops();

        // if (type == bin) {
        //     std::cerr << myrank << "R b 105\n";
        //     waiter_thr = boost::scoped_thread<>(boost::thread
        //             (&waiter_c::waiter_loop, this));

        MPI_Comm *comm = nullptr;
        MPI_Comm leader_comm;
        
        if (type == bin) {
            // comm = orig_comm;
            // peer_count = nproc;
        } else if (type == bin_shmem) {
            // For shared memory binomial algorithm,
            // create new communicator and allocate shared buffer
            MPI_Comm_split_type(orig_comm, MPI_COMM_TYPE_SHARED, 0,
                                MPI_INFO_NULL, &comm_sh);

            MPI_Comm_rank(comm_sh, &rank_sh);

            // MPI_Aint size = 0;
            // if (rank_sh == 0)
            //     size = SHMEM_BUF_DEFSIZE * sizeof(buf_dtype);
            // else
            //     size = 0;

            win_sh_g.init(SHMEM_BUF_DEFSIZE, comm_sh, wintype_t::shmem);

            // MPI_Win_allocate_shared(size, sizeof(buf_dtype), MPI_INFO_NULL,
            //                         comm_sh, &shbuf, &win_sh);

            // We allocate shared buffer on proc 0, 
            // so all the rest ranks do query the address of it

            MPI_Aint shbuf_sz = 0;
            int disp_unit = 0;
            MPI_Win_shared_query(win_sh_g.get_win(), 0, &shbuf_sz, 
                                 &disp_unit, win_sh_g.get_ptr());

            // Create array with leader info
            leaders.resize(nproc, 0);

            char isleader = 0;

            if (rank_sh == 0) {
                isleader = 1;
            }

            // Collect info about leaders on all procs
            MPI_Allgather(&isleader, 1, MPI_CHAR,
                          leaders.data(), 1, MPI_CHAR, orig_comm);

            if (myrank == 5)
                for (auto i = 0; i < nproc; i++) {
                    std::cout << i << " " << int(leaders[i]) << std::endl;
                }

            // if (rank_sh == 0) {
                // Group of orig communicator and group of leaders
                MPI_Group orig_group, leader_group;
                MPI_Comm_group(orig_comm, &orig_group);

                // Mapping of leaders and nodes
                // (index is node, value is the rank)
                for (auto rank = 0; rank < nproc; rank++) {
                    if (leaders[rank] == 1) {
                        leader_map.push_back(rank);

                        nnodes++;
                    }
                }

                // Add to the new group of leaders
                MPI_Group_incl(leader_group, leader_map.size(), 
                               leader_map.data(), &leader_group);

                MPI_Comm_create(orig_comm, leader_group, &leader_comm);

                comm = &leader_comm;
                // peer_count = nnodes;
            // }
            // peer_count = nproc;

            // waiter_func = &waiter_c::waiter_loop_shmem;

        } else {
            goto _err_lbl;
        }

        MPI_Comm_rank(*comm, &myrank);

        MPI_Comm_size(*comm, &nproc);

        // Init RMA window with array of requests
        req_win_g.init(nproc, *comm);

        // Init RMA window with array of descriptions
        descr_win_g.init(nproc, *comm);

        // Init window for complete counter
        donecntr_win_g.init(1, *comm);

        set_donecntr(nproc - 1);

        // Init op array by no_op
        set_ops();

        // Start broadcaster thread
        if (type == bin) {
            waiter_thr = boost::scoped_thread<>(boost::thread
                    (&waiter_c::waiter_loop, this));
        } else if (type == bin_shmem) {
            waiter_thr = boost::scoped_thread<>(boost::thread
                    (&waiter_c::waiter_loop_shmem, this));
        } else {
            goto _err_lbl;
        }

        return;
        
        _err_lbl:
            std::cerr << "Invalid algorithm type" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // set_donecntr: Set done counter to val
    void set_donecntr(int val)
    {
        *donecntr_win_g.get_ptr() = val;
    }

    // term: Terminate waiter thread
    void term()
    {
        waiter_term_flag = true;
    }

    // get_datawin: get window for request
    MPI_Win &get_datawin()
    {
        return descr_win_g.get_win();
    }

    // get_opwin: get window for operation request
    MPI_Win &get_reqwin()
    {
        return req_win_g.get_win();
    }
    
    // get_winguard: Get win guard for done flag
    MPI_Win &get_donecntrwin()
    {
        return donecntr_win_g.get_win();
    }

    auto &get_fut()
    {
        return ready_fut;
    }

    int get_nproc()
    {
        return nproc;
    }

    int get_peer_count()
    {
        return peer_count;
    }

    int get_myrank()
    {
        return myrank;
    }

    ~waiter_c() {
        waiter_term_flag = true;
        waiter_thr.join();
    }

private:
    type_t type;

    std::atomic<bool> waiter_term_flag{false};

    std::promise<void> ready_prom;
    std::future<void> ready_fut = ready_prom.get_future();
    
    // Window for operation request
    RMA_Win_guard<req_t> req_win_g;

    // Window for data (buffer + info)
    RMA_Win_guard<descr_t> descr_win_g;

    // Counter of completed processes
    RMA_Win_guard<int> donecntr_win_g;

    // Size of req, descr arrays
    // For binomial it equals nproc
    // For binomial shmem is equals nnodes
    int peer_count = 1;

    // Window for binomial shmem algorithm
    RMA_Win_guard<buf_dtype> win_sh_g;

    /////////////////////////////////////////////
    // Fields for binomial shared algorithm
    /////////////////////////////////////////////

    // Shmem communicator
    MPI_Comm comm_sh;

    // Rank in shmem comm
    int rank_sh = 0;

    // Array of leaders
    std::vector<char> leaders;

    // Mapping of nodes and leaders (actual for leaders only)
    // Index of an element is the node number, 
    // value is the leader's rank
    std::vector<int> leader_map;

    // Number of nodes (actual for leaders only)
    int nnodes = 0;

    /////////////////////////////////////////////

    boost::scoped_thread<> waiter_thr;

    // waiter_loop: Waiter thread on each process
    void waiter_loop();

    // waiter_loop_shmem: Waiter thread on each process (for shmem algorithm)
    void waiter_loop_shmem();

    int myrank, nproc;

    void set_ops()
    {
        auto req_arr = req_win_g.get_ptr();
        for (auto rank = 0; rank < nproc; rank++)
            req_arr[rank] = req_t::noop;
    }
};
