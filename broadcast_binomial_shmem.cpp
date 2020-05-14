//
// broadcast_binomial_shmem.cpp: 
//      Binomial tree algorithm for RMA broadcast
//      with shared memory optimization (MPI+MPI)
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"
#include "broadcast_binomial.h"

extern int iter;

// Avoid global pointers?
extern std::weak_ptr<waiter_c> waiter_weak_ptr;

// #define _DEBUG
// #define _PROF

// put_req_data: Put request and data into remote memory
static void put_req_data(const req_t &req, const descr_t &descr, 
                         buf_dtype *buf, int put_rank, 
                         const MPI_Win &req_win, 
                         const MPI_Win &descr_win,
                         const MPI_Win &bcast_win)
{
    // 1. Put buffer (data)
    MPI_Put(buf, descr.bufsize, MPI_BYTE, put_rank, 0, 
            descr.bufsize, MPI_BYTE, bcast_win);

    // 2. Put description
    auto disp = descr_t_size * descr.root;
    MPI_Put(&descr, sizeof(descr_t), MPI_BYTE, put_rank, disp, 
            sizeof(descr_t), MPI_BYTE, descr_win);

    // 3. Put request (flag) into remote memory
    disp = req_size * descr.root;
    MPI_Accumulate(&req, req_size, MPI_BYTE, put_rank, disp, req_size, MPI_BYTE,
                   MPI_REPLACE, req_win);

    // ?? Optimize: put flush after each MPI_Put or here after the MPI_Acc?
    MPI_Win_flush(put_rank, bcast_win);
    MPI_Win_flush(put_rank, descr_win);
    MPI_Win_flush(put_rank, req_win);
}

// Put requests to all peers
static void put_loop(const req_t &req, const descr_t &descr, 
                     buf_dtype *buf, int myrank, int nproc, 
                     const MPI_Win &req_win, 
                     const MPI_Win &data_win,
                     MPI_Win &bcast_win)
{
    RMA_Lock_guard lock_all_waiters_data(bcast_win);

    // Main binomial tree algorithm
    auto srank = comp_srank(myrank, descr.root, nproc);

    auto mask = 1;

    while (mask < nproc) {
        if ((srank & mask) == 0) {
            // Put (send) data to the next process if bit is not set
            auto put_rank = srank | mask;
            if (put_rank < nproc) {
                put_rank = comp_rank(put_rank, descr.root, nproc);

                put_req_data(req, descr, buf, put_rank, 
                             req_win, data_win, bcast_win);
            }
        } else {
            // If bit is set, break 
            // (in original non-RMA algorithm it's the receive phase)
            break;
        }

        mask = mask << 1;
    }
}

// RMA_Bcast_binomial: Binomial tree broadcast
// (?) Remove MPI_Win argument?
// TODO replace by common binomial shmem
int RMA_Bcast_binomial_shmem(const void *origin_addr, int origin_count, 
                             MPI_Datatype origin_datatype, 
                             MPI_Aint target_disp, int target_count, 
                             MPI_Datatype target_datatype,
                             win_id_t wid, MPI_Comm comm)
{
    std::cerr << "DEBUG 00" << std::endl;

    // Wait until previous bcast is completed
    RMA_Bcast_flush();

    auto sp = waiter_weak_ptr.lock();

    if (!sp) {
        std::cerr << "waiter_weak_ptr is expired" << std::endl;
        return RET_CODE_ERROR;
    }

    sp->set_donecntr(0);

    auto myrank = 0, nproc = 0; 
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nproc);

    auto type_size = 0;
    MPI_Type_size(origin_datatype, &type_size);

    // Fill the request with operation time
    req_t req = req_t::bcast;

    // Fill the description with root, bufsize, wid
    descr_t descr;
    descr.root = myrank;
    descr.bufsize = origin_count * type_size;
    descr.wid = wid;

    // Search for bcast window id in the windows list
    winlist_item_t bcast_winlist_item;
    auto isfound = find_win(wid, bcast_winlist_item);

    if (isfound == false) {
        std::cerr << "Window " << wid << " was not found" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, RET_CODE_ERROR);
    }

    std::cerr << "DEBUG 11" << std::endl;

    // Wait until passive synchronization epoch starts in waiter
    sp->get_fut().wait();
    std::cerr << "DEBUG 22" << std::endl;

    put_loop(req, descr, (buf_dtype *) origin_addr, myrank, sp->get_peer_count(), 
             sp->get_reqwin(), sp->get_datawin(), *bcast_winlist_item.win_sptr);
    std::cerr << "DEBUG 33" << std::endl;

    return RET_CODE_SUCCESS;
}


////////////////////////////////////////////////////////////////////
// Class for waiter thread for binomial broadcast -- implementation
//

// waiter_loop: Waiter thread on each process. 
// Active on each thread except root.
void waiter_c::waiter_loop_shmem()
{
    auto req_sptr = req_win_g.get_sptr();
    auto descr_sptr = descr_win_g.get_sptr();

    auto &req_raw_win = req_win_g.get_win();
    auto &descr_raw_win = descr_win_g.get_win();

    // Allocate and init req_read (array of operations for all procs)
    // TODO replace to vector
    // std::shared_ptr<req_t[]> req_read_sptr(new req_t[peer_count]);
    // decltype(auto) req_read = req_read_sptr.get(); // remove decltype?

    std::vector<req_t> req_read(nproc, req_t::noop);

    // for (auto rank = 0; rank < peer_count; rank++) {
    //     req_read[rank] = req_t::noop;
    // }

    RMA_Lock_guard lock_all_waiters_req(req_raw_win);
    RMA_Lock_guard lock_all_waiters_descr(descr_raw_win);

    ready_prom.set_value();

    while (waiter_term_flag == false) {
        // 1. Check request field

        // Atomically read (in my memory) request array for all processes 
        const auto req_arr_sz = req_size * peer_count;

        MPI_Get_accumulate(NULL, 0, MPI_BYTE, 
                           req_read.data(), req_arr_sz, 
                           MPI_BYTE, myrank, 0, req_arr_sz, MPI_BYTE, MPI_NO_OP,
                           req_raw_win);

        MPI_Win_flush(myrank, req_raw_win);

        // 1. Look through request array and find all requests
        // 2. Get data for active requests
        // 3. Put requests to the next processes
        for (auto rank = 0; rank < peer_count; rank++) {
            if (req_read[rank] == req_t::bcast) {
                req_sptr[rank] = req_t::noop;

                // Search for bcast window id in the windows list
                winlist_item_t bcast_winlist_item;
                auto isfound = find_win(descr_sptr[rank].wid, 
                                        bcast_winlist_item);

                if (isfound == false) {
                    std::cerr << "Window " << descr_sptr[rank].wid 
                              << " was not found" << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, RET_CODE_ERROR);
                }

                // Put request to all next peers
                put_loop(req_read[rank], descr_sptr[rank], 
                         (buf_dtype *) bcast_winlist_item.bufptr.get(), 
                         myrank, peer_count, req_raw_win, 
                         descr_raw_win, *bcast_winlist_item.win_sptr);

                // Increment finalization counter on root
                auto donecntr_win = donecntr_win_g.get_win();
                RMA_Lock_guard lock_root(descr_sptr[rank].root, donecntr_win);

                auto val_to_incr = 1;
                auto result = 0;
                MPI_Fetch_and_op(&val_to_incr, &result, MPI_INT, 
                                 descr_sptr[rank].root, 0, MPI_SUM, donecntr_win);

                lock_root.unlock();
            }
        }

        usleep(waiter_timeout);
    }
}
