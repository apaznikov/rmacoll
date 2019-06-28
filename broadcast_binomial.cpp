//
// broadcast_binomial.cpp: Binomial tree algorithm for RMA broadcast
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"
#include "broadcast_binomial.h"

// Avoid global pointers?
std::weak_ptr<waiter_c> waiter_weak_ptr;

// Avoid global variables?
// waiter_c waiter;

template <typename T>
static void printbin(T val)
{
    std::bitset<sizeof(val) * 8> bits(val);
    std::cout << "mask\t" << bits << std::endl;
}

// comp_srank: Compute rank relative to root
int comp_srank(int myrank, int root, int nproc)
{
    return (myrank - root + nproc) % nproc;
}

// comp_rank: Compute rank from srank
int comp_rank(int srank, int root, int nproc)
{
    return (srank + root) % nproc;
}

static void put_request(const req_t &req, int rank, int root, const MPI_Win &win)
{
    const auto disp = req_size * root;
    MPI_Accumulate(&req, req_size, MPI_BYTE, rank, disp, req_size, MPI_BYTE, 
                   MPI_REPLACE, win);

    MPI_Win_flush(rank, win);
}

// Put requests to all peers
static void put_loop(const req_t &req, const MPI_Win &win, 
                     int myrank, int nproc)
{
    auto srank = comp_srank(myrank, req.root, nproc);
    // std::cout << myrank << " my srank: " << srank << std::endl;

    auto mask = 1;

    while (mask < nproc) {
        if ((srank & mask) == 0) {
            // Put (send) data to the next process if bit is not set
            auto put_rank = srank | mask;
            // std::cout << myrank << "R put_rank = " << put_rank << std::endl;
            if (put_rank < nproc) {
                put_rank = comp_rank(put_rank, req.root, nproc);
                put_request(req, put_rank, req.root, win);
                std::cout << myrank << "R\tPUT to " << put_rank << std::endl;
            }
        } else {
            // If bit is set, break
            break;
        }

        mask = mask << 1;
    }
}

// RMA_Bcast_binomial: Binomial tree broadcast
// (?) Remove MPI_Win argument?
int RMA_Bcast_binomial(const void *origin_addr, int origin_count, 
                       MPI_Datatype origin_datatype, 
                       MPI_Aint target_disp, int target_count, 
                       MPI_Datatype target_datatype,
                       MPI_Win win, win_id_t wid, MPI_Comm comm)
{
    auto sp = waiter_weak_ptr.lock();

    if (!sp) {
        std::cerr << "waiter_weak_ptr is expired" << std::endl;
        return RET_CODE_ERROR;
    }

    auto &winguard = sp->get_winguard();
    auto &req_raw_win = winguard.get_win();
    auto req_ptr = winguard.get_sptr();

    // Set request fields
    req_t req;

    auto size = 0;
    MPI_Type_size(origin_datatype, &size);
    // std::cout << "size " << size << " count " << origin_count 
    //           << " sizeof " << sizeof(req.buf) << std::endl;
    memcpy(&req.buf, origin_addr, size * origin_count);

    auto myrank = 0, nproc = 0; 
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nproc);

    req.bufsize = size;
    req.op = req_t::op_t::bcast;
    req.root = myrank;
    req.wid = wid;

    // Wait until synch epoch starts in waiter
    sp->get_fut().wait();

    put_loop(req, req_raw_win, myrank, nproc);

    return RET_CODE_SUCCESS;
}


////////////////////////////////////////////////////////////////////
// Class for waiter thread for binomial broadcast -- implementation
//

// waiter_loop: Waiter thread on each process. 
// Active on each thread except root.
void waiter_c::waiter_loop()
{
    auto req_sptr = req_win.get_sptr();

    // Init requests
    for (auto i = 0; i < nproc; i++) {
        req_sptr[i].op = req_t::noop;
    }

    auto &req_raw_win = req_win.get_win();

    // Allocate req_read (array of reqeusts for all procs)
    std::shared_ptr<req_t[]> req_read_sptr(new req_t[nproc]);
    decltype(auto) req_read = req_read_sptr.get();

    for (auto rank = 0; rank < nproc; rank++)
        req_read[rank].op = req_t::op_t::noop;

    // if (myrank == 2) {
    //     for (auto i = 0; i < nproc; i++) {
    //         std::cout << myrank << "R " << i << " " << req_read[i].op << std::endl;
    //         std::cout << myrank << "R " << i << " " << req_sptr[i].op << std::endl;
    //     }
    // }

    const auto req_arr_sz = sizeof(req_t) * nproc;

    RMA_Lock_guard lock_all_waiters(req_raw_win);

    ready_prom.set_value();

    while (waiter_term_flag == false) {
        // 1. Check request field

        // ?? Devide quering of req devide by lurking (common load) 
        // and attack (MPI_Get_acc) phases

        // req_t req_read;
        // req_read.op = req_t::op_t::noop;

        // Atomically read request field in my memory

        MPI_Get_accumulate(NULL, 0, MPI_BYTE, 
                           req_read, req_arr_sz, 
                           MPI_BYTE, myrank, 0, req_arr_sz, MPI_BYTE, MPI_NO_OP, 
                           req_raw_win);

        MPI_Win_flush(myrank, req_raw_win);

        // if (myrank == 2)
        //     for (auto i = 0; i < nproc; i++) 
        //         std::cout << myrank << "R " << i << " " << req_read[i].op << std::endl;


        // Look through request array and find all requests
        for (auto rank = 0; rank < nproc; rank++) {
            if (req_read[rank].op == req_t::op_t::bcast) {
                req_sptr[rank].op = req_t::op_t::noop;

                std::cout << myrank << "R GOT op = " << req_read[rank].op 
                          << " buf " << req_read[rank].buf 
                          << " root " << req_read[rank].root 
                          << " wid " << req_read[rank].wid << std::endl;

                // Copy buf to local memory
                // Search for bcast windows id in windows list
                // MPI_Win *bcast_win = nullptr;

                std::shared_ptr<MPI_Win> bcast_win;
                auto isfound = find_win(req_read[rank].wid, bcast_win);

                if (isfound == false) {
                    std::cerr << "Window " << req_read[rank].wid 
                              << " was not found" << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, RET_CODE_ERROR);
                }

                // std::cout << myrank << "R found " << (void*) bcast_win << std::endl;

                // Atomically set local value
                RMA_Lock_guard lock(myrank, *bcast_win);

                MPI_Accumulate(&req_read[rank].buf, 
                               req_read[rank].bufsize, MPI_BYTE, 
                               myrank, offset_buf, 
                               req_read[rank].bufsize, MPI_BYTE, MPI_REPLACE, 
                               *bcast_win);

                lock.unlock();

                // Put request to all next peers
                put_loop(req_read[rank], req_raw_win, myrank, nproc);
            }
        }

        usleep(waiter_timeout);
    }
}
