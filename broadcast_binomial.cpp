//
// broadcast_binomial.cpp: Binomial tree algorithm for RMA broadcast
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <mpi.h>

#include "rmacoll.h"
#include "broadcast_binomial.h"

// Avoid global pointers?
std::weak_ptr<waiter_c> waiter_wp;

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

// RMA_Bcast_binomial: Binomial tree broadcast
int RMA_Bcast_binomial(const void *origin_addr, int origin_count, 
                       MPI_Datatype origin_datatype, MPI_Aint target_disp,
                       int target_count, MPI_Datatype target_datatype,
                       MPI_Win win, MPI_Comm comm)
{
    auto myrank = 0, nproc = 0; 

    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nproc);
    
    auto sp = waiter_wp.lock();

    if (!sp) {
        std::cerr << "waiter_wp is expired" << std::endl;
        return RET_CODE_ERROR;
    }

    auto &winguard = sp->get_winguard();
    auto &req_raw_win = winguard.get_win();
    auto req_ptr = winguard.get_sptr();

    // auto winguard = waiter_wp.lock()->get_winguard();
    // auto req_win = sp->get_winguard().get_win();
    // auto req_ptr = sp->get_winguard().get_sptr();

    // RMA_Lock_guard lock_all_waiters(req_raw_win);

    auto mask = 1;

    while (mask < nproc) {
        printbin(mask);

        // Put data to ranks in which current bit is not set
        auto rank_put = comp_rank(mask, myrank, nproc);
        std::cout << " rank_put2 " << rank_put << std::endl;

        // Prepare request
        req_t req;
        req.buf = *((int*) origin_addr);
        req.op = req_t::op_t::bcast;
        req.root = myrank;
        auto count = sizeof(req_t);

        // Atomic put to request on remote process
        MPI_Accumulate(&req, count, MPI_BYTE, rank_put, 0, count, MPI_BYTE,
                       MPI_REPLACE, req_raw_win);

        MPI_Win_flush(rank_put, req_raw_win);

        // MPI_Put(&req, count, MPI_BYTE, rank_put, 0, count, MPI_BYTE, req_win);

        // MPI_Put(&op, 1, MPI_INT, rank_put, offset_op, 1, MPI_INT, req_win);

        mask = mask << 1;
    }

    return RET_CODE_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Class for waiter thread for binomial broadcast -- implementation
//

// waiter_loop: Waiter thread on each process
void waiter_c::waiter_loop()
{
    // std::cout << "hello0" << std::endl;
    auto req_sptr = req_win.get_sptr();
    req_sptr->op = req_t::noop;
    req_sptr->root = 0;
    auto &req_raw_win = req_win.get_win();

    while (waiter_term_flag == false) {
        // std::cout << myrank << " hello " << i++ << std::endl;
        // std::cout << myrank << " op " << req_ptr->op << std::endl;

        // 1. Check request field

        // Devide quering of req devide by lurking (common load) 
        // and attack (get_acc) phases

        if (myrank == 0)
            continue;

        req_t req_read;
        req_read.op = req_t::op_t::noop;
        auto count = sizeof(req_t);

        // // Atomically read request field in my memory
        MPI_Get_accumulate(req_sptr.get(), count, MPI_BYTE, &req_read, count, 
                           MPI_BYTE, myrank, 0, count, MPI_BYTE, MPI_NO_OP, 
                           req_raw_win);

        MPI_Win_flush(myrank, req_raw_win);

        if (req_read.op == req_t::op_t::bcast) {
            std::cout << myrank << "\t I GOT IT! op = " << req_sptr->op 
                      << " buf " << req_sptr->buf 
                      << " root " << req_sptr->root << std::endl;

            auto root = req_sptr->root;
            // auto buf = req_sptr->buf;

            req_sptr->op = req_t::op_t::noop;

            auto srank = comp_srank(myrank, req_sptr->root, nproc);
            std::cout << myrank << " my srank: " << srank << std::endl;

            auto mask = 1;

            while (mask < nproc) {
                if ((srank & mask) == 0) {
                    // Put (send) data if bit is not set
                    auto put_rank = srank | mask;
                    if (put_rank < nproc) {
                        // std::cout << myrank << "\t1 put to " << put_rank << std::endl;
                        put_rank = comp_rank(put_rank, root, nproc);
                        std::cout << myrank << "\tput to " << put_rank << std::endl;
                    }
                } else {
                    // Get (recv) data if bit is set
                    auto get_rank = srank & (~mask);
                    get_rank = comp_rank(get_rank, root, nproc);
                    std::cout << myrank << "\tget from " << get_rank << std::endl;
                    break;
                }

                mask = mask << 1;
            }
        }

        usleep(1000);
    }
}
