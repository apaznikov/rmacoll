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
    std::cout << bits << std::endl;
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
    
    if (myrank == 0) {

        auto mask = 1;

        while (mask < nproc) {
            std::cout << "\t mask\t";
            printbin(mask);
            mask = mask << 1;
        }

        // decltype(auto) winguard = waiter.get_winguard();

        auto sp = waiter_wp.lock();

        if (!sp) {
            std::cerr << "waiter_wp is expired" << std::endl;
            return RET_CODE_ERROR;
        }

        // decltype(auto) winguard = sp->get_winguard();
        // decltype(auto) req_win = winguard.get_win();
        // decltype(auto) req_ptr = winguard.get_sptr();
        
        auto &winguard = sp->get_winguard();
        auto &req_win = winguard.get_win();
        auto req_ptr = winguard.get_sptr();

        // auto winguard = waiter_wp.lock()->get_winguard();

        // auto req_win = sp->get_winguard().get_win();
        // auto req_ptr = sp->get_winguard().get_sptr();

        auto orig_req = 1;

        RMA_Lock_guard lock_all_waiters(req_win);
    
        for (auto rank = 0; rank < nproc; rank++) {
            MPI_Put(&orig_req, 1, MPI_INT, rank, <CHANGE TO DISP>, 1, MPI_INT, req_win);
        }
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
    
    auto req_ptr = req_win.get_sptr();
    req_ptr->op = 0;
    req_ptr->root = 0;

    while (waiter_term_flag == false) {
        // std::cout << myrank << " hello " << i++ << std::endl;

        // 
        // 1. Check request field
        //

        if (req_ptr->op == 1) {
            std::cout << myrank << "\t I GOT IT!" << *req_ptr << std::endl;
            *req_ptr = 0;

            // Compute rank relative to root
            // auto srand = (myrank - root +p) % p;

            auto mask = 1;

            while (mask < nproc) {
                std::cout << "\t mask\t";
                printbin(mask);
                mask = mask << 1;
            }
        }

        usleep(1000);
    }
}
