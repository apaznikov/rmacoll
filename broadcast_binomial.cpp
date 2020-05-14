//
// broadcast_binomial.cpp: Binomial tree algorithm for RMA broadcast
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <mpi.h>

#include "rmacoll.h"
#include "rmautils.h"
#include "broadcast_binomial.h"

extern int iter;

// Avoid global pointers?
std::weak_ptr<waiter_c> waiter_weak_ptr;

// #define _DEBUG
// #define _PROF

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


// bcast_isdone: Read and check flag array
bool bcast_isdone(int peer_count, int myrank, MPI_Win &donecntr_win)
{
    int read_cntr = 0;
    MPI_Fetch_and_op(NULL, &read_cntr, MPI_INT, myrank, 0, MPI_NO_OP, 
                     donecntr_win);

    if (read_cntr == peer_count - 1)
        return true;
    else
        return false;
}

// RMA_Bcast_test: Test if RMA bcast is done
int RMA_Bcast_test(bool &done)
{
    auto sp = waiter_weak_ptr.lock();

    if (!sp) {
        std::cerr << "waiter_weak_ptr is expired" << std::endl;
        return RET_CODE_ERROR;
    }

    auto &doneflag_win = sp->get_donecntrwin();

    // Atomically lock myself and read doneflag array
    RMA_Lock_guard lock_root(myrank, doneflag_win);

    auto peer_count = sp->get_peer_count();
    auto myrank = sp->get_myrank();

    done = bcast_isdone(peer_count, myrank, doneflag_win);

    return RET_CODE_SUCCESS;
}

// RMA_Bcast_flush: Wait until RMA_Bcast is completed
int RMA_Bcast_flush()
{
    return RET_CODE_SUCCESS;

    auto sp = waiter_weak_ptr.lock();

    if (!sp) {
        std::cerr << "waiter_weak_ptr is expired" << std::endl;
        return RET_CODE_ERROR;
    }

    auto &doneflag_win = sp->get_donecntrwin();

    // Atomically lock myself and read doneflag array
    RMA_Lock_guard lock_root(myrank, doneflag_win);

    auto peer_count = sp->get_peer_count();
    auto myrank = sp->get_myrank();

    while (!bcast_isdone(peer_count, myrank, doneflag_win)) {
        usleep(flush_timeout);
    };

    return RET_CODE_SUCCESS;
}

// put_req_data: Put request and data into remote memory
static void put_req_data(const req_t &req, const descr_t &descr, 
                         buf_dtype *buf, int put_rank, 
                         const MPI_Win &req_win, 
                         const MPI_Win &descr_win,
                         const MPI_Win &bcast_win)
{
#ifdef _PROF
    auto t4 = MPI_Wtime();
#endif

    // 1. Put buffer (data)
    // // TODO here disp should be equal 0
    MPI_Put(buf, descr.bufsize, MPI_BYTE, put_rank, 0, 
            descr.bufsize, MPI_BYTE, bcast_win);

#ifdef _PROF
    auto t5 = MPI_Wtime();
    std::cout << myrank << "R i " << iter << " PUT1 " << t5 - t4 << std::endl;
#endif

    // FIXME why disp here is 0?? should be
    // 2. Put description
    auto disp = descr_t_size * descr.root;
    MPI_Put(&descr, sizeof(descr_t), MPI_BYTE, put_rank, disp, 
            sizeof(descr_t), MPI_BYTE, descr_win);

#ifdef _PROF
    auto t6 = MPI_Wtime();
    std::cout << myrank << "R i " << iter << " PUT2 " << t6 - t5 << std::endl;
#endif

    // 3. Put request (flag) into remote memory
    disp = req_size * descr.root;
    MPI_Accumulate(&req, req_size, MPI_BYTE, put_rank, disp, req_size, MPI_BYTE,
                   MPI_REPLACE, req_win);

    // ?? Optimize: put flush after each MPI_Put or here after the MPI_Acc?
    MPI_Win_flush(put_rank, bcast_win);
    MPI_Win_flush(put_rank, descr_win);
    MPI_Win_flush(put_rank, req_win);

#ifdef _PROF
    auto t7 = MPI_Wtime();
    std::cout << myrank << "R i " << iter << " ACC " << t7 - t6 << std::endl;
#endif
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
#ifdef _DEBUG
                std::cout << myrank << "R\tPUT to " << put_rank << std::endl;
#endif

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
int RMA_Bcast_binomial(const void *origin_addr, int origin_count, 
                       MPI_Datatype origin_datatype, 
                       MPI_Aint target_disp, int target_count, 
                       MPI_Datatype target_datatype,
                       win_id_t wid, MPI_Comm comm)
{
    // Wait until previous bcast is completed

#ifdef _PROF
    auto t1 = MPI_Wtime();
#endif

    std::cerr << myrank << "R b 11\n";

    RMA_Bcast_flush();

    std::cerr << myrank << "R b 12\n";

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

    // Wait until passive synchronization epoch starts in waiter
    sp->get_fut().wait();

#ifdef _PROF
    auto t5 = MPI_Wtime();
    std::cout << myrank << "R i " << iter 
              << " WHOLE NO LOOP " << t5 - t1 << std::endl;
#endif

    std::cerr << myrank << "R b 13\n";

    put_loop(req, descr, (buf_dtype *) origin_addr, myrank, nproc, 
            sp->get_reqwin(), sp->get_datawin(), *bcast_winlist_item.win_sptr);

    std::cerr << myrank << "R b 14\n";

    return RET_CODE_SUCCESS;
}


////////////////////////////////////////////////////////////////////
// Class for waiter thread for binomial broadcast -- implementation
//

// waiter_loop: Waiter thread on each process. 
// Active on each thread except root.
void waiter_c::waiter_loop()
{
    auto req_sptr = req_win_g.get_sptr();
    auto descr_sptr = descr_win_g.get_sptr();

    auto &req_raw_win = req_win_g.get_win();
    auto &descr_raw_win = descr_win_g.get_win();

    // Allocate and init req_read (array of operations for all procs)
    std::vector<req_t> req_read(nproc, req_t::noop);

    RMA_Lock_guard lock_all_waiters_req(req_raw_win);
    RMA_Lock_guard lock_all_waiters_descr(descr_raw_win);

    ready_prom.set_value();

// #ifdef _PROF
//     auto t2 = MPI_Wtime();
//     std::cout << myrank << "R 11 " << t2 - t1 << std::endl << std::endl;
// #endif

    while (waiter_term_flag == false) {
        // 1. Check request field

        // ?? Devide quering of req devide by lurking (common load) 
        // and attack (MPI_Get_acc) phases

        // Atomically read (in my memory) request array for all processes 
        const auto req_arr_sz = req_size * nproc;

        // auto t3 = MPI_Wtime();

        // memcpy(req_read, req_sptr.get(), req_arr_sz);

        // auto t4 = MPI_Wtime();
        // std::cout << myrank << "R MEMCPY " << t4 - t3 << std::endl;

        MPI_Get_accumulate(NULL, 0, MPI_BYTE, 
                           req_read.data(), req_arr_sz, 
                           MPI_BYTE, myrank, 0, req_arr_sz, MPI_BYTE, MPI_NO_OP,
                           req_raw_win);

        MPI_Win_flush(myrank, req_raw_win);

        // auto t5 = MPI_Wtime();
        // std::cout << myrank << "R GET REQ " << t5 - t4 << std::endl;

        // 1. Look through request array and find all requests
        // 2. Get data for active requests
        // 3. Put requests to the next processes
        for (auto rank = 0; rank < nproc; rank++) {
            if (req_read[rank] == req_t::bcast) {
                req_sptr[rank] = req_t::noop;

#ifdef _DEBUG
                std::cout << myrank << "R GOT op = " << req_read[rank].op 
                          << " size " << req_read[rank].bufsize 
                          << std::endl;
                          // << " buf " << op_read[rank].buf 
                          // << " root " << op_read[rank].root 
                          // << " wid " << op_read[rank].wid << std::endl;
#endif

#ifdef _DEBUG
                std::cout << myrank << "R buf = " << descr_read->buf[0]
                          << " sptrbuf " << descr_sptr[rank].buf[0] 
                          << " data_size " << descr_t_size
                          << " root " << descr_read->root
                          << " sptr root " << descr_sptr[rank].root
                          << " wid " << descr_read->wid
                          << std::endl;
#endif

#ifdef _PROF
                auto t5 = MPI_Wtime();
#endif

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
                         myrank, nproc, req_raw_win, 
                         descr_raw_win, *bcast_winlist_item.win_sptr);

#ifdef _PROF
                auto t6 = MPI_Wtime();
                std::cout << myrank << "R i " << iter << " PUTLOOP " 
                          << t6 - t5 << std::endl;
#endif

                // Increment finalization counter on root
                auto donecntr_win = donecntr_win_g.get_win();
                RMA_Lock_guard lock_root(descr_sptr[rank].root, donecntr_win);

                auto val_to_incr = 1;
                auto result = 0;
                MPI_Fetch_and_op(&val_to_incr, &result, MPI_INT, 
                                 descr_sptr[rank].root, 0, MPI_SUM, donecntr_win);

                lock_root.unlock();

#ifdef _PROF
                auto t7 = MPI_Wtime();
                std::cout << myrank << "R i " << iter << " FOP " << t7 - t6 << std::endl;
#endif
            }
        }

        usleep(waiter_timeout);
    }
}
