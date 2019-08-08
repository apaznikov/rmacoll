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

// #define _DEBUG

// Auxiliary structure to avoid buffering in the root process
struct data_ptr_t {
    int *buf = nullptr;
    int root = -1;
    win_id_t wid = MPI_WIN_NO_ID;
};

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
bool bcast_isdone(int nproc, int myrank, MPI_Win &donecntr_win)
{
    int read_cntr = 0;
    MPI_Fetch_and_op(NULL, &read_cntr, MPI_INT, myrank, 0, MPI_NO_OP, 
                     donecntr_win);

    if (read_cntr == nproc - 1)
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

    auto nproc = sp->get_nproc();
    auto myrank = sp->get_myrank();

    done = bcast_isdone(nproc, myrank, doneflag_win);

    return RET_CODE_SUCCESS;
}

// RMA_Bcast_flush: Wait until RMA_Bcast is completed
int RMA_Bcast_flush()
{
    auto sp = waiter_weak_ptr.lock();

    if (!sp) {
        std::cerr << "waiter_weak_ptr is expired" << std::endl;
        return RET_CODE_ERROR;
    }

    auto &doneflag_win = sp->get_donecntrwin();

    // Atomically lock myself and read doneflag array
    RMA_Lock_guard lock_root(myrank, doneflag_win);

    auto nproc = sp->get_nproc();
    auto myrank = sp->get_myrank();

    while (!bcast_isdone(nproc, myrank, doneflag_win)) {
        usleep(flush_timeout);
    };

    return RET_CODE_SUCCESS;
}

// put_request: Put request and data into remote memory
static void put_request(const req_t &req, const data_ptr_t &data, int rank, 
                        int root, const MPI_Win &req_win, 
                        const MPI_Win &data_win)
{
    auto disp = data_t_size * root;
#ifdef _DEBUG
    std::cout << "disp " << disp << " bufsize " << req.bufsize
              << " data " << data->buf[0] << std::endl;
#endif
    auto t4 = MPI_Wtime();

    // Put data
    MPI_Put(data.buf, req.bufsize, MPI_BYTE, rank, disp, req.bufsize, MPI_BYTE, 
            data_win);

    MPI_Put(&data.root, root_wid_size, MPI_BYTE, rank, disp + offset_root_wid,
            root_wid_size, MPI_BYTE, data_win);

    MPI_Win_flush(rank, data_win);

    auto t5 = MPI_Wtime();
    std::cout << myrank << "R PUT " << t5 - t4 << std::endl;

    // Put request (flag) into remote memory
    disp = req_size * root;
    MPI_Accumulate(&req, req_size, MPI_BYTE, rank, disp, req_size, MPI_BYTE,
                   MPI_REPLACE, req_win);

    MPI_Win_flush(rank, req_win);

    auto t6 = MPI_Wtime();
    std::cout << myrank << "R ACC " << t6 - t5 << std::endl;
}

// Put requests to all peers
static void put_loop(const req_t &req, const data_ptr_t &data, 
                     int myrank, int nproc, const MPI_Win &req_win, 
                     const MPI_Win &data_win)
{
    auto srank = comp_srank(myrank, data.root, nproc);

    auto mask = 1;

    while (mask < nproc) {
        if ((srank & mask) == 0) {
            // Put (send) data to the next process if bit is not set
            auto put_rank = srank | mask;
            if (put_rank < nproc) {
                put_rank = comp_rank(put_rank, data.root, nproc);
#ifdef _DEBUG
                std::cout << myrank << "R\tPUT to " << put_rank << std::endl;
#endif

                put_request(req, data, put_rank, data.root, req_win, data_win);
            }
        } else {
            // If bit is set, break 
            // (in original non-rma algorithm it's receive phase)
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
    // Wait until previous bcast is completed
    auto t1 = MPI_Wtime();
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

    // Set operation type
    req_t req;
    req.op = req_t::op_t::bcast;
    req.bufsize = origin_count * type_size;

    // Wait until passive synchronization epoch starts in waiter
    sp->get_fut().wait();

#ifdef _PROF
    auto t4 = MPI_Wtime();
#endif

    // Auxiliary structure to store pointer to the buffer
    data_ptr_t data_ptr;

    data_ptr.buf = (buf_dtype *) origin_addr;
    data_ptr.root = myrank;
    data_ptr.wid = wid;

    auto t5 = MPI_Wtime();
    std::cout << myrank << "R WHOLE NO LOOP " << t5 - t1 << std::endl;

    put_loop(req, data_ptr, myrank, nproc, sp->get_reqwin(), sp->get_datawin());

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
    auto data_sptr = data_win_g.get_sptr();

    auto &req_raw_win = req_win_g.get_win();
    auto &data_raw_win = data_win_g.get_win();

    // Allocate and init req_read (array of operations for all procs)
    std::shared_ptr<req_t[]> req_read_sptr(new req_t[nproc]);
    decltype(auto) req_read = req_read_sptr.get(); // remove decltype?

    for (auto rank = 0; rank < nproc; rank++) {
        req_read[rank].op = req_t::op_t::noop;
        req_read[rank].bufsize = 0;
    }

    RMA_Lock_guard lock_all_waiters_req(req_raw_win);
    RMA_Lock_guard lock_all_waiters_data(data_raw_win);

    ready_prom.set_value();

#ifdef _PROF
    auto t2 = MPI_Wtime();
    std::cout << myrank << "R 11 " << t2 - t1 << std::endl << std::endl;
#endif

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
                           req_read, req_arr_sz, 
                           MPI_BYTE, myrank, 0, req_arr_sz, MPI_BYTE, MPI_NO_OP,
                           req_raw_win);

        MPI_Win_flush(myrank, req_raw_win);

        // auto t5 = MPI_Wtime();
        // std::cout << myrank << "R GET REQ " << t5 - t4 << std::endl;

        // 1. Look through request array and find all requests
        // 2. Get data for active requests
        // 3. Put requests to the next processes
        for (auto rank = 0; rank < nproc; rank++) {
            if (req_read[rank].op == req_t::op_t::bcast) {
                req_sptr[rank].op = req_t::op_t::noop;

#ifdef _DEBUG
                std::cout << myrank << "R GOT op = " << req_read[rank].op 
                          << " size " << req_read[rank].bufsize 
                          << std::endl;
                          // << " buf " << op_read[rank].buf 
                          // << " root " << op_read[rank].root 
                          // << " wid " << op_read[rank].wid << std::endl;
#endif

#ifdef _DEBUG
                std::cout << myrank << "R buf = " << data_read->buf[0]
                          << " sptrbuf " << data_sptr[rank].buf[0] 
                          << " data_size " << data_t_size
                          << " root " << data_read->root
                          << " sptr root " << data_sptr[rank].root
                          << " wid " << data_read->wid
                          << std::endl;
#endif

#ifdef _DEBUG
                std::cout << myrank << "buf\t";
                for (auto i = 0u; i < req_read[rank].bufsize / sizeof(int); i++)
                    std::cout << data_sptr[rank].buf[i] << " ";
                std::cout << std::endl;
#endif
                auto t5 = MPI_Wtime();

                // put_loop(req_read[rank], data_read.get(), myrank, nproc, 
                //          req_raw_win, data_raw_win);

                // Auxiliary structure to store pointer to the buffer
                data_ptr_t data_ptr;

                data_ptr.buf = data_sptr[rank].buf;
                data_ptr.root = data_sptr[rank].root;
                data_ptr.wid = data_sptr[rank].wid;

                // Put request to all next peers
                put_loop(req_read[rank], data_ptr, myrank, nproc, 
                         req_raw_win, data_raw_win);

                auto t6 = MPI_Wtime();
                std::cout << myrank << "R PUTLOOP " << t6 - t5 << std::endl;

                // Copy buf to local memory
                
                // Search for bcast window id in the windows list
                // std::shared_ptr<MPI_Win> bcast_win;
                winlist_item_t item;
                auto isfound = find_win(data_sptr[rank].wid, item);

                if (isfound == false) {
                    std::cerr << "Window " << data_sptr[rank].wid
                              << " was not found" << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, RET_CODE_ERROR);
                }

                auto t7 = MPI_Wtime();
                std::cout << myrank << "R FIND " << t7 - t6 << std::endl;

                memcpy(item.bufptr.get(), data_sptr[rank].buf, req_read[rank].bufsize);

                auto t9 = MPI_Wtime();
                std::cout << myrank << "R CPY " << t9 - t7 << std::endl;

                auto donecntr_win = donecntr_win_g.get_win();
                RMA_Lock_guard lock_root(data_sptr[rank].root, donecntr_win);

                auto val_to_incr = 1;
                auto result = 0;
                MPI_Fetch_and_op(&val_to_incr, &result, MPI_INT, 
                                 data_sptr[rank].root, 0, MPI_SUM, donecntr_win);

                lock_root.unlock();

                auto t10 = MPI_Wtime();
                std::cout << myrank << "R FOP " << t10 - t9 << std::endl;
            }
        }

        usleep(waiter_timeout);
    }
}
