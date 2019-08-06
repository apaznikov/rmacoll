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


// bcast_isdone: Read and check flag array
bool bcast_isdone(int nproc, int myrank, bool doneflags[], 
                      MPI_Win &doneflag_win)
{
    MPI_Get_accumulate(NULL, 0, MPI_BYTE,
                       doneflags, nproc,
                       MPI_BYTE, myrank, 0, nproc, MPI_BYTE, MPI_NO_OP,
                       doneflag_win);

    MPI_Win_flush(myrank, doneflag_win);

    // std::cout << myrank << "R doneflags " << nproc << " ";
    for (auto i = 0; i < nproc; i++) {
        // std::cout << doneflags[i] << " ";
        if (doneflags[i] == false) {
            // std::cout << std::endl;
            return false;
        }
    }

    // std::cout << std::endl;
    return true;
}

// RMA_Bcast_test: Test if RMA bcast is done
int RMA_Bcast_test(bool &done)
{
    auto sp = waiter_weak_ptr.lock();

    if (!sp) {
        std::cerr << "waiter_weak_ptr is expired" << std::endl;
        return RET_CODE_ERROR;
    }

    auto &doneflag_win = sp->get_doneflagwin();

    // Atomically lock myself and read doneflag array
    RMA_Lock_guard lock_root(myrank, doneflag_win);

    auto nproc = sp->get_nproc();
    auto myrank = sp->get_myrank();
    bool doneflags[nproc];

    done = bcast_isdone(nproc, myrank, doneflags, doneflag_win);

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

    auto &doneflag_win = sp->get_doneflagwin();

    // Atomically lock myself and read doneflag array
    RMA_Lock_guard lock_root(myrank, doneflag_win);

    auto nproc = sp->get_nproc();
    auto myrank = sp->get_myrank();
    bool doneflags[nproc];

    while (!bcast_isdone(nproc, myrank, doneflags, doneflag_win));

    return RET_CODE_SUCCESS;
}

// put_request: Put request and data into remote memory
static void put_request(const req_t &req, data_t *data, int rank, int root, 
                        const MPI_Win &req_win, const MPI_Win &data_win)
{
    auto disp = data_t_size * root;
#ifdef _DEBUG
    std::cout << "disp " << disp << " bufsize " << req.bufsize
              << " data " << data->buf[0] << std::endl;
#endif

    // Put data
    // TODO devide into Acc for buf and root, wid
    MPI_Accumulate(data, req.bufsize, MPI_BYTE, rank, disp, req.bufsize, 
                   MPI_BYTE, MPI_REPLACE, data_win);

    MPI_Accumulate(&data->root, root_wid_size, MPI_BYTE, rank, 
                   disp + offset_root_wid, root_wid_size, MPI_BYTE, 
                   MPI_REPLACE, data_win);

    MPI_Win_flush(rank, data_win);

    // Put request (flag) into remote memory
    disp = req_size * root;
    MPI_Accumulate(&req, req_size, MPI_BYTE, rank, disp, req_size, MPI_BYTE,
                   MPI_REPLACE, req_win);

    MPI_Win_flush(rank, req_win);
}

// Put requests to all peers
static void put_loop(const req_t &req, data_t *data, int myrank, int nproc, 
                     const MPI_Win &req_win, const MPI_Win &data_win)
{
    auto srank = comp_srank(myrank, data->root, nproc);

    auto mask = 1;

    while (mask < nproc) {
        if ((srank & mask) == 0) {
            // Put (send) data to the next process if bit is not set
            auto put_rank = srank | mask;
            if (put_rank < nproc) {
                put_rank = comp_rank(put_rank, data->root, nproc);
#ifdef _DEBUG
                std::cout << myrank << "R\tPUT to " << put_rank << std::endl;
#endif

                put_request(req, data, put_rank, data->root, req_win, data_win);
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
    RMA_Bcast_flush();

    auto sp = waiter_weak_ptr.lock();

    if (!sp) {
        std::cerr << "waiter_weak_ptr is expired" << std::endl;
        return RET_CODE_ERROR;
    }

    sp->set_doneflags(false);

    auto myrank = 0, nproc = 0; 
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nproc);

    auto type_size = 0;
    MPI_Type_size(origin_datatype, &type_size);

    // Set operation type
    req_t req;
    req.op = req_t::op_t::bcast;
    req.bufsize = origin_count * type_size;

    // Set data fields
    auto data = std::make_unique<data_t>();
    data->root = myrank;
    data->wid = wid;

    // Copy memory to buffer
    memcpy(&data->buf, origin_addr, type_size * origin_count);

    // Wait until passive synchronization epoch starts in waiter
    sp->get_fut().wait();

    put_loop(req, data.get(), myrank, nproc, sp->get_reqwin(), sp->get_datawin());

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

    // Allocate data_read (array of requests for all procs)
    // std::shared_ptr<data_t[]> data_read_sptr(new data_t[nproc]);
    // decltype(auto) data_read = data_read_sptr.get(); // remove decltype?

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

    while (waiter_term_flag == false) {
        // 1. Check request field

        // ?? Devide quering of req devide by lurking (common load) 
        // and attack (MPI_Get_acc) phases

        // Atomically read (in my memory) request array for all processes 
        const auto req_arr_sz = req_size * nproc;

        MPI_Get_accumulate(NULL, 0, MPI_BYTE, 
                           req_read, req_arr_sz, 
                           MPI_BYTE, myrank, 0, req_arr_sz, MPI_BYTE, MPI_NO_OP,
                           req_raw_win);

        MPI_Win_flush(myrank, req_raw_win);

        // if (myrank == 2)
        //     for (auto i = 0; i < nproc; i++) 
        //         std::cout << myrank << "R " << i << " " << req_read[i].op << std::endl;

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

                // Read data from local memory
                const auto disp = data_t_size * rank;

                auto data_read = std::make_unique<data_t>();

                // Get buf with specified bufsize
                MPI_Get_accumulate(NULL, 0, MPI_BYTE, 
                                   data_read.get(), req_read[rank].bufsize, 
                                   MPI_BYTE, myrank, disp, 
                                   req_read[rank].bufsize, MPI_BYTE, 
                                   MPI_NO_OP, data_raw_win);

                // Get root and wid
                MPI_Get_accumulate(NULL, 0, MPI_BYTE, 
                                   &data_read->root, root_wid_size, 
                                   MPI_BYTE, myrank, disp + offset_root_wid, 
                                   root_wid_size, MPI_BYTE, MPI_NO_OP, 
                                   data_raw_win);

                MPI_Win_flush(myrank, data_raw_win);

                // data_sptr[0] because RMA_Win_guard always stores 
                // shared_ptr<T[]> for now
#ifdef _DEBUG
                std::cout << myrank << "R buf = " << data_read->buf[0]
                          << " sptrbuf " << data_sptr[0].buf[0] 
                          << " data_size " << data_t_size
                          << " root " << data_read->root
                          << " sptr root " << data_sptr[0].root
                          << " wid " << data_read->wid
                          << std::endl;
#endif

#ifdef _DEBUG
                std::cout << myrank << "buf\t";
                for (auto i = 0u; i < req_read[rank].bufsize / sizeof(int); i++)
                    std::cout << data_read->buf[i] << " ";
                std::cout << std::endl;
#endif


                // Copy buf to local memory
                
                // Search for bcast window id in the windows list
                std::shared_ptr<MPI_Win> bcast_win;
                auto isfound = find_win(data_read->wid, bcast_win);

                if (isfound == false) {
                    std::cerr << "Window " << data_read->wid 
                              << " was not found" << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, RET_CODE_ERROR);
                }

                // std::cout << myrank << "R found " << (void*) bcast_win << std::endl;

                // Atomically set my local value
                // Why cannot replace with memcpy?? 
                RMA_Lock_guard lock_myself(myrank, *bcast_win);

                MPI_Accumulate(&data_read->buf, 
                               req_read[rank].bufsize, MPI_BYTE, 
                               myrank, offset_buf, 
                               req_read[rank].bufsize, MPI_BYTE, MPI_REPLACE, 
                               *bcast_win);

                lock_myself.unlock();

                // Set complete flag to root
                auto doneflag_win = doneflag_win_g.get_win();
                RMA_Lock_guard lock_root(data_read->root, doneflag_win);

                auto flag = true;
                MPI_Accumulate(&flag, 1, MPI_BYTE, data_read->root, myrank, 
                               1, MPI_BYTE, MPI_REPLACE, doneflag_win);

                lock_root.unlock();

                // Put request to all next peers
                put_loop(req_read[rank], data_read.get(), myrank, nproc, 
                         req_raw_win, data_raw_win);
            }
        }

        usleep(waiter_timeout);
    }
}
