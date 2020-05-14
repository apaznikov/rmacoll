//
// rmautils.cpp: Utils for MPI RMA
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#pragma once

#include <iostream>
#include <functional>
#include <memory>
#include <map>
#include <mutex>

#include <mpi.h>

extern int myrank;

using win_id_t = int;

const win_id_t MPI_WIN_NO_ID = -1;

extern win_id_t last_wid;

struct winlist_item_t {
    std::shared_ptr<MPI_Win> win_sptr;
    std::shared_ptr<void> bufptr;
};

// using winlist_t = std::map<win_id_t, std::shared_ptr<MPI_Win>>;
using winlist_t = std::map<win_id_t, winlist_item_t>;

extern winlist_t winlist;

extern std::mutex winlock;

// Find window by window's id
bool find_win(win_id_t id, winlist_item_t &item);

// RMA_Lock_guard: RAII implementation of MPI_Win_lock/MPI_Win_unlock
class RMA_Lock_guard
{
public:
    RMA_Lock_guard() = delete;

    RMA_Lock_guard(int _rank, MPI_Win &_win): rank(_rank), win(_win)
    {
        // int myrank;
        // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        // std::cout << myrank << "R WIN " << (void*) &win << std::endl;

        // MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);
        MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, _win);
        unlock_func = [this](){ MPI_Win_unlock(rank, win); };
        locked = true;
    }

    RMA_Lock_guard(MPI_Win &_win): win(_win)
    {
        init_lock_all();
    }

    ~RMA_Lock_guard() {
        // std::cout << myrank << " ~RMA_Lock_guard()" << std::endl;
        if (locked)
            unlock();
    }

    void init_lock_all()
    {
        // win = _win;
        MPI_Win_lock_all(0, win);
        unlock_func = [this](){ MPI_Win_unlock_all(win); };
        locked = true;

        // auto myrank = 0;
        // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        // std::cerr << myrank << "R lockall " << (void *) &win << std::endl;
    }

    void init_lock_one(int _rank)
    {
        rank = _rank;
        // win = _win;
        MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);
        unlock_func = [this](){ MPI_Win_unlock(rank, win); };
        locked = true;

        // auto myrank = 0;
        // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        // std::cerr << myrank << "R lock " << (void *) &win << std::endl;
    }

    void unlock() {
        // auto myrank = 0;
        // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        // std::cout << myrank << "R unlock " << (void *) &win << std::endl;

        unlock_func();
        locked = false;
    }

private:
    int rank = -1;
    std::function<void()> unlock_func;
    MPI_Win &win;
    bool locked = false;
};

// Window type
enum wintype_t { common = 0, dynamic = 1, shmem = 2 };

// RMA_Win_guard: RAII implementation for RMA windows
template<typename T>
class RMA_Win_guard
{
public:

    // init: Initialize window: allocate memory and init win
    void init(unsigned int _count, MPI_Comm comm, wintype_t wintype = common)
    {
        count = _count;

        T *raw_ptr = nullptr;
        const auto bufsize = count * sizeof(T);

        std::cerr << myrank << "R b 1001\n";

        // Common window
        if (wintype == common) {
            MPI_Alloc_mem(bufsize, MPI_INFO_NULL, &raw_ptr);

            std::cerr << myrank << "R b 1002\n";

            MPI_Win_create(raw_ptr, bufsize, disp_unit, MPI_INFO_NULL,
                           comm, win.get());

            std::cerr << myrank << "R b 1003\n";

        // Shared memory window
        } else if (wintype == shmem) {
            MPI_Win_allocate_shared(bufsize, disp_unit, MPI_INFO_NULL,
                                    comm, raw_ptr, win.get());
                                     
        }

        if (raw_ptr == nullptr) {
            std::cerr << "MPI_Alloc_mem() failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Win_create(raw_ptr, bufsize, disp_unit, MPI_INFO_NULL,
                       comm, win.get());

        ptr = std::move(std::shared_ptr<T[]>(raw_ptr, 
                    [this](auto ptr) { ptr_deleter(ptr); }));

        std::cerr << myrank << "R b 1004\n";

        add_to_list();

        is_init = true;
        
        // Wait until window will be added on all procs
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // init: Initialize window: initialize memory in allocated memory
    void init(T *buf, unsigned int _count, MPI_Comm comm)
    {
        count = _count;

        MPI_Win_create(buf, count * sizeof(T), disp_unit, MPI_INFO_NULL,
                       comm, win.get());

        // Add deleter
        ptr = std::move(std::shared_ptr<T[]>(buf, 
                    [this](auto ptr) { ptr_deleter(ptr); }));

        // ptr = std::move(std::shared_ptr<T>(buf, 
        //             [this](auto ptr) { ptr_deleter(ptr); }));

        add_to_list();

        is_init = true;

        // Wait until window will be added on all procs
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // init: Initialize window: initialize memory in allocated memory
    void init(std::shared_ptr<T[]> _ptr, unsigned int _count, MPI_Comm comm)
    {
        count = _count;
        ptr = _ptr;

        MPI_Win_create(_ptr.get(), count * sizeof(T), disp_unit, MPI_INFO_NULL,
                       comm, win.get());

        add_to_list();

        is_init = true;
        
        // Wait until window will be added on all procs
        MPI_Barrier(MPI_COMM_WORLD);
    }

    RMA_Win_guard() { }

    RMA_Win_guard(unsigned int _count, MPI_Comm comm)
    {
        init(_count, comm);
    }

    RMA_Win_guard(T *buf, unsigned int _count, MPI_Comm comm) 
    {
        init(buf, _count, comm);
    }

    RMA_Win_guard(std::shared_ptr<T[]> _ptr, unsigned int _count, 
                  MPI_Comm comm) 
    {
        init(_ptr, _count, comm);
    }

    ~RMA_Win_guard()
    {
        free();
    }

    void free()
    {
        if (is_init == true) {
            auto rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::cerr << rank << "R RMADest " << id << std::endl;

            remove_from_list();

            MPI_Win_free(win.get());
            
            is_init = false;

            std::cerr << rank << "R RMADest " << id << " - ok" << std::endl;
        }
    }

    MPI_Win &get_win()
    {
        return *win;
    }

    MPI_Win *get_win_ptr()
    {
        return win.get();
    }

    T *get_ptr()
    {
        return ptr.get();
    }

    std::shared_ptr<T[]> get_sptr()
    {
        return ptr;
    }

    win_id_t get_id()
    {
        return id;
    }

    unsigned int get_count()
    {
        return count;
    }

private:
    void ptr_deleter(T *ptr)
    {
        // Debug -- begin
        auto rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << rank << "R deleter for ptr WID " << id << std::endl;

        MPI_Free_mem(ptr);

        std::cerr << rank << "R deleter for ptr WID " << id << " - ok" << std::endl;
    }

    // std::unique_ptr<MPI_Win> win = std::make_unique<MPI_Win>(MPI_WIN_NULL);

    std::shared_ptr<MPI_Win> win = std::make_shared<MPI_Win>(MPI_WIN_NULL);

    // std::shared_ptr<MPI_Win> win = std::shared_ptr<MPI_Win>(new MPI_Win,
    //          [](auto w) { 
    //             auto myrank = 0;
    //             MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //             std::cerr << myrank << "R WIN FREE\n";
    //             MPI_Win_free(w); 
    //             std::cerr << myrank << "R WIN FREE -- ok\n";
    //             delete w;
    //          });

    std::shared_ptr<T[]> ptr;

    unsigned int count = 0;

    // ID for list of windows
    win_id_t id;

    bool is_init = false;
   
    // Unit for displacements (1 byte)
    const int disp_unit = 1;

    // Add to the list of RMA windows
    void add_to_list()
    {
        std::lock_guard<std::mutex> lock(winlock);

        id = last_wid;
        last_wid++;
        
        // Debug -- begin
        // int myrank;
        // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // if ((id == 0)) {
        //     std::cout << myrank << "R last_wid " << last_wid << std::endl;
        //     std::cout << myrank << " INSERT " << id << " addr " 
        //               << (void *) win.get() << std::endl;
        // }
        // Debug -- end
        // std::cout << myrank << " INSERT " << id << " addr " 
        //           << (void *) win.get() << std::endl;

        // winlist.insert(std::pair<win_id_t, MPI_Win*>(id, win.get())); 

        winlist_item_t item;
        item.bufptr = ptr;
        item.win_sptr = win;

        winlist.insert(std::pair<win_id_t, winlist_item_t>(id, item)); 


        // Debug -- begin
        // for (auto &elem: winlist) {
        //     auto rank = 0;
        //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            // std::cout << rank << "R ELEM2 " << elem.first << std::endl;
        // }
        // Debug -- end
    }

    // Remove from the list of RMA windows
    void remove_from_list()
    {
        std::lock_guard<std::mutex> lock(winlock);

        // std::cout << myrank << " R REMOVE" << std::endl;

        // // Debug -- begin
        // auto rank = 0;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // std::cout << rank << "R REMOVE " << id << std::endl;
        // // Debug -- end

        winlist.erase(id);
    }
};

// template <typename T>
// win_id_t RMA_Win_guard<T>::last_id = 0;

