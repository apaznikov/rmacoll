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

using win_id_t = unsigned int;

extern std::map<win_id_t, MPI_Win> winlist;

extern std::mutex winlock;

// RMA_Lock_guard: RAII implementation of MPI_Win_lock/MPI_Win_unlock
class RMA_Lock_guard
{
public:
    RMA_Lock_guard()
    {
    }

    RMA_Lock_guard(int _rank, MPI_Win _win)
    {
        init_lock_one(_rank, _win);
    }

    RMA_Lock_guard(MPI_Win _win)
    {
        init_lock_all(_win);
    }

    ~RMA_Lock_guard() {
        // std::cout << myrank << " ~RMA_Lock_guard()" << std::endl;
        if (locked)
            unlock();
    }

    void init_lock_all(MPI_Win _win)
    {
        win = _win;
        MPI_Win_lock_all(0, win);
        unlock_func = [this](){ MPI_Win_unlock_all(win); };
        locked = true;
    }

    void init_lock_one(int _rank, MPI_Win _win)
    {
        rank = _rank;
        win = _win;
        MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);
        unlock_func = [this](){ MPI_Win_unlock(rank, win); };
        locked = true;
    }

    void unlock() {
        unlock_func();
        locked = true;
    }

private:
    int rank = 0;
    std::function<void()> unlock_func;
    MPI_Win win;
    bool locked = false;
};

// RMA_Win_guard: RAII implementation for RMA windows
template<typename T>
class RMA_Win_guard
{
public:
    // init: Initialize window: allocate memory and init win
    void init(unsigned int count, MPI_Comm comm)
    {
        T *raw_ptr = nullptr;
        const auto bufsize = count * sizeof(T);

        MPI_Alloc_mem(bufsize, MPI_INFO_NULL, &raw_ptr);

        MPI_Win_create(raw_ptr, bufsize, disp_unit, MPI_INFO_NULL,
                       comm, &win);

        ptr = std::move(std::shared_ptr<T>(raw_ptr, 
                    [this](auto ptr) { ptr_deleter(ptr); }));

        add_to_list();

        is_init = true;
    }

    // init: Initialize window: initialize memory in allocated memory
    void init(T *buf, unsigned int count, MPI_Comm comm)
    {
        MPI_Win_create(buf, count * sizeof(T), disp_unit, MPI_INFO_NULL,
                       comm, &win);

        add_to_list();

        is_init = true;
    }

    RMA_Win_guard() { }

    RMA_Win_guard(unsigned int count, MPI_Comm comm)
    {
        init(count, comm);
    }

    RMA_Win_guard(T *buf, unsigned int count, MPI_Comm comm)
    {
        init(buf, count, comm);
    }

    ~RMA_Win_guard()
    {
        std::lock_guard<std::mutex> lock(winlock);

        remove_from_list();
        free();
    }

    void free()
    {
        if (is_init) {
            MPI_Win_free(&win);
        }
    }

    MPI_Win &get_win()
    {
        return win;
    }

    MPI_Win *get_win_ptr()
    {
        return &win;
    }

    T *get_ptr()
    {
        return ptr.get();
    }

    std::shared_ptr<T> get_sptr()
    {
        return ptr;
    }

private:
    static void ptr_deleter(T *ptr)
    {
        // std::cout << "deleter for ptr" << std::endl;
        MPI_Free_mem(ptr);
    }

    std::shared_ptr<T> ptr;
    MPI_Win win;

    // ID for list of windows
    win_id_t id;
    static win_id_t last_id;

    bool is_init = false;
   
    // Unit for displacements (1 byte)
    const int disp_unit = 1;

    // Add to the list of RMA windows
    void add_to_list()
    {
        std::lock_guard<std::mutex> lock(winlock);

        id = last_id;
        last_id++;
        winlist.insert(std::pair<win_id_t, MPI_Win>(id, win)); 
    }

    // Remove from the list of RMA windows
    void remove_from_list()
    {
        std::lock_guard<std::mutex> lock(winlock);

        winlist.erase(id);
    }
};

template <typename T>
win_id_t RMA_Win_guard<T>::last_id = 0;
