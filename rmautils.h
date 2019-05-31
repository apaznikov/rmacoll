//
// rmautils.cpp: Utils for MPI RMA
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#pragma once

#include <iostream>
#include <functional>
#include <memory>

#include <mpi.h>

extern int myrank;

// RMA_Lock_guard: RAII implementation of MPI_Win_lock/MPI_Win_unlock
class RMA_Lock_guard
{
public:
    RMA_Lock_guard(int _rank, MPI_Win &_win): rank(_rank), win(_win) {
        MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);
        unlock_func = [this](){ MPI_Win_unlock(rank, win); };
    }

    RMA_Lock_guard(MPI_Win &_win): win(_win)
    {
        MPI_Win_lock_all(0, win);
        unlock_func = [this](){ MPI_Win_unlock_all(win); };
    }

    ~RMA_Lock_guard() {
        // std::cout << myrank << " ~RMA_Lock_guard()" << std::endl;
        unlock_func();
    }

private:
    const int rank = 0;
    std::function<void()> unlock_func;
    MPI_Win &win;
};

// RMA_Win_guard: RAII implementation for RMA windows
template<typename T>
class RMA_Win_guard
{
public:
    // init: Initialize window: allocate memory and init win
    void init(unsigned int count)
    {
        T *raw_ptr = nullptr;
        const auto bufsize = count * sizeof(T);

        MPI_Alloc_mem(bufsize, MPI_INFO_NULL, &raw_ptr);

        MPI_Win_create(raw_ptr, bufsize, sizeof(int), MPI_INFO_NULL,
                       MPI_COMM_WORLD, &win);

        ptr = std::move(std::shared_ptr<T>(raw_ptr, 
                    [this](auto ptr) { ptr_deleter(ptr); }));
        is_init = true;
    }

    // init: Initialize window: initialize memory in allocated memory
    void init(T *buf, unsigned int count)
    {
        MPI_Win_create(buf, count * sizeof(T), sizeof(T), MPI_INFO_NULL,
                       MPI_COMM_WORLD, &win);
        is_init = true;
    }

    RMA_Win_guard() { }

    RMA_Win_guard(unsigned int count)
    {
        init(count);
    }

    RMA_Win_guard(T *buf, unsigned int count)
    {
        init(buf, count);
    }

    ~RMA_Win_guard()
    {
        if (is_init) {
            // std::cerr << myrank << " ~RMA_Win_guard()" << std::endl;
            MPI_Win_free(&win);
        }
    }

    MPI_Win &get_win()
    {
        return win;
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

    bool is_init = false;
};
