//
// rmautils.cpp: Utils for MPI RMA
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include "rmautils.h"

// Lock for all not-threadsafe operations (add/remove from list, destructor)
std::mutex winlock;

std::map<win_id_t, MPI_Win> winlist;

bool find_win(win_id_t id, MPI_Win &win)
{
    std::lock_guard<std::mutex> lock(winlock);

    auto search = winlist.find(id);

    if (search != winlist.end()) {
        win = search->second;
        return true;
    } else {
        return false;
    }
}
