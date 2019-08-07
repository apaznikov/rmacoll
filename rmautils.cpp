//
// rmautils.cpp: Utils for MPI RMA
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include "rmautils.h"

// Lock for all not-threadsafe operations (add/remove from list, destructor)
std::mutex winlock;

winlist_t winlist;

// The last window id to make it unique
win_id_t last_wid = 0;

// Find window by window's id
bool find_win(win_id_t id, winlist_item_t &item)
{
    std::lock_guard<std::mutex> lock(winlock);

    // auto rank = 0;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // for (auto &elem: winlist) {
    //     std::cout << rank << "R ELEM " << elem.first 
    //               << " win " << (void*) elem.second.get() << std::endl;
    // }

    auto search = winlist.find(id);

    if (search != winlist.end()) {
        item = search->second;

        // *win = search->second;
        // std::cout << rank << "R RETURN " << (void*) search->second << std::endl;
        return true;
    } else {
        return false;
    }
}
