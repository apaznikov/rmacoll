//
// broadcast_linear.cpp: Linear algorithm for RMA broadcast
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#include <mpi.h>

#include "rmacoll.h"
#include "broadcast_linear.h"
#include "rmautils.h"

// RMA_Bcast_linear: linear broadcast algorithm
int RMA_Bcast_linear(const void *origin_addr, int origin_count, 
                     MPI_Datatype origin_datatype, MPI_Aint target_disp,
                     int target_count, MPI_Datatype target_datatype,
                     win_id_t wid, MPI_Comm comm)
{
    auto myrank = 0, nproc = 0; 

    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nproc);

    // Search for bcast window id in the windows list
    winlist_item_t bcast_winlist_item;
    auto isfound = find_win(wid, bcast_winlist_item);

    if (isfound == false) {
        std::cerr << "Window " << wid << " was not found" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, RET_CODE_ERROR);
    }

    RMA_Lock_guard lock_all(*bcast_winlist_item.win_sptr);

    for (auto rank = 0; rank < nproc; rank++) {
        MPI_Put(origin_addr, origin_count, origin_datatype, rank, target_disp, 
                target_count, target_datatype, *bcast_winlist_item.win_sptr);
    }

    return RET_CODE_SUCCESS;
}
