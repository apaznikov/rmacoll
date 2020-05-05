//
// broadcast_linear.h: Linear algorithm for RMA broadcast
//
// (C) 2019 Alexey Paznikov <apaznikov@gmail.com> 
//

#pragma once

#include <mpi.h>

#include "rmautils.h"

// RMA_Bcast_linear: linear broadcast algorithm
int RMA_Bcast_linear(const void *origin_addr, int origin_count, 
                     MPI_Datatype origin_datatype, MPI_Aint target_disp,
                     int target_count, MPI_Datatype target_datatype,
                     win_id_t wid, MPI_Comm comm);
