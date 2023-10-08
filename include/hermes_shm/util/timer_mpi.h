/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_MPI_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_MPI_H_

#include "timer.h"
#include <mpi.h>

namespace hshm {

class MpiTimer : public Timer {
 public:
  int rank_;
  int comm_;
  int nprocs_;

 public:
  explicit MpiTimer(int comm) : comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);
  }

  void Collect() {
    MPI_Barrier(comm_);
    double my_nsec = GetNsec();
    MPI_Reduce(&my_nsec, &time_ns_, 1,
               MPI_DOUBLE, MPI_MAX,
               0, comm_);
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_MPI_H_
