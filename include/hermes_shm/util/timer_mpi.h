//
// Created by lukemartinlogan on 10/8/23.
//

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
  MpiTimer(int comm) : comm_(comm) {
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
