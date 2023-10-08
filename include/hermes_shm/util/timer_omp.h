//
// Created by lukemartinlogan on 10/8/23.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_OMP_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_OMP_H_

#include "timer.h"
#include "omp.h"

namespace hshm {

class OmpTimer : public NsecTimer {
 public:
  int rank_;
  int nprocs_;
  std::vector<Timer> timers_;

 public:
  OmpTimer() {
    rank_ = omp_get_thread_num();
    nprocs_ = omp_get_num_threads();
    timers_.resize(nprocs_);
  }

  void Start() {
    timers_[rank_].Resume();
  }

  void Pause() {
    timers_[rank_].Pause();
  }

  void Reset() {
    timers_[rank_].Reset();
  }

  void Collect() {
    std::vector<double> rank_times;
    rank_times.reserve(nprocs_);
    for (Timer &t : timers_) {
      rank_times.push_back(t.GetNsec());
    }
    time_ns_ = *std::max_element(rank_times.begin(), rank_times.end());
  }
};


}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_OMP_H_
