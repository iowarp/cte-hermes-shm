//
// Created by lukemartinlogan on 10/8/23.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_THREAD_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_THREAD_H_

#include "timer.h"
#include "omp.h"

namespace hshm {

class ThreadTimer : public NsecTimer {
 public:
  int rank_;
  int nprocs_;
  std::vector<Timer> timers_;

 public:
  ThreadTimer(int nthreads) {
    nprocs_ = nthreads;
    timers_.resize(nprocs_);
  }

  void SetRank(int rank) {
    rank_ = rank;
  }

  void Resume() {
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

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_TIMER_THREAD_H_
