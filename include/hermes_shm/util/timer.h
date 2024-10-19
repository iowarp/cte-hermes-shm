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

#ifndef HERMES_TIMER_H
#define HERMES_TIMER_H

#include <chrono>
#include <algorithm>
#include <vector>
#include <functional>
#include "hermes_shm/constants/macros.h"

namespace hshm {

template<typename T>
class TimepointBase {
 public:
  std::chrono::time_point<T> start_;

 public:
  HSHM_ALWAYS_INLINE void Now() {
    start_ = T::now();
  }
  HSHM_ALWAYS_INLINE double GetNsecFromStart(TimepointBase &now) const {
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.start_ - start_).count();
    return elapsed;
  }
  HSHM_ALWAYS_INLINE double GetUsecFromStart(TimepointBase &now) const {
    return GetNsecFromStart(now)/1000;
  }
  HSHM_ALWAYS_INLINE double GetMsecFromStart(TimepointBase &now) const {
    return GetNsecFromStart(now)/1000000;
  }
  HSHM_ALWAYS_INLINE double GetSecFromStart(TimepointBase &now) const {
    return GetNsecFromStart(now)/1000000000;
  }
  HSHM_ALWAYS_INLINE double GetNsecFromStart() const {
    std::chrono::time_point<T> end_ = T::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_ - start_).count();
    return elapsed;
  }
  HSHM_ALWAYS_INLINE double GetUsecFromStart() const {
    return GetNsecFromStart()/1000;
  }
  HSHM_ALWAYS_INLINE double GetMsecFromStart() const {
    return GetNsecFromStart()/1000000;
  }
  HSHM_ALWAYS_INLINE double GetSecFromStart() const {
    return GetNsecFromStart()/1000000000;
  }
};

class NsecTimer {
 public:
  double time_ns_;

 public:
  NsecTimer() : time_ns_(0) {}

  HSHM_ALWAYS_INLINE double GetNsec() const {
    return time_ns_;
  }
  HSHM_ALWAYS_INLINE double GetUsec() const {
    return time_ns_/1000;
  }
  HSHM_ALWAYS_INLINE double GetMsec() const {
    return time_ns_/1000000;
  }
  HSHM_ALWAYS_INLINE double GetSec() const {
    return time_ns_/1000000000;
  }
  HSHM_ALWAYS_INLINE void Reset() {
    time_ns_ = 0;
  }
};

template<typename T>
class TimerBase : public TimepointBase<T>, public NsecTimer {
 private:
  std::chrono::time_point<T> end_;

 public:
  TimerBase() = default;

  HSHM_ALWAYS_INLINE void Resume() {
    TimepointBase<T>::Now();
  }
  HSHM_ALWAYS_INLINE double Pause() {
    time_ns_ += TimepointBase<T>::GetNsecFromStart();
    return time_ns_;
  }
  HSHM_ALWAYS_INLINE double GetUsFromEpoch() const {
    std::chrono::time_point<std::chrono::system_clock> point =
        std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        point.time_since_epoch()).count();
  }
};

typedef TimerBase<std::chrono::high_resolution_clock> HighResCpuTimer;
typedef TimerBase<std::chrono::steady_clock> HighResMonotonicTimer;
typedef HighResMonotonicTimer Timer;
typedef TimepointBase<std::chrono::high_resolution_clock> HighResCpuTimepoint;
typedef TimepointBase<std::chrono::steady_clock> HighResMonotonicTimepoint;
typedef HighResMonotonicTimepoint Timepoint;

}  // namespace hshm

#endif  // HERMES_TIMER_H
