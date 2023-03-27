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

#ifndef HERMES_INCLUDE_HERMES_UTIL_AUTO_TRACE_H_
#define HERMES_INCLUDE_HERMES_UTIL_AUTO_TRACE_H_

#include "formatter.h"
#include "timer.h"
#include <iostream>

namespace hermes_shm {

#define AUTO_TRACE(LOG_LEVEL) \
  hermes_shm::AutoTrace<LOG_LEVEL> hshm_tracer_(__PRETTY_FUNCTION__);

#define TIMER_START() \
  hshm_tracer_.StartTimer(__PRETTY_FUNCTION__);

#define TIMER_END()  \
  hshm_tracer_.EndTimer();

/** Trace function execution times */
template<int LOG_LEVEL>
class AutoTrace {
 private:
  HighResMonotonicTimer timer_;
  HighResMonotonicTimer timer2_;
  std::string fname_;

 public:
  template<typename ...Args>
  explicit AutoTrace(const char *fname) : fname_(fname) {
    _StartTimer(timer_, fname);
  }

  ~AutoTrace() {
    _EndTimer(timer_);
  }

  void StartTimer(const char *fname) {
    _StartTimer(timer2_, fname);
  }

  void EndTimer() {
    _EndTimer(timer2_);
  }

 private:
  void _StartTimer(HighResMonotonicTimer &timer, const char *fname) {
#ifdef HERMES_ENABLE_PROFILING
    if constexpr(LOG_LEVEL <= HERMES_ENABLE_PROFILING) {
      timer.Resume();
    }
#endif
  }

  void _EndTimer(HighResMonotonicTimer &timer) {
#ifdef HERMES_ENABLE_PROFILING
    if constexpr(LOG_LEVEL <= HERMES_ENABLE_PROFILING) {
      timer.Pause();
      std::cout << hermes_shm::Formatter::format("{};{}ns\n",
                                                 fname_,
                                                 timer_.GetNsec());
      timer.Reset();
    }
#endif
  }
};

}  // namespace hermes_shm

#endif  // HERMES_INCLUDE_HERMES_UTIL_AUTO_TRACE_H_
