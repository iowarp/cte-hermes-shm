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

#ifndef HERMES_THREAD_PTHREAD_H_
#define HERMES_THREAD_PTHREAD_H_

#include "thread_model.h"
#include <errno.h>
#include "hermes_shm/util/errors.h"
#include <omp.h>
#include "hermes_shm/introspect/system_info.h"

namespace hshm::thread_model {

class Pthread : public ThreadModel {
 public:
  /** Default constructor */
  Pthread() = default;

  /** Virtual destructor */
  virtual ~Pthread() = default;

  /** Yield the thread for a period of time */
  HSHM_CROSS_FUN
  void SleepForUs(size_t us) override {
#ifndef __CUDA_ARCH__
    usleep(us);
#endif
  }

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  void Yield() override {
#ifndef __CUDA_ARCH__
    sched_yield();
#endif
  }

  /** Get the TID of the current thread */
  tid_t GetTid() override {
    return (tid_t)omp_get_thread_num();
    // return static_cast<tid_t>(pthread_self());
  }
};

}  // namespace hshm::thread_model

#endif  // HERMES_THREAD_PTHREAD_H_
