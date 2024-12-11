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

#ifndef HERMES_THREAD_THREAD_H_
#define HERMES_THREAD_THREAD_H_

#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include "hermes_shm/types/bitfield.h"
#include "hermes_shm/types/numbers.h"

#ifdef HERMES_ENABLE_PTHREADS
#include <omp.h>
#endif
#ifdef HERMES_RPC_THALLIUM
#include <thallium.hpp>
#endif
#ifdef HERMES_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace hshm {

/** Available threads that are mapped */
enum class ThreadType {
  kNone,
  kPthread,
  kArgobots,
  kCuda
};

}  // namespace hshm

namespace hshm::thread {
/** Thread-local storage */
class ThreadLocalData {
 public:
  // HSHM_CROSS_FUN
  // virtual void destroy() = 0;

  template<typename TLS>
  HSHM_CROSS_FUN
  static void destroy_wrap(void *data) {
    if (data) {
      static_cast<TLS *>(data)->destroy();
    }
  }
};

/** Thread-local key */
union ThreadLocalKey {
#ifdef HERMES_ENABLE_PTHREADS
  pthread_key_t pthread_key_;
#endif
#ifdef HERMES_RPC_THALLIUM
  ABT_key argobots_key_;
#endif
};

/** Represents the generic operations of a thread */
class ThreadModel {
 public:
  ThreadType type_;

 public:
  /** Initializer */
  HSHM_INLINE_CROSS
  ThreadModel(ThreadType type) : type_(type) {}

  /** Sleep thread for a period of time */
  HSHM_CROSS_FUN
  virtual void SleepForUs(size_t us) = 0;

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  virtual void Yield() = 0;

  /** Get the TID of the current thread */
  HSHM_CROSS_FUN
  virtual ThreadId GetTid() = 0;

  /** Get the thread model type */
  HSHM_INLINE_CROSS
  ThreadType GetType() {
    return type_;
  }
};

}  // namespace hshm::thread

#endif  // HERMES_THREAD_THREAD_H_
