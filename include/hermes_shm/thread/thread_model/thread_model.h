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

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

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
#ifdef HERMES_ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace hshm {

/** Available threads that are mapped */
enum class ThreadType { kNone, kPthread, kArgobots, kCuda, kRocm, kWindows };

/** Thread-local key */
union ThreadLocalKey {
#ifdef HERMES_ENABLE_PTHREADS
  pthread_key_t pthread_key_;
#endif
#ifdef HERMES_RPC_THALLIUM
  ABT_key argobots_key_;
#endif
#ifdef HERMES_ENABLE_WINDOWS_THREADS
  DWORD windows_key_;
#endif
};

}  // namespace hshm

namespace hshm::thread {

/** Thread-local key */
using hshm::ThreadLocalKey;

/** Thread-local storage */
class ThreadLocalData {
 public:
  // HSHM_CROSS_FUN
  // void destroy() = 0;

  template <typename TLS>
  HSHM_CROSS_FUN static void destroy_wrap(void *data) {
    if (data) {
      if constexpr (std::is_base_of_v<ThreadLocalData, TLS>) {
        static_cast<TLS *>(data)->destroy();
      }
    }
  }
};

/** Represents the generic operations of a thread */
class ThreadModel {
 public:
  ThreadType type_;

 public:
  /** Initializer */
  HSHM_INLINE_CROSS_FUN
  ThreadModel(ThreadType type) : type_(type) {}

  // /** Sleep thread for a period of time */
  // HSHM_CROSS_FUN
  // void SleepForUs(size_t us) = 0;

  // /** Yield thread time slice */
  // HSHM_CROSS_FUN
  // void Yield() = 0;

  // /** Get the TID of the current thread */
  // HSHM_CROSS_FUN
  // ThreadId GetTid() = 0;

  /** Get the thread model type */
  HSHM_INLINE_CROSS_FUN
  ThreadType GetType() { return type_; }
};

}  // namespace hshm::thread

#endif  // HERMES_THREAD_THREAD_H_
