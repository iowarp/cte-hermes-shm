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

#ifndef HSHM_THREAD_THREAD_H_
#define HSHM_THREAD_THREAD_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "hermes_shm/types/bitfield.h"
#include "hermes_shm/types/numbers.h"

#ifdef HSHM_ENABLE_PTHREADS
#include <pthread.h>
#endif
#ifdef HSHM_RPC_THALLIUM
#include <thallium.hpp>
#endif
#ifdef HSHM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef HSHM_ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace hshm {

/** Available threads that are mapped */
enum class ThreadType { kNone, kPthread, kArgobots, kCuda, kRocm, kWindows };

/** Thread-local key */
union ThreadLocalKey {
#ifdef HSHM_ENABLE_PTHREADS
  pthread_key_t pthread_key_;
#endif
#ifdef HSHM_RPC_THALLIUM
  ABT_key argobots_key_;
#endif
#ifdef HSHM_ENABLE_WINDOWS_THREADS
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
  explicit ThreadModel(ThreadType type) : type_(type) {}

  // /** Yield the current thread for a period of time */
  // HSHM_CROSS_FUN
  // void SleepForUs(size_t us);

  // /** Yield thread time slice */
  // HSHM_CROSS_FUN
  // void Yield();

  // /** Create thread-local storage */
  // template <typename TLS>
  // HSHM_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data);

  // /** Get thread-local storage */
  // template <typename TLS>
  // HSHM_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key);

  // /** Create thread-local storage */
  // template <typename TLS>
  // HSHM_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data);

  // /** Get the TID of the current thread */
  // HSHM_CROSS_FUN
  // ThreadId GetTid();

  /** Get the thread model type */
  HSHM_INLINE_CROSS_FUN
  ThreadType GetType() { return type_; }
};

}  // namespace hshm::thread

#endif  // HSHM_THREAD_THREAD_H_
