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

#ifndef HSHM_THREAD_MUTEX_H_
#define HSHM_THREAD_MUTEX_H_

#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/numbers.h"

namespace hshm {

struct Mutex {
  ipc::atomic<u32> lock_;
#ifdef HSHM_DEBUG_LOCK
  u32 owner_;
#endif

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Mutex() : lock_(0) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN
  Mutex(const Mutex &other) {}

  /** Explicit initialization */
  HSHM_INLINE_CROSS_FUN
  void Init() { lock_ = 0; }

  /** Acquire lock */
  HSHM_INLINE_CROSS_FUN
  void Lock(uint32_t owner) {
    do {
      for (int i = 0; i < 1; ++i) {
        if (TryLock(owner)) {
          return;
        }
      }
      HSHM_THREAD_MODEL->Yield();
    } while (true);
  }

  /** Try to acquire the lock */
  HSHM_INLINE_CROSS_FUN
  bool TryLock(uint32_t owner) {
    if (lock_.load() != 0) {
      return false;
    }
    uint32_t tkt = lock_.fetch_add(1);
    if (tkt != 0) {
      lock_.fetch_sub(1);
      return false;
    }
#ifdef HSHM_DEBUG_LOCK
    owner_ = owner;
#endif
    return true;
  }

  /** Unlock */
  HSHM_INLINE_CROSS_FUN
  void Unlock() {
#ifdef HSHM_DEBUG_LOCK
    owner_ = 0;
#endif
    lock_.fetch_sub(1);
  }
};

struct ScopedMutex {
  Mutex &lock_;
  bool is_locked_;

  /** Acquire the mutex */
  HSHM_INLINE_CROSS_FUN explicit ScopedMutex(Mutex &lock, uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the mutex */
  HSHM_INLINE_CROSS_FUN
  ~ScopedMutex() { Unlock(); }

  /** Explicitly acquire the mutex */
  HSHM_INLINE_CROSS_FUN
  void Lock(uint32_t owner) {
    if (!is_locked_) {
      lock_.Lock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly try to lock the mutex */
  HSHM_INLINE_CROSS_FUN
  bool TryLock(uint32_t owner) {
    if (!is_locked_) {
      is_locked_ = lock_.TryLock(owner);
    }
    return is_locked_;
  }

  /** Explicitly unlock the mutex */
  HSHM_INLINE_CROSS_FUN
  void Unlock() {
    if (is_locked_) {
      lock_.Unlock();
      is_locked_ = false;
    }
  }
};

}  // namespace hshm

namespace hshm::ipc {

using hshm::Mutex;
using hshm::ScopedMutex;

}  // namespace hshm::ipc

#undef Mutex
#undef ScopedMutex

#endif  // HSHM_THREAD_MUTEX_H_
