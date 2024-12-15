//
// Created by llogan on 26/10/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_LOCK_SPIN_LOCK_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_LOCK_SPIN_LOCK_H_

#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/numbers.h"

namespace hshm {

struct SpinLock {
  ipc::atomic<u32> lock_;
#ifdef HERMES_DEBUG_LOCK
  u32 owner_;
#endif

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  SpinLock() : lock_(0) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN
  SpinLock(const SpinLock &other) {}

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
#ifdef HERMES_MAKE_MUTEX
      HERMES_THREAD_MODEL->Yield();
#endif
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
#ifdef HERMES_DEBUG_LOCK
    owner_ = owner;
#endif
    return true;
  }

  /** Unlock */
  HSHM_INLINE_CROSS_FUN
  void Unlock() {
#ifdef HERMES_DEBUG_LOCK
    owner_ = 0;
#endif
    lock_.fetch_sub(1);
  }
};

struct ScopedSpinLock {
  SpinLock &lock_;
  bool is_locked_;

  /** Acquire the mutex */
  HSHM_INLINE_CROSS_FUN explicit ScopedSpinLock(SpinLock &lock, uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the mutex */
  HSHM_INLINE_CROSS_FUN
  ~ScopedSpinLock() { Unlock(); }

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

using hshm::ScopedSpinLock;
using hshm::SpinLock;

}  // namespace hshm::ipc

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_LOCK_SPIN_LOCK_H_
