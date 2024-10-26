#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/numbers.h"

/**
 * The EasySingleton used for the HERMES_THREAD_MODEL_MANAGER
 * macro needs a SpinLock. This file specifically disables
 * the yielding even being included.
 * */
#ifdef HERMES_MAKE_MUTEX
// Make a mutex with HERMES_THREAD_MODEL_MANAGER->Yield()
#include "hermes_shm/thread/thread_model_manager.h"
#define MUTEX_CLASS_NAME Mutex
#define SCOPED_MUTEX_CLASS_NAME ScopedMutex
#else
// Make a spinlock without HERMES_THREAD_MODEL_MANAGER->Yield()
#define MUTEX_CLASS_NAME SpinLock
#define SCOPED_MUTEX_CLASS_NAME ScopedSpinLock
#endif

namespace hshm {

struct MUTEX_CLASS_NAME {
  ipc::atomic<u32> lock_;
#ifdef HERMES_DEBUG_LOCK
  u32 owner_;
#endif

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  MUTEX_CLASS_NAME() : lock_(0) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN
  MUTEX_CLASS_NAME(const MUTEX_CLASS_NAME &other) {}

  /** Explicit initialization */
  HSHM_INLINE_CROSS_FUN
  void Init() {
    lock_ = 0;
  }

  /** Acquire lock */
  HSHM_INLINE_CROSS_FUN
  void Lock(uint32_t owner) {
    do {
      for (int i = 0; i < 1; ++i) {
        if (TryLock(owner)) { return; }
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

struct SCOPED_MUTEX_CLASS_NAME {
  MUTEX_CLASS_NAME &lock_;
  bool is_locked_;

  /** Acquire the mutex */
  HSHM_INLINE_CROSS_FUN explicit
  SCOPED_MUTEX_CLASS_NAME(MUTEX_CLASS_NAME &lock,
                          uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the mutex */
  HSHM_INLINE_CROSS_FUN
  ~SCOPED_MUTEX_CLASS_NAME() {
    Unlock();
  }

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

#undef MUTEX_CLASS_NAME
#undef SCOPED_MUTEX_CLASS_NAME