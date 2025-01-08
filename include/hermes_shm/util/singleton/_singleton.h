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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_

#include <memory>

#include "_easy_singleton.h"
#include "hermes_shm/constants/macros.h"

namespace hshm {

#ifdef HSHM_IS_HOST
/**
 * Makes a singleton. Constructs the first time GetInstance is called.
 * Requires user to define the static storage of obj_ in separate file.
 * @tparam T
 */
template <typename T, bool WithLock>
class SingletonBase {
 private:
  /** static instance. */
  HSHM_DLL_SINGLETON static char data_[sizeof(T)];
  HSHM_DLL_SINGLETON static T *obj_;
  HSHM_DLL_SINGLETON static hshm::SpinLock lock_;

 public:
  /** Get or create an instance of type T */
  inline static T *GetInstance() {
    if (!obj_) {
      if constexpr (WithLock) {
        hshm::ScopedSpinLock lock(lock_, 0);
        new ((T *)data_) T();
        obj_ = (T *)data_;
      } else {
        new ((T *)data_) T();
        obj_ = (T *)data_;
      }
    }
    return obj_;
  }
};
template <typename T, bool WithLock>
T *SingletonBase<T, WithLock>::obj_;
template <typename T, bool WithLock>
char SingletonBase<T, WithLock>::data_[sizeof(T)];
template <typename T, bool WithLock>
hshm::SpinLock SingletonBase<T, WithLock>::lock_;

template <typename T>
using Singleton = SingletonBase<T, true>;

template <typename T>
using LockfreeSingleton = SingletonBase<T, false>;

#define DEFINE_SINGLETON_BASE_CC(T, WithLock)                    \
  template <>                                                    \
  char hshm::SingletonBase<T, WithLock>::data_[sizeof(T)] = {0}; \
  template <>                                                    \
  T *hshm::SingletonBase<T, WithLock>::obj_ = nullptr;           \
  template <>                                                    \
  hshm::SpinLock hshm::SingletonBase<T, WithLock>::lock_ = hshm::SpinLock();

#define DEFINE_SINGLETON_CC(T) DEFINE_SINGLETON_BASE_CC(T, true)
#define DEFINE_LOCKFREE_SINGLETON_CC(T) DEFINE_SINGLETON_BASE_CC(T, false)

#else
// Regular singleton replace
template <typename T>
using Singleton = EasyLockfreeSingleton<T>;

// Lockfree singleton replace
template <typename T>
using LockfreeSingleton = EasyLockfreeSingleton<T>;

// Empty overrides for the defn macros
#define DEFINE_SINGLETON_CC(T)
#define DEFINE_LOCKFREE_SINGLETON_CC(T)
#endif

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_
