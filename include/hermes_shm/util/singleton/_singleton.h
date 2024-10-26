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
#include "hermes_shm/constants/macros.h"
#include "_easy_singleton.h"

namespace hshm {

#ifndef __CUDA_ARCH__
/**
 * Makes a singleton. Constructs the first time GetInstance is called.
 * Requires user to define the static storage of obj_ in separate file.
 * @tparam T
 */
template<typename T, bool WithLock>
class SingletonBase {
 private:
  /** static instance. */
  static char data_[sizeof(T)];
  static T* obj_;
  static hshm::SpinLock lock_;

 public:
  /** Get or create an instance of type T */
  template<typename ...Args>
  inline static T *GetInstance(Args&& ...args) {
    if (!obj_) {
      if constexpr (WithLock) {
        hshm::ScopedSpinLock lock(lock_, 0);
        ConstructInstance(std::forward<Args>(args)...);
      } else {
        ConstructInstance(std::forward<Args>(args)...);
      }
    }
    return obj_;
  }

  /** Construct the instance */
  template<typename ...Args>
  static void ConstructInstance(Args&& ...args) {
    if (obj_ == nullptr) {
      new((T *) data_) T(std::forward<Args>(args)...);
      obj_ = (T *) data_;
    }
  }
};

template<typename T>
using Singleton = SingletonBase<T, true>;

template<typename T>
using LockfreeSingleton = SingletonBase<T, false>;

#define DEFINE_SINGLETON_CC(T)\
  template<> char hshm::SingletonBase<T, true>::data_[sizeof(T)] = {0}; \
  template<> T* hshm::SingletonBase<T, true>::obj_ = nullptr; \
  template<> hshm::SpinLock hshm::SingletonBase<T, true>::lock_ = hshm::SpinLock();

#define DEFINE_LOCKFREE_SINGLETON_CC(T)\
  template<> char hshm::SingletonBase<T, false>::data_[sizeof(T)] = {0}; \
  template<> T* hshm::SingletonBase<T, false>::obj_ = nullptr; \
  template<> hshm::SpinLock hshm::SingletonBase<T, false>::lock_ = hshm::SpinLock();

#else
// Regular singleton replace
template<typename T>
using Singleton = EasyLockfreeSingleton<T>;

// Lockfree singleton replace
template<typename T>
using LockfreeSingleton = EasyLockfreeSingleton<T>;

// Empty overrides for the defn macros
#define DEFINE_SINGLETON_CC(T)
#define DEFINE_LOCKFREE_SINGLETON_CC(T)
#endif

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_
