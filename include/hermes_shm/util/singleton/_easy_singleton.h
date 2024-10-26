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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_EASY_SINGLETON_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_EASY_SINGLETON_H_

#include <memory>
#include "hermes_shm/constants/macros.h"
#include "hermes_shm/thread/lock/spin_lock.h"

namespace hshm {

/**
 * A class to represent singleton pattern
 * Does not require specific initialization of the static variable
 * */
template<typename T, bool WithLock>
class EasySingletonBase {
 protected:
 public:
  /**
   * Uses unique pointer to build a static global instance of variable.
   * @tparam T
   * @return instance of T
   */
  template<typename ...Args>
  HSHM_CROSS_FUN
  static T* GetInstance(Args&& ...args) {
    static char data_[sizeof(T)] = {0};
    static char spinlock_data_[sizeof(hshm::SpinLock)] = {0};
    static T* obj_ = nullptr;

    if (obj_ == nullptr) {
      if constexpr (WithLock) {
        hshm::SpinLock *lock_ = (hshm::SpinLock *) spinlock_data_;
        hshm::ScopedSpinLock lock(*lock_, 0);
        ConstructInstance(
            data_, obj_,
            std::forward<Args>(args)...);
      } else {
        ConstructInstance(
            data_, obj_,
            std::forward<Args>(args)...);
      }
    }
    return obj_;
  }

  template<typename ...Args>
  HSHM_CROSS_FUN
  static void ConstructInstance(
      char *data_,
      T *obj_,
      Args&& ...args) {
    if (obj_ == nullptr) {
      new((T *) data_) T(std::forward<Args>(args)...);
      obj_ = (T *) data_;
    }
  }
};

template<typename T>
using EasySingleton = EasySingletonBase<T, true>;

template<typename T>
using EasyLockfreeSingleton = EasySingletonBase<T, false>;

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_EASY_SINGLETON_H_
