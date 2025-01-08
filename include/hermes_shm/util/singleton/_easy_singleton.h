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
template <typename T, bool WithLock>
class EasySingletonBase {
 public:
  HSHM_INLINE_CROSS_FUN
  static T *GetInstance() {
    if (GetObject() == nullptr) {
      if constexpr (WithLock) {
        hshm::ScopedSpinLock lock(GetSpinLock(), 0);
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      } else {
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      }
    }
    return GetObject();
  }

  HSHM_INLINE_CROSS_FUN
  static hshm::SpinLock &GetSpinLock() {
    static char spinlock_data_[sizeof(hshm::SpinLock)] = {0};
    return *(hshm::SpinLock *)spinlock_data_;
  }

  HSHM_INLINE_CROSS_FUN
  static T *GetData() {
    static char data_[sizeof(T)] = {0};
    return (T *)data_;
  }

  HSHM_INLINE_CROSS_FUN
  static T *&GetObject() {
    static T *obj_ = nullptr;
    return obj_;
  }
};

template <typename T>
using EasySingleton = EasySingletonBase<T, true>;

template <typename T>
using EasyLockfreeSingleton = EasySingletonBase<T, false>;

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_EASY_SINGLETON_H_
