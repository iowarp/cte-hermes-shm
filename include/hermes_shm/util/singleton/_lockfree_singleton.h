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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_LOCKFREE_SINGLETON_SINGLETON_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_LOCKFREE_SINGLETON_SINGLETON_H_

#include <memory>
#include "hermes_shm/constants/macros.h"
#include "_easy_lockfree_singleton.h"

namespace hshm {

#ifndef __CUDA_ARCH__
/**
 * Makes a singleton. Constructs the first time GetInstance is called.
 * Requires user to define the static storage of obj_ in separate file.
 * @tparam T
 */
template<typename T>
class LockfreeSingleton {
 private:
#ifndef __CUDA_ARCH__
  static char data_[sizeof(T)];
  static T *obj_;
#else
  __device__ static char data_[sizeof(T)];
  __device__ static T *obj_;
#endif

 public:
  /** Get or create an instance of type T */
  inline static T* GetInstance() {
    if (!obj_) {
      if (obj_ == nullptr) {
        obj_ = (T*)data_;
        new (obj_) T();
      }
    }
    return obj_;
  }
};

#define DEFINE_LOCKFREE_SINGLETON_CC(T)\
  template<> char hshm::LockfreeSingleton<T>::data_[sizeof(T)] = {0}; \
  template<> T* hshm::LockfreeSingleton<T>::obj_ = nullptr;

#else
#include "_easy_lockfree_singleton.h"
template<typename T>
using LockfreeSingleton = EasyLockfreeSingleton<T>;
#define DEFINE_LOCKFREE_SINGLETON_CC(T)
#endif

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_LOCKFREE_SINGLETON_SINGLETON_H_
