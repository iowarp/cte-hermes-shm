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
#include "_easy_lockfree_singleton.h"
#ifndef __CUDA_ARCH__
#include "hermes_shm/thread/lock/mutex.h"
#endif

namespace hshm {

#ifndef __CUDA_ARCH__
/**
 * Makes a singleton. Constructs the first time GetInstance is called.
 * Requires user to define the static storage of obj_ in separate file.
 * @tparam T
 */
template<typename T>
class Singleton {
 private:
  static T *obj_;
  static hshm::Mutex lock_;

 public:
  /** Get or create an instance of type T */
  inline static T *GetInstance() {
    if (!obj_) {
      hshm::ScopedMutex lock(lock_, 0);
      if (obj_ == nullptr) {
        obj_ = new T();
      }
    }
    return obj_;
  }
};

#define DEFINE_SINGLETON_CC(T)\
  template<> T* hshm::Singleton<T>::obj_ = nullptr; \
  template<> hshm::Mutex hshm::Singleton<T>::lock_ = hshm::Mutex();

#else
#include "_easy_lockfree_singleton.h"
template<typename T>
using Singleton = EasyLockfreeSingleton<T>;
#endif

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_
