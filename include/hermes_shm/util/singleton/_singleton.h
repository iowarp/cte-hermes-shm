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
template<typename T>
class Singleton {
 private:
  /** static instance. */
  HSHM_CROSS_VAR static char data_[sizeof(T)];
  HSHM_CROSS_VAR static T* obj_;
  HSHM_CROSS_VAR static hshm::SpinLock lock_;

 public:
  /** Get or create an instance of type T */
  template<typename ...Args>
  inline static T *GetInstance(Args&& ...args) {
    if (!obj_) {
      hshm::ScopedSpinLock lock(lock_, 0);
      ConstructInstance(std::forward<Args>(args)...);
    }
    return obj_;
  }

  /** Construct the instance */
  template<typename ...Args>
  HSHM_CROSS_FUN
  static void ConstructInstance(Args&& ...args) {
    if (obj_ == nullptr) {
      obj_ = (T *) data_;
      new(obj_) T(std::forward<Args>(args)...);
    }
  }
};

#define DEFINE_SINGLETON_CC(T)\
  template<> char hshm::Singleton<T>::data_[sizeof(T)] = {0}; \
  template<> T* hshm::Singleton<T>::obj_ = nullptr; \
  template<> hshm::SpinLock hshm::Singleton<T>::lock_ = hshm::SpinLock();

#else
template<typename T>
using Singleton = EasySingleton<T>;
#define DEFINE_SINGLETON_CC(T)
#endif

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_
