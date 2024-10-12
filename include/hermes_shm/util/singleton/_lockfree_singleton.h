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

namespace hshm {

/**
 * Makes a singleton. Constructs the first time GetInstance is called.
 * Requires user to define the static storage of obj_ in separate file.
 * @tparam T
 */
template<typename T>
class LockfreeSingleton {
 private:
  static T *obj_;

 public:
  /** Get or create an instance of type T */
  HSHM_CROSS_FUN
  inline static T *GetInstance() {
    if (!obj_) {
      if (obj_ == nullptr) {
        obj_ = new T();
      }
    }
    return obj_;
  }
};

#define DEFINE_LOCKFREE_SINGLETON_CC(T)\
  template<> T* hshm::LockfreeSingleton<T>::obj_ = nullptr;

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_LOCKFREE_SINGLETON_SINGLETON_H_
